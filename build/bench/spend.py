"""RunPod balance + spend reconciliation.

We persist balance snapshots into the SQLite results DB so each run
records (a) the per-second-rate cost predicted by our pricing table
joined against measured wall-time, and (b) the actual spend reported
by RunPod's account ledger over the same window. The two should
agree — if they drift more than a few %, either the rate table is
out of date or something about our cost model (maybe missing a
fee class, or RunPod's pool tier shifted) is wrong.

Snapshots are taken at meaningful phase boundaries:

  sweep_start         — before any deploys
  pair_start          — before deploying this (gpu, flavor) endpoint
  pair_after_deploy   — endpoint up + cold-start measured. Delta
                        from pair_start = startup spend (image pull,
                        worker spawn, first-job warmup).
  pair_after_workload — all benchmark workloads completed. Delta
                        from pair_after_deploy = workload spend
                        (steady-state inference + queue idle time).
  pair_end            — after teardown. Delta from
                        pair_after_workload should be near zero
                        (RunPod stops billing on delete).
  sweep_end           — final snapshot.

Spend attribution = balance_before − balance_after for each window.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def fetch_balance(api_key: str) -> Optional[dict]:
    """Snapshot of RunPod account balance + current spend rate.
    Returns None on any error (we never want a failed snapshot to
    abort a benchmark run — the data is supplementary, not load-
    bearing).

    Returned dict: {balance_usd, spend_per_hr_usd, ts_utc}."""
    body = json.dumps({"query":
        "{ myself { clientBalance currentSpendPerHr } }"}).encode()
    req = urllib.request.Request(
        RUNPOD_GRAPHQL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # UA header required — Cloudflare blocks Python's default
            # (see reference_runpod_graphql memory).
            "User-Agent": "real-esrgan-bench-spend/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    me = (resp.get("data") or {}).get("myself") or {}
    if "clientBalance" not in me:
        return None
    return {
        "balance_usd": float(me.get("clientBalance") or 0.0),
        "spend_per_hr_usd": float(me.get("currentSpendPerHr") or 0.0),
        "ts_utc": _utc_now_iso(),
    }


def init_schema(conn: sqlite3.Connection) -> None:
    """Add the cost_snapshots table. Idempotent."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cost_snapshots (
      snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_utc TEXT NOT NULL,
      sweep_id TEXT,                     -- groups all snapshots from one sweep
      pair_label TEXT,                   -- e.g. "trt/rtx-4090" — null at sweep level
      run_id TEXT,                       -- runs.run_id when phase is run-bound
      phase TEXT NOT NULL,
      balance_usd REAL,
      spend_per_hr_usd REAL,
      notes TEXT,
      FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL
    );
    CREATE INDEX IF NOT EXISTS cost_by_sweep ON cost_snapshots (sweep_id, ts_utc);
    """)
    conn.commit()


def record_snapshot(conn: sqlite3.Connection, *,
                    api_key: str, phase: str,
                    sweep_id: Optional[str] = None,
                    pair_label: Optional[str] = None,
                    run_id: Optional[str] = None,
                    notes: Optional[str] = None) -> Optional[dict]:
    """Fetch + persist a snapshot. Returns the snapshot dict or None
    on RunPod-side failure (still logged with a note so attribution
    queries can spot the gap)."""
    snap = fetch_balance(api_key)
    if snap is None:
        # Record the gap so analysis can flag it
        conn.execute(
            """INSERT INTO cost_snapshots
               (ts_utc, sweep_id, pair_label, run_id, phase,
                balance_usd, spend_per_hr_usd, notes)
               VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)""",
            (_utc_now_iso(), sweep_id, pair_label, run_id, phase,
             "fetch_balance returned None"),
        )
        conn.commit()
        return None
    conn.execute(
        """INSERT INTO cost_snapshots
           (ts_utc, sweep_id, pair_label, run_id, phase,
            balance_usd, spend_per_hr_usd, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (snap["ts_utc"], sweep_id, pair_label, run_id, phase,
         snap["balance_usd"], snap["spend_per_hr_usd"], notes),
    )
    conn.commit()
    return snap


# Reconciliation queries — surface in report.py as canonical views.
RECONCILE_QUERIES = {
    "spend_per_pair": """
        -- Measured spend per (gpu, flavor) pair, broken into
        -- startup vs workload phases. NEGATIVE deltas are a balance
        -- top-up landing mid-sweep (very unusual; flag in analysis).
        WITH paired AS (
          SELECT pair_label, sweep_id,
            MAX(CASE WHEN phase = 'pair_start'          THEN balance_usd END) AS bal_start,
            MAX(CASE WHEN phase = 'pair_after_deploy'   THEN balance_usd END) AS bal_after_deploy,
            MAX(CASE WHEN phase = 'pair_after_workload' THEN balance_usd END) AS bal_after_workload,
            MAX(CASE WHEN phase = 'pair_end'            THEN balance_usd END) AS bal_end
          FROM cost_snapshots
          WHERE pair_label IS NOT NULL
          GROUP BY pair_label, sweep_id
        )
        SELECT pair_label, sweep_id,
          ROUND(bal_start - bal_after_deploy, 6)   AS spend_startup,
          ROUND(bal_after_deploy - bal_after_workload, 6) AS spend_workload,
          ROUND(bal_after_workload - bal_end, 6)   AS spend_teardown,
          ROUND(bal_start - bal_end, 6)            AS spend_total,
          bal_start, bal_end
        FROM paired
        ORDER BY sweep_id, pair_label
    """,

    "predicted_vs_measured": """
        -- Per-pair predicted vs measured spend, scoped to the LATEST
        -- sweep so old runs don't pollute the comparison. Predicted =
        -- this-sweep's runs' walltime × pricing. Measured = balance
        -- delta over the workload window (between pair_after_deploy
        -- and pair_after_workload snapshots) for the SAME sweep.
        WITH latest_sweep AS (
          SELECT sweep_id
          FROM cost_snapshots
          WHERE sweep_id IS NOT NULL
          ORDER BY ts_utc DESC LIMIT 1
        ),
        per_pair_walltime AS (
          SELECT r.gpu_class, r.flavor,
            SUM(j.walltime_ms) / 1000.0 AS total_walltime_s
          FROM runs r
          JOIN jobs j ON j.run_id = r.run_id
          JOIN latest_sweep ls ON ls.sweep_id = r.sweep_id
          WHERE j.status = 'COMPLETED'
          GROUP BY r.gpu_class, r.flavor
        ),
        measured AS (
          SELECT
            SUBSTR(cs.pair_label, 1, INSTR(cs.pair_label,'/')-1) AS flavor,
            SUBSTR(cs.pair_label, INSTR(cs.pair_label,'/')+1)    AS gpu_class,
            SUM(CASE WHEN cs.phase = 'pair_after_deploy'   THEN -balance_usd
                     WHEN cs.phase = 'pair_after_workload' THEN balance_usd
                     END) AS measured_workload_spend
          FROM cost_snapshots cs
          JOIN latest_sweep ls ON ls.sweep_id = cs.sweep_id
          WHERE cs.pair_label IS NOT NULL
          GROUP BY cs.pair_label
        )
        SELECT w.gpu_class, w.flavor,
          ROUND(w.total_walltime_s, 1) AS walltime_s,
          ROUND(w.total_walltime_s * p.flex_usd_per_s, 6) AS predicted_flex,
          ROUND(m.measured_workload_spend, 6) AS measured,
          ROUND(m.measured_workload_spend
                / NULLIF(w.total_walltime_s * p.flex_usd_per_s, 0), 3) AS ratio
        FROM per_pair_walltime w
        LEFT JOIN measured m ON m.gpu_class = w.gpu_class AND m.flavor = w.flavor
        LEFT JOIN gpu_pricing p ON p.gpu_class = w.gpu_class
        ORDER BY w.gpu_class, w.flavor
    """,
}
