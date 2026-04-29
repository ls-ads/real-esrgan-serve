"""Read-only analysis CLI over the benchmark results DB.

Surfaces three views by default:

  - Cost-per-image leaderboard (the "optimal GPU" question, ranked
    by USD/image at flex pricing).
  - Cold-start latency by (gpu_class, flavor).
  - GPU utilisation summary — flags workloads that don't saturate.

Plus a `--query <name>` form that runs any of the canonical queries
in `schema.QUERIES` and prints the full result set.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from . import schema, spend


def _print_table(rows: list[sqlite3.Row], cols: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(c), max(len(str(r[c])) if r[c] is not None else 4 for r in rows))
              for c in cols]
    sep = "  "
    print(sep.join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(
            (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) if r[c] is not None else "—").ljust(w)
            for c, w in zip(cols, widths)
        ))


def cmd_summary(conn: sqlite3.Connection) -> None:
    """One-screen overview: counts, then leaderboard, then telemetry."""
    cur = conn.execute(
        "SELECT COUNT(*) FROM runs WHERE finished_at_utc IS NOT NULL"
    )
    print(f"Completed runs: {cur.fetchone()[0]}")
    cur = conn.execute(
        "SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY COUNT(*) DESC"
    )
    for status, n in cur.fetchall():
        print(f"  jobs[{status}]: {n}")

    print("\n=== cost per image (warm, flex pricing) ===")
    rows = list(conn.execute(schema.QUERIES["cost_per_image"]))
    if rows:
        cols = ["gpu_class", "flavor", "workload",
                "s_per_image", "usd_per_image_flex", "flex_usd_per_s"]
        _print_table([dict(zip([d[0] for d in conn.execute(schema.QUERIES["cost_per_image"]).description], r)) for r in rows], cols)

    print("\n=== cold-start latency ===")
    rows = list(conn.execute(schema.QUERIES["cold_start_ms"]))
    if rows:
        cols = ["gpu_class", "flavor", "cold_start_ms_min", "cold_start_ms_avg"]
        _print_table([dict(zip([d[0] for d in conn.execute(schema.QUERIES["cold_start_ms"]).description], r)) for r in rows], cols)

    print("\n=== telemetry summary ===")
    rows = list(conn.execute(schema.QUERIES["telemetry_summary"]))
    if rows:
        cols = ["gpu_class", "flavor", "workload",
                "avg_gpu_util", "peak_gpu_util",
                "peak_vram_mb", "peak_temp_c"]
        _print_table([dict(zip([d[0] for d in conn.execute(schema.QUERIES["telemetry_summary"]).description], r)) for r in rows], cols)

    # Spend reconciliation — only show if there's any snapshot data,
    # so older runs (predating the snapshot wiring) don't print empty
    # tables.
    cur = conn.execute("SELECT COUNT(*) FROM cost_snapshots WHERE balance_usd IS NOT NULL")
    if cur.fetchone()[0] > 0:
        print("\n=== spend per pair (measured) ===")
        cur = conn.execute(spend.RECONCILE_QUERIES["spend_per_pair"])
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        _print_table(rows, cols)

        print("\n=== predicted vs measured ===")
        cur = conn.execute(spend.RECONCILE_QUERIES["predicted_vs_measured"])
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        _print_table(rows, cols)


def cmd_query(conn: sqlite3.Connection, name: str) -> None:
    queries = {**schema.QUERIES, **spend.RECONCILE_QUERIES}
    if name not in queries:
        print(f"unknown query '{name}'. available: {list(queries.keys())}",
              file=sys.stderr)
        sys.exit(2)
    cur = conn.execute(queries[name])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    _print_table(rows, cols)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db-path", default=str(schema.DEFAULT_DB_PATH))
    p.add_argument("--query", default=None,
                   help=f"named query: {list(schema.QUERIES.keys())}")
    args = p.parse_args(argv)

    conn = schema.open_db(Path(args.db_path))
    schema.init_schema(conn)
    if args.query:
        cmd_query(conn, args.query)
    else:
        cmd_summary(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
