"""SQLite schema for benchmark results + the GPU-pricing reference table.

We use plain SQLite (stdlib only) over Postgres / DuckDB / etc because:

  - Self-contained: a single .db file in the repo (gitignored) is the
    canonical location, no external service needed.
  - Survives across runs: results from a sweep last week are still
    available for re-analysis without re-paying for GPU time.
  - SQL is the right query language for the "find optimal GPU"
    derivation — the pricing table joins cleanly against the per-job
    timings.

Schema is deliberately denormalised in places (e.g. gpu_class is
copied into `jobs` rather than hung off `runs`) so ad-hoc SELECTs
during analysis don't require multi-table joins.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Pricing rows the user supplied. Pulled into the DB at init so the
# cost-per-image derivation is a single SQL JOIN rather than a Python
# dict lookup. Tier names are RunPod's marketing labels; "flex" is
# spot pricing, "active" is reserved (we use flex throughout).
#
# All units: $/second of wall-clock GPU time.
_GPU_PRICING_ROWS = [
    # (gpu_class, tier_name, vram_gb, flex_usd_per_s, active_usd_per_s)
    ("l40",          "48GB Pro",  48, 0.00053, 0.00045),
    ("l40s",         "48GB Pro",  48, 0.00053, 0.00045),
    ("rtx-6000",     "48GB Pro",  48, 0.00053, 0.00045),
    ("a6000",        "48GB",      48, 0.00034, 0.00029),
    ("a40",          "48GB",      48, 0.00034, 0.00029),
    ("rtx-5090",     "32GB Pro",  32, 0.00044, 0.00037),
    ("rtx-4090",     "24GB Pro",  24, 0.00031, 0.00026),
    ("l4",           "24GB",      24, 0.00019, 0.00016),
    ("a5000",        "24GB",      24, 0.00019, 0.00016),
    ("rtx-3090",     "24GB",      24, 0.00019, 0.00016),
    ("a4000",        "16GB",      16, 0.00016, 0.00014),
    ("a4500",        "16GB",      16, 0.00016, 0.00014),
    ("rtx-4000",     "16GB",      16, 0.00016, 0.00014),
    ("rtx-2000",     "16GB",      16, 0.00016, 0.00014),
]

# Mapping from gpu_class to its CUDA SM arch — used by the benchmark
# runner to pick which engine artefact to expect at the worker.
GPU_CLASS_TO_SM = {
    "l40": "sm89", "l40s": "sm89", "rtx-6000": "sm89", "rtx-4000": "sm89",
    "rtx-4090": "sm89", "l4": "sm89",
    "a6000": "sm86", "a40": "sm86", "a5000": "sm86", "a4000": "sm86",
    "a4500": "sm86", "rtx-3090": "sm86", "rtx-2000": "sm86",
    "rtx-5090": "sm120",
}

DEFAULT_DB_PATH = Path("build/bench/results.db")


def open_db(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because the bench runner submits jobs from
    # a ThreadPoolExecutor (sustained_concurrent workload). The runner
    # serialises writes through its own _db_lock, so SQLite's
    # thread-affinity check would be a false positive.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent — safe to call on every run. Adds tables if they
    don't exist; existing data is preserved."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS gpu_pricing (
      gpu_class TEXT PRIMARY KEY,
      tier_name TEXT NOT NULL,
      vram_gb INTEGER NOT NULL,
      flex_usd_per_s REAL NOT NULL,
      active_usd_per_s REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY,
      started_at_utc TEXT NOT NULL,
      finished_at_utc TEXT,
      flavor TEXT NOT NULL CHECK (flavor IN ('cpu', 'cuda', 'trt')),
      image_tag TEXT NOT NULL,
      gpu_class TEXT,
      sm_arch TEXT,
      endpoint_id TEXT,
      workload TEXT NOT NULL,
      params_json TEXT,
      notes TEXT
    );
    CREATE INDEX IF NOT EXISTS runs_by_workload ON runs (workload, gpu_class, flavor);

    -- Per-job (i.e. per RunPod request) record. A request may carry
    -- multiple images — see batch_size + sum-of-exec_ms across items.
    CREATE TABLE IF NOT EXISTS jobs (
      job_id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      runpod_job_id TEXT,
      submitted_at_utc TEXT NOT NULL,
      completed_at_utc TEXT,
      walltime_ms REAL,        -- client-observed: submit → final response
      delay_ms REAL,           -- runpod's own queue+spawn measurement
      exec_ms REAL,            -- runpod's own handler-exec measurement
      batch_size INTEGER NOT NULL,
      input_bytes_total INTEGER,
      cold_start INTEGER NOT NULL DEFAULT 0,  -- 1 = first job of run
      status TEXT NOT NULL,    -- COMPLETED | FAILED | TIMEOUT
      error TEXT,
      providers_json TEXT,     -- JSON array of active EPs
      FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS jobs_by_run ON jobs (run_id, submitted_at_utc);

    -- One row per image inside a batch. Keeps per-image timings
    -- separate from the request-level walltime so we can compare
    -- batch=1 vs batch=N exec patterns directly.
    CREATE TABLE IF NOT EXISTS items (
      job_id TEXT NOT NULL,
      idx INTEGER NOT NULL,
      input_w INTEGER,
      input_h INTEGER,
      output_w INTEGER,
      output_h INTEGER,
      exec_ms REAL,            -- handler-side per-image timing
      output_bytes INTEGER,    -- when discard_output=true, just the size
      error TEXT,
      PRIMARY KEY (job_id, idx),
      FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
    );

    -- Worker-side telemetry samples. One row per nvidia-smi snapshot
    -- captured during a job. t_ms_offset is relative to job start.
    CREATE TABLE IF NOT EXISTS telemetry (
      job_id TEXT NOT NULL,
      t_ms_offset INTEGER NOT NULL,
      gpu_util_pct INTEGER,
      mem_util_pct INTEGER,
      vram_used_mb INTEGER,
      vram_total_mb INTEGER,
      gpu_temp_c INTEGER,
      PRIMARY KEY (job_id, t_ms_offset),
      FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
    );
    """)

    # Refresh pricing rows. UPSERT in case the user ever passes new
    # rates — pricing changes don't invalidate prior runs (each run
    # records its own `flex_usd_per_s` snapshot via the JOIN below).
    conn.executemany(
        """INSERT INTO gpu_pricing (gpu_class, tier_name, vram_gb, flex_usd_per_s, active_usd_per_s)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(gpu_class) DO UPDATE SET
             tier_name=excluded.tier_name,
             vram_gb=excluded.vram_gb,
             flex_usd_per_s=excluded.flex_usd_per_s,
             active_usd_per_s=excluded.active_usd_per_s
        """,
        _GPU_PRICING_ROWS,
    )
    conn.commit()


# ────────────────────────────────────────────────────────────────────
# Reference views surfaced at init time. Created via plain SELECT
# rather than CREATE VIEW because SQLite re-resolves view definitions
# on each query and we'd rather hand-tune the SQL when the schema
# evolves. These docstrings double as analyst documentation.
# ────────────────────────────────────────────────────────────────────

QUERIES = {
    "throughput_per_run": """
        -- Effective images-per-second per run, ignoring failed jobs.
        SELECT
          r.run_id, r.flavor, r.gpu_class, r.workload,
          SUM(j.batch_size) * 1000.0 / SUM(j.walltime_ms) AS images_per_sec,
          AVG(j.walltime_ms) AS avg_walltime_ms,
          COUNT(*) AS jobs,
          SUM(j.batch_size) AS images
        FROM runs r
        JOIN jobs j ON j.run_id = r.run_id
        WHERE j.status = 'COMPLETED' AND j.cold_start = 0
        GROUP BY r.run_id
    """,

    "cost_per_image": """
        -- Cost in USD to upscale one image at this GPU's flex price,
        -- using the run's average warm walltime. Cold starts excluded.
        SELECT
          r.gpu_class, r.flavor, r.workload,
          AVG(j.walltime_ms) / 1000.0 AS avg_walltime_s,
          AVG(j.walltime_ms) / 1000.0 / NULLIF(AVG(j.batch_size), 0) AS s_per_image,
          (AVG(j.walltime_ms) / 1000.0 / NULLIF(AVG(j.batch_size), 0))
            * p.flex_usd_per_s AS usd_per_image_flex,
          p.flex_usd_per_s
        FROM runs r
        JOIN jobs j ON j.run_id = r.run_id
        JOIN gpu_pricing p ON p.gpu_class = r.gpu_class
        WHERE j.status = 'COMPLETED' AND j.cold_start = 0
        GROUP BY r.gpu_class, r.flavor, r.workload
        ORDER BY usd_per_image_flex
    """,

    "cold_start_ms": """
        -- The "first image" metric: walltime from submit to response
        -- on a fresh worker (no warm helper).
        SELECT
          r.gpu_class, r.flavor,
          MIN(j.walltime_ms) AS cold_start_ms_min,
          AVG(j.walltime_ms) AS cold_start_ms_avg
        FROM runs r
        JOIN jobs j ON j.run_id = r.run_id
        WHERE j.cold_start = 1 AND j.status = 'COMPLETED'
        GROUP BY r.gpu_class, r.flavor
    """,

    "telemetry_summary": """
        -- Average GPU utilisation + peak VRAM by run. Used to spot
        -- workloads that don't saturate the GPU (cost left on the
        -- table) or saturate VRAM (next-up batch sizes will OOM).
        SELECT
          r.gpu_class, r.flavor, r.workload,
          AVG(t.gpu_util_pct) AS avg_gpu_util,
          MAX(t.gpu_util_pct) AS peak_gpu_util,
          MAX(t.vram_used_mb) AS peak_vram_mb,
          MAX(t.gpu_temp_c) AS peak_temp_c
        FROM runs r
        JOIN jobs j ON j.run_id = r.run_id
        JOIN telemetry t ON t.job_id = j.job_id
        WHERE j.status = 'COMPLETED'
        GROUP BY r.run_id
    """,
}
