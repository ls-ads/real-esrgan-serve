"""Benchmark runner: submit workload jobs to a RunPod endpoint, capture
client-side timing + worker-side telemetry, and write everything to
the SQLite results DB.

Design choices:

  - /runsync: we use RunPod's synchronous endpoint rather than /run +
    poll. The blocking call simplifies timing — submitted-at and
    completed-at brackets are clean. Synchronous timeout is per-job
    (we set 600 s for cold starts, 120 s for warm jobs), and the
    whole benchmark runs in front of a long-lived endpoint anyway,
    so blocking is fine.

  - One thread per concurrency slot: sustained_concurrent workloads
    use a small ThreadPoolExecutor. Threading is the right primitive
    here because the work is HTTP-bound, not CPU-bound — adding asyncio
    + aiohttp would buy nothing for this load.

  - Cold-start handling: we don't try to GUARANTEE a cold worker
    via the endpoint API (RunPod doesn't expose a "kill workers"
    knob). Instead we mark the FIRST job of a run as cold_start=1
    (and the runner recommends idling for several minutes between
    runs, or recreating the endpoint via the deploy script). When
    a worker is actually warm at submission, the cold_start metric
    just becomes a warm number — analysis filters by walltime
    distribution to spot misclassified rows.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import schema, spend, workloads
from .workloads import JobSpec, Workload, png_for_size

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _runpod_request(url: str, api_key: str, payload: Optional[dict] = None,
                    timeout_s: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "real-esrgan-serve-bench/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _submit_async(endpoint_id: str, api_key: str, payload: dict,
                  timeout_s: int = 600, poll_interval_s: float = 1.0) -> dict:
    """POST /run, then poll /status until COMPLETED or FAILED. Used in
    place of /runsync because runsync has a ~90s server-side timeout
    (cold starts exceed that). The polling adds a small client-side
    overhead but is the only way to wait reliably for long jobs."""
    submit_url = f"{RUNPOD_API_BASE}/{endpoint_id}/run"
    submit = _runpod_request(submit_url, api_key, payload)
    runpod_job_id = submit.get("id")
    if not runpod_job_id:
        return {"status": "FAILED", "error": f"submit returned no id: {submit}"}

    status_url = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{runpod_job_id}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = _runpod_request(status_url, api_key)
        except urllib.error.URLError:
            time.sleep(poll_interval_s)
            continue
        if resp.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
            return resp
        time.sleep(poll_interval_s)
    return {"status": "TIMEOUT", "id": runpod_job_id,
            "error": f"client timeout after {timeout_s}s"}


def _record_job(conn: sqlite3.Connection, *,
                run_id: str, spec: JobSpec, payload: dict,
                resp: dict, walltime_ms: float, status: str,
                error: Optional[str]) -> str:
    """Persist one job submission to the DB. Returns the local job_id."""
    job_id = str(uuid.uuid4())
    runpod_job_id = (resp.get("id") if isinstance(resp, dict) else None)
    out = (resp.get("output") or {}) if isinstance(resp, dict) else {}
    diag = out.get("_diagnostics") or {}

    conn.execute(
        """INSERT INTO jobs
           (job_id, run_id, runpod_job_id, submitted_at_utc, completed_at_utc,
            walltime_ms, delay_ms, exec_ms,
            batch_size, input_bytes_total, cold_start, status, error,
            providers_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id, run_id, runpod_job_id,
            _utcnow_iso(), _utcnow_iso(),
            walltime_ms,
            (resp.get("delayTime") if isinstance(resp, dict) else None),
            (resp.get("executionTime") if isinstance(resp, dict) else None),
            spec.batch_size,
            len(json.dumps(payload).encode()),
            1 if spec.cold_start else 0,
            status,
            error,
            json.dumps(diag.get("providers")) if diag.get("providers") else None,
        ),
    )

    # Per-image rows. The handler echoes input_resolution +
    # output_resolution + exec_ms per item.
    for idx, item in enumerate(out.get("outputs") or []):
        try:
            iw, ih = (int(x) for x in item.get("input_resolution", "0x0").split("x"))
            ow, oh = (int(x) for x in item.get("output_resolution", "0x0").split("x"))
        except (ValueError, AttributeError):
            iw = ih = ow = oh = None
        conn.execute(
            """INSERT INTO items (job_id, idx, input_w, input_h,
                                  output_w, output_h, exec_ms,
                                  output_bytes, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, idx, iw, ih, ow, oh,
             item.get("exec_ms"),
             item.get("output_size_bytes"),
             item.get("error")),
        )

    # Telemetry samples (only present when spec.telemetry=True).
    tele = diag.get("telemetry") or {}
    for sample in tele.get("samples") or []:
        conn.execute(
            """INSERT OR IGNORE INTO telemetry
               (job_id, t_ms_offset, gpu_util_pct, mem_util_pct,
                vram_used_mb, vram_total_mb, gpu_temp_c)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, sample.get("t_ms"),
             sample.get("gpu_util_pct"), sample.get("mem_util_pct"),
             sample.get("vram_used_mb"), sample.get("vram_total_mb"),
             sample.get("gpu_temp_c")),
        )

    conn.commit()
    return job_id


def run_workload(*, endpoint_id: str, api_key: str,
                 flavor: str, image_tag: str, gpu_class: str,
                 sm_arch: Optional[str],
                 workload: Workload,
                 db_path: Path = schema.DEFAULT_DB_PATH,
                 notes: Optional[str] = None,
                 sweep_id: Optional[str] = None,
                 record_balance: bool = True) -> str:
    """Execute a workload against a deployed endpoint. Writes a `runs`
    row plus N `jobs` rows to the DB. Returns the run_id.

    When `record_balance=True` (default), captures balance snapshots
    at run-start and run-end so the report can reconcile predicted vs
    measured spend. Set False for tests / dry runs that shouldn't
    pollute the snapshot timeline."""
    conn = schema.open_db(db_path)
    schema.init_schema(conn)
    spend.init_schema(conn)

    run_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO runs (run_id, started_at_utc, flavor, image_tag,
                             gpu_class, sm_arch, endpoint_id, workload,
                             params_json, notes, sweep_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, _utcnow_iso(), flavor, image_tag, gpu_class, sm_arch,
         endpoint_id, workload.name,
         json.dumps({"description": workload.description,
                     "concurrency": workload.concurrency,
                     "spec_count": len(workload.specs)}),
         notes, sweep_id),
    )
    conn.commit()

    if record_balance:
        spend.record_snapshot(conn, api_key=api_key, phase="run_start",
                              sweep_id=sweep_id,
                              pair_label=f"{flavor}/{gpu_class}",
                              run_id=run_id)

    print(f"[bench] run_id={run_id} workload={workload.name} "
          f"flavor={flavor} gpu={gpu_class} jobs={len(workload.specs)}",
          file=sys.stderr)

    def _run_spec(spec: JobSpec) -> None:
        payload = spec.make_payload(png_for_size)
        t0 = time.monotonic()
        try:
            timeout = 600 if spec.cold_start else 180
            resp = _submit_async(endpoint_id, api_key, payload, timeout_s=timeout)
            walltime_ms = (time.monotonic() - t0) * 1000
            status = resp.get("status", "UNKNOWN")
            err = (resp.get("output") or {}).get("error") or resp.get("error")
            with _db_lock:
                _record_job(conn, run_id=run_id, spec=spec, payload=payload,
                            resp=resp, walltime_ms=walltime_ms, status=status,
                            error=err)
            providers = (((resp.get("output") or {}).get("_diagnostics") or {}).get("providers")) or []
            print(f"[bench] {spec.name} batch={spec.batch_size} "
                  f"walltime={walltime_ms:.0f}ms exec={resp.get('executionTime')}ms "
                  f"status={status} EPs={providers}",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            walltime_ms = (time.monotonic() - t0) * 1000
            with _db_lock:
                _record_job(conn, run_id=run_id, spec=spec, payload=payload,
                            resp={}, walltime_ms=walltime_ms,
                            status="FAILED", error=str(e))
            print(f"[bench] {spec.name} FAILED after {walltime_ms:.0f}ms: {e}",
                  file=sys.stderr)

    _db_lock = threading.Lock()
    if workload.concurrency <= 1:
        for spec in workload.specs:
            _run_spec(spec)
    else:
        # Warmup spec runs serially first; the rest go through the pool.
        warmup = next((s for s in workload.specs if s.cold_start), None)
        body = [s for s in workload.specs if not s.cold_start]
        if warmup:
            _run_spec(warmup)
        with ThreadPoolExecutor(max_workers=workload.concurrency) as ex:
            futures = [ex.submit(_run_spec, s) for s in body]
            for f in as_completed(futures):
                f.result()

    conn.execute("UPDATE runs SET finished_at_utc = ? WHERE run_id = ?",
                 (_utcnow_iso(), run_id))
    conn.commit()

    if record_balance:
        spend.record_snapshot(conn, api_key=api_key, phase="run_end",
                              sweep_id=sweep_id,
                              pair_label=f"{flavor}/{gpu_class}",
                              run_id=run_id)
    conn.close()
    print(f"[bench] run {run_id} done.", file=sys.stderr)
    return run_id


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def _build_workload(name: str, image_w: int, image_h: int,
                    max_batch: int, concurrency: int,
                    jobs_per_worker: int) -> Workload:
    if name == "cold_start":
        return workloads.workload_cold_start()
    if name == "batch_sweep":
        return workloads.workload_batch_sweep(max_batch=max_batch,
                                              image_w=image_w, image_h=image_h)
    if name == "sustained_concurrent":
        return workloads.workload_sustained_concurrent(
            concurrency=concurrency, jobs_per_worker=jobs_per_worker,
            image_w=image_w, image_h=image_h,
        )
    if name == "image_size_sweep":
        return workloads.workload_image_size_sweep()
    raise SystemExit(f"unknown workload: {name}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--endpoint-id", required=True,
                   help="RunPod serverless endpoint id (or set RUNPOD_ENDPOINT_ID)")
    p.add_argument("--flavor", required=True, choices=["cpu", "cuda", "trt"])
    p.add_argument("--image-tag", required=True,
                   help="docker image deployed to the endpoint, for the runs table")
    p.add_argument("--gpu-class", required=True,
                   help="kebab-case GPU class (rtx-4090, l40s, …) for cost lookup")
    p.add_argument("--workload", required=True,
                   choices=list(workloads.WORKLOADS.keys()))
    p.add_argument("--image-w", type=int, default=workloads.DEFAULT_IMAGE_W)
    p.add_argument("--image-h", type=int, default=workloads.DEFAULT_IMAGE_H)
    p.add_argument("--max-batch", type=int, default=64,
                   help="ceiling for batch_sweep; smaller VRAM tiers should "
                        "lower this to avoid OOM")
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel client threads for sustained_concurrent")
    p.add_argument("--jobs-per-worker", type=int, default=8)
    p.add_argument("--db-path", default=str(schema.DEFAULT_DB_PATH))
    p.add_argument("--notes", default=None)
    args = p.parse_args(argv)

    import os
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        return _err("RUNPOD_API_KEY not set; use build/.with-iosuite-key")

    workload = _build_workload(args.workload, args.image_w, args.image_h,
                               args.max_batch, args.concurrency,
                               args.jobs_per_worker)
    sm_arch = schema.GPU_CLASS_TO_SM.get(args.gpu_class)

    run_workload(
        endpoint_id=args.endpoint_id,
        api_key=api_key,
        flavor=args.flavor,
        image_tag=args.image_tag,
        gpu_class=args.gpu_class,
        sm_arch=sm_arch,
        workload=workload,
        db_path=Path(args.db_path),
        notes=args.notes,
    )
    return 0


def _err(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
