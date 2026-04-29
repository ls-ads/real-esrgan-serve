"""Multi-(GPU-pool × image-flavor) sweep.

For each (gpu_class, flavor) pair: deploy a serverless endpoint, run
all three workloads (cold_start, batch_sweep, sustained_concurrent),
tear the endpoint down. Each run writes one row to `runs` plus N rows
to `jobs` / `items` / `telemetry` in the SQLite results DB.

Why a sweep and not 10 manual `runner.py` invocations: deploys can
fail mid-run (RunPod capacity churn, host-driver issues). The sweep
catches those and continues, so a 2-hour run isn't lost to one bad
GPU class. Each (gpu, flavor) failure is logged + recorded as a
status='FAILED' run; the analysis report ignores them automatically.

Pools we test (5, after collapsing the user's 12 GPU classes by
RunPod's pool grouping — see `build/runpod_deploy.py`):

  ADA_24       — rtx-4090     (24 GB Ada)
  ADA_48_PRO   — l40s         (48 GB Ada Pro)
  AMPERE_24    — rtx-3090     (24 GB Ampere; groups L4 + A5000 too)
  AMPERE_48    — a40          (48 GB Ampere; groups A6000)
  AMPERE_16    — a4000        (16 GB; mixed Ampere + Ada)

Blackwell (rtx-5090, sm120) is excluded — TRT 10.1 doesn't have sm120
codegen. See the `project_blackwell_unsupported` memory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Local imports — run with `python3 -m build.bench.sweep` from the
# repo root so the package layout resolves. The `build` directory is
# already on sys.path because we're inside it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from build import runpod_deploy as rpd  # noqa: E402
from build.bench import runner, schema, workloads  # noqa: E402


# Each entry: (gpu_class, flavor, image_tag, max_batch).
# max_batch tunes the batch_sweep ceiling per VRAM tier so the 16 GB
# pool doesn't OOM at b=64.
def _build_matrix(image_tag_cuda: str, image_tag_trt: str) -> list[tuple]:
    pools = [
        # gpu_class, max_batch (bounded by smallest VRAM in pool)
        ("rtx-4090", 32),  # ADA_24
        ("l40s",     64),  # ADA_48_PRO
        ("rtx-3090", 32),  # AMPERE_24
        ("a40",      64),  # AMPERE_48
        ("a4000",    16),  # AMPERE_16 (16 GB floor)
    ]
    matrix = []
    for gpu_class, max_batch in pools:
        matrix.append((gpu_class, "cuda", image_tag_cuda, max_batch))
        matrix.append((gpu_class, "trt",  image_tag_trt,  max_batch))
    return matrix


def _deploy(client: rpd.RunPodClient, gpu_class: str, image: str,
            endpoint_name: str, auth_id: Optional[str]) -> str:
    """Create endpoint via runpod_deploy.deploy_endpoint(). Returns
    the endpoint id. Re-uses the deploy module so we benefit from its
    validated template + endpoint mutations + idempotency."""
    pool = rpd.GPU_CLASS_TO_POOL.get(gpu_class)
    if not pool:
        raise SystemExit(f"unknown gpu_class for pool lookup: {gpu_class}")
    args = Namespace(
        image=image,
        gpu_class=gpu_class,
        endpoint_name=endpoint_name,
        workers_max=2,
        idle_timeout=10,    # short — we want cold workers between runs
        container_disk_gb=10,
    )
    return rpd.deploy_endpoint(client, args, auth_id, pool)


def _run_one(api_key: str, *, endpoint_id: str, gpu_class: str,
             flavor: str, image_tag: str, max_batch: int,
             db_path: Path,
             workload_names: Optional[list[str]] = None) -> None:
    """Run the requested workloads against an endpoint, in order. Each
    workload is its own row in `runs`; failures don't stop the next
    workload (we want partial data on bad runs).

    `workload_names`=None runs the standard trio (cold_start, batch_sweep,
    sustained_concurrent). Pass a subset to scope a re-sweep, e.g.
    ["image_size_sweep"] to only run the resolution scan."""
    sm_arch = schema.GPU_CLASS_TO_SM.get(gpu_class)
    common = dict(
        endpoint_id=endpoint_id,
        api_key=api_key,
        flavor=flavor,
        image_tag=image_tag,
        gpu_class=gpu_class,
        sm_arch=sm_arch,
        db_path=db_path,
    )

    if workload_names is None:
        ws = [
            workloads.workload_cold_start(),
            workloads.workload_batch_sweep(max_batch=max_batch),
            workloads.workload_sustained_concurrent(concurrency=4,
                                                    jobs_per_worker=4),
        ]
    else:
        ws = []
        for name in workload_names:
            if name == "cold_start":
                ws.append(workloads.workload_cold_start())
            elif name == "batch_sweep":
                ws.append(workloads.workload_batch_sweep(max_batch=max_batch))
            elif name == "sustained_concurrent":
                ws.append(workloads.workload_sustained_concurrent(
                    concurrency=4, jobs_per_worker=4))
            elif name == "image_size_sweep":
                ws.append(workloads.workload_image_size_sweep())
            else:
                raise SystemExit(f"unknown workload: {name}")

    for workload in ws:
        try:
            runner.run_workload(workload=workload, **common)
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] workload {workload.name} on "
                  f"{flavor}/{gpu_class} failed: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image-cuda",
                   default="ghcr.io/ls-ads/real-esrgan-serve:runpod-cuda-dev")
    p.add_argument("--image-trt",
                   default="ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-dev")
    p.add_argument("--db-path", default=str(schema.DEFAULT_DB_PATH))
    p.add_argument("--only-gpu", action="append", default=None,
                   help="restrict to a subset of gpu_classes (repeatable)")
    p.add_argument("--only-flavor", action="append", default=None,
                   choices=["cuda", "trt"],
                   help="restrict to a subset of flavors (repeatable)")
    p.add_argument("--workloads", action="append", default=None,
                   choices=["cold_start", "batch_sweep",
                            "sustained_concurrent", "image_size_sweep"],
                   help="restrict to a subset of workloads per pair "
                        "(repeatable). Default = standard trio.")
    args = p.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not set; use build/.with-iosuite-key", file=sys.stderr)
        return 1
    rp = rpd.RunPodClient(api_key)
    auth_id = rp.find_registry_auth_for("ghcr") if (
        args.image_cuda.startswith("ghcr.io/") or args.image_trt.startswith("ghcr.io/")
    ) else None

    matrix = _build_matrix(args.image_cuda, args.image_trt)
    if args.only_gpu:
        matrix = [m for m in matrix if m[0] in args.only_gpu]
    if args.only_flavor:
        matrix = [m for m in matrix if m[1] in args.only_flavor]

    db_path = Path(args.db_path)
    started = datetime.now(timezone.utc)
    print(f"[sweep] {started.isoformat()} — {len(matrix)} (gpu, flavor) "
          f"pairs to test", file=sys.stderr)

    for i, (gpu_class, flavor, image_tag, max_batch) in enumerate(matrix, 1):
        endpoint_name = f"bench-{flavor}-{gpu_class}-{int(time.time())}"
        endpoint_id: Optional[str] = None
        print(f"\n[sweep] === {i}/{len(matrix)} {flavor}/{gpu_class} === "
              f"image={image_tag}", file=sys.stderr)
        try:
            endpoint_id = _deploy(rp, gpu_class, image_tag, endpoint_name, auth_id)
            print(f"[sweep] endpoint id: {endpoint_id}", file=sys.stderr)
            # Brief pause: workers idle out (idle_timeout=10), so the
            # cold_start workload's first job actually hits a cold spawn.
            time.sleep(15)
            _run_one(api_key, endpoint_id=endpoint_id, gpu_class=gpu_class,
                     flavor=flavor, image_tag=image_tag, max_batch=max_batch,
                     db_path=db_path, workload_names=args.workloads)
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] {flavor}/{gpu_class} FAILED: {e}", file=sys.stderr)
        finally:
            if endpoint_id:
                try:
                    rp.delete_endpoint(endpoint_id)
                except Exception as e:  # noqa: BLE001
                    print(f"[sweep] WARN: tear-down for {endpoint_id} failed: {e}. "
                          f"Check the RunPod console!", file=sys.stderr)

    print(f"\n[sweep] done. Run "
          f"`python3 -m build.bench.report --db-path {db_path}` for analysis.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
