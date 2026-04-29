# Benchmark harness

Drive workloads against a deployed RunPod endpoint, capture client-side
timings + worker-side telemetry, and persist everything to a SQLite
database for later analysis.

## What it measures

Three workload patterns:

| Workload | What it answers |
|---|---|
| `cold_start` | First-image latency on a freshly-spawned worker. |
| `batch_sweep` | Throughput vs batch size — at what batch size does request overhead become invisible? |
| `sustained_concurrent` | Steady-state throughput under N parallel client threads. |

For each job: walltime (client-observed), RunPod's `delayTime` +
`executionTime`, per-image execution timing, GPU utilisation /
VRAM / temperature samples (when telemetry is enabled), and active
ONNX/TRT execution providers.

## DB schema

Five tables:

- `runs` — one row per benchmark invocation (image, GPU, workload).
- `jobs` — one row per RunPod request.
- `items` — one row per image inside a batch (so per-image timings
  are queryable independently from request walltimes).
- `telemetry` — nvidia-smi samples captured on the worker during
  jobs that opted in.
- `gpu_pricing` — reference table seeded at init. Joined into
  cost-per-image queries.

## Running

### One-shot

Deploy a long-lived endpoint, note the id, run a workload:

```bash
build/.with-iosuite-key python3 build/runpod_deploy.py \
  --image ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-dev \
  --gpu-class rtx-4090 \
  --endpoint-name bench-test \
  --warmup-jobs 0 --keep-endpoint
# note the endpoint id from `[deploy] endpoint id: <id>`

build/.with-iosuite-key python3 -m build.bench.runner \
  --endpoint-id <id> \
  --flavor trt --image-tag ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-dev \
  --gpu-class rtx-4090 \
  --workload batch_sweep
```

Repeat with `--workload cold_start` and `--workload sustained_concurrent`
to fill out the matrix for one (gpu, flavor) pair.

### Reading results

```bash
python3 -m build.bench.report                # all default views
python3 -m build.bench.report --query cost_per_image
python3 -m build.bench.report --query throughput_per_run
```

The DB is at `build/bench/results.db` (gitignored — regeneratable
from rerunning workloads against the same endpoint).

## Workload knobs

```
--image-w / --image-h    image resolution; default 720x720 (engine
                         optimisation profile centroid)
--max-batch              ceiling for batch_sweep; lower for 16 GB
                         tier to stay under VRAM
--concurrency            client-thread count for sustained_concurrent
--jobs-per-worker        per-thread job count
```

## Cost notes

- 4090 flex (`$0.00031/s`): a full 3-workload run is ~10 min ≈ $0.18.
- 16 GB tier (`$0.00016/s`): same workload ~10 min ≈ $0.10.
- 48 GB Pro flex (`$0.00053/s`): ~$0.32.

A full sweep across all 6 GPU tiers × {cuda, trt} ≈ $5–8.

## Adding a new GPU tier

1. Add a row to `_GPU_PRICING_ROWS` in `schema.py` (gpu_class,
   tier, vram, flex_$/s, active_$/s).
2. Add the gpu_class → sm_arch mapping to `GPU_CLASS_TO_SM`.
3. Make sure a TensorRT engine for that sm_arch exists in
   `models/MANIFEST.json` (or compile one via
   `make remote-build-engine GPU_CLASS=<class>`).
4. Run the workloads — the runner uses the manifest entry to pick
   the right engine artefact at worker boot.
