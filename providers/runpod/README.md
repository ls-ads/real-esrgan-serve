# RunPod template for real-esrgan-serve

Serverless GPU template that wraps the project's Python runtime
helper. One container = one warm onnxruntime session; RunPod hands
us jobs, we feed them to the helper as JSONL frames, return results.

## Build

From the repo root (build the base image first):

```bash
docker build -t real-esrgan-serve:dev .
docker build \
  --build-arg BASE_TAG=dev \
  -f providers/runpod/Dockerfile \
  -t ghcr.io/ls-ads/real-esrgan-serve:runpod-dev \
  .
docker push ghcr.io/ls-ads/real-esrgan-serve:runpod-dev
```

## Register in RunPod

1. RunPod console → Serverless → Templates → New Template
2. Container image: `ghcr.io/ls-ads/real-esrgan-serve:runpod-dev`
3. Container disk: 10 GB (the image + model fits in ~6 GB; leave
   headroom for scratch files)
4. Volume disk: 0 GB (we read/write inside the container)
5. Environment variables (optional):
   - `REAL_ESRGAN_MODEL` — override pre-baked model path
   - `GPU_ID` — pin to a specific GPU index (default 0)

## GPU class recommendations

| Class       | Throughput (jobs/min) | Cold start | Notes                                              |
|-------------|-----------------------|------------|----------------------------------------------------|
| RTX 4090    | ~23 (1280×1280 input) | ~30s       | Cost/perf sweet spot for real-esrgan-x4plus        |
| L40S        | ~30                   | ~25s       | Better steady-state at higher per-hour cost        |
| A100 40GB   | ~28                   | ~20s       | Good if your account already has A100 quota        |
| H100        | ~45                   | ~18s       | Overkill unless you're saturating consistently     |

Numbers come from the iosuite.io load tests on prod-image inputs;
see `iosuite.io/load/results/` for the underlying runs.

## Input contract

The handler accepts the same payload shape iosuite.io uses today:

```json
{
  "input": {
    "image_url":     "https://...",
    "image_base64":  "iVBORw0KG...",
    "image_path":    "uploads/abc.jpg",
    "output_path":   "results/abc-4x.jpg",
    "output_format": "jpg"
  }
}
```

Provide one of `image_url`, `image_base64`, or `image_path`. If
`output_path` is set, the handler writes the result to that path on
the container's `/workspace` volume; otherwise the result returns as
base64 in the response body.

## Cold start optimisation

The pre-fetched `.onnx` lives in the image (so no network at boot).
The helper starts in `--serve` mode and emits `{"event":"ready"}`
once onnxruntime has loaded the model + warmed up TensorRT EP if
present. The handler waits for that signal before declaring itself
healthy to RunPod.

For sustained workloads, prefer this template over the one-shot
`real-esrgan-serve upscale` invocation: the engine stays warm across
all jobs in the container's lifetime.

## Observability

The handler streams the helper's stderr through `RunPodLogger`. To
debug a stuck job, find the request ID in the iosuite.io traces and
search RunPod console logs for that ID.
