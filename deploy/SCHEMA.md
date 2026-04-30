# Deploy manifests

Source-of-truth for how this `*-serve` module gets deployed. iosuite
(the cross-tool CLI at github.com/ls-ads/iosuite) reads these files
to drive `iosuite endpoint deploy --tool real-esrgan` and friends —
zero implementation knowledge baked into iosuite itself.

## `runpod.json`

The RunPod-serverless deploy manifest. Field reference:

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | `"1"` today. Bump when the shape changes incompatibly. |
| `tool` | string | yes | Stable kebab-case name. Must match what iosuite users pass to `--tool`. |
| `description` | string | no | One-liner shown in `iosuite endpoint list` output. |
| `image` | string | yes | Full image ref including tag. iosuite passes this verbatim to RunPod's `saveTemplate`. Bump in lockstep with this manifest's git tag. |
| `endpoint.container_disk_gb` | int | yes | Image size + a few GB headroom. |
| `endpoint.workers_max_default` | int | yes | Default concurrency cap. iosuite users override via `--workers-max`. |
| `endpoint.idle_timeout_s_default` | int | yes | RunPod scaler idle timeout. Lower = faster scale-to-zero, more cold starts. |
| `endpoint.flashboot_default` | bool | yes | RunPod FlashBoot (snapshot resume). Strong default for any image >1 GB. |
| `endpoint.min_cuda_version` | string \| null | yes | RunPod's CUDA-pinning enum. Set this whenever the image bundles a tensorrt / cuda runtime that demands a minimum host driver — without it, workers can land on hosts with older drivers and die at container init with `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=...`. The error stays invisible to anything but worker stderr; the daemon and api see "queued forever". Valid values mirror RunPod's REST schema: `"11.8"`, `"12.0"` … `"13.0"`, or null. |
| `gpu_pools` | object | yes | Map of user-facing kebab-case GPU class → RunPod pool ID. Mirror the canonical list at <https://docs.runpod.io/references/gpu-types#gpu-pools>. |
| `env` | array | yes | Each entry `{key, value}` becomes a container env variable on the worker. |

## Why these live here, not in iosuite

The `*-serve` module owns its deploy specs because:

1. **Image-tag truth**: when this repo cuts a new image (e.g. `runpod-trt-0.3.0`), the right disk / CUDA / pool config lands in the same commit. iosuite doesn't need a corresponding patch.
2. **Tool authorship**: anyone writing a new `*-serve` module can declare its deploy shape without touching iosuite's source. iosuite is one binary that supports many tools.
3. **Versioning**: `iosuite endpoint deploy --tool real-esrgan --version <git-tag>` fetches this file at that git tag. Pinning is exact.

## `benchmark.json`

Drives `iosuite endpoint benchmark --tool real-esrgan --endpoint-id <id>`. The serve module owns the workload (which input to use, how many warmups, what metrics matter); iosuite owns the wire (POST loop, timing, aggregation, output formatting).

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | Independent of `runpod.json`'s version. `"1"` today. |
| `tool` | string | yes | Must match `runpod.json`'s tool. |
| `warmup` | int | yes | Requests issued + ignored before measurement begins. Catches RunPod cold-start variance. |
| `measure` | int | yes | Requests recorded for metric aggregation. |
| `input_resource` | string | yes | Repo-relative path to a base64-encoded image (e.g. `deploy/bench/64x64-rgb.png.b64`). iosuite fetches this at the same git tag as the manifest itself. |
| `request_template.input` | object | yes | Worker-side payload. iosuite injects `images:[{image_base64: ...}]` from `input_resource`; everything else (tile, output_format, discard_output) ships verbatim. |
| `metrics` | array | yes | Each entry `{name, from, agg}`. `from` names a numeric field on the worker's per-item response (e.g. `exec_ms`). `agg` is one of `mean`, `p50`, `p95`, `p99`, `max`, `min`. |

Sized for "is the endpoint healthy + how fast does a small request return" rather than "find the optimal GPU class across the matrix" (that's still in `build/bench/sweep.py`, kept separate as a maintainer tool).

## Validation

`build/validate_manifest.py` parses every manifest under `deploy/` against the schema above. CI runs it on every PR; failures block merge. The same script is used by iosuite's manifest fetcher to second-guess the wire data.

## How iosuite resolves the URL

```
https://raw.githubusercontent.com/<owner>/<repo>/<git-tag>/deploy/runpod.json
```

iosuite has a small registry mapping `tool` → `<owner>/<repo>` + a default git tag for the `--version` flag. New tools are registered by a one-line addition there; everything else (image, disk, GPU pools, …) flows out of *this* file.
