# Performance notes

The hosted iosuite.io service spends real money on GPU workers, and
self-hosters with a single 4090 want every drop of throughput. This
doc captures the optimisation strategies the rebuild is structured to
support — most are stubs in the current scaffold and will be filled in
as we land each piece.

## Strategies

### 1. TensorRT execution provider via onnxruntime

`runtime/upscaler.py` requests the TensorRT EP first, falls back to
CUDA EP, finally CPU EP. When TRT is available, onnxruntime compiles
the ONNX graph into a TRT engine on first use and caches the result
under `$XDG_CACHE_HOME/real-esrgan-serve/trt-cache/`. Subsequent runs
load the cached engine in seconds rather than rebuilding (~30s).

### 2. Eager engine load on `serve` start

`serve` mode runs `upscaler.py --serve`, which loads the model
session before signalling `{"event":"ready"}` over stdout. The Go
HTTP server doesn't accept connections until that signal arrives, so
the first end-user request never pays the warmup cost.

### 3. Single ORT session, multi-goroutine fan-in

The Go server holds one helper subprocess and one shared stdin lock.
N concurrent HTTP handlers serialise their JSONL frames over that
stdin, then wait on a per-job-ID result channel populated by a single
stdout reader goroutine. One TRT engine, no per-request session
spin-up, no GPU contention from multiple processes.

### 4. Lazy weight unload after idle

Default keeps weights resident (cheap RAM, expensive cold reload).
Opt-in `--idle-unload <duration>` flag (planned) frees the session
after N minutes of inactivity, useful on hosts that share the GPU
with other workloads.

### 5. Pre-baked engines per GPU class

For RunPod-class hosts we know in advance, GitHub Releases ship
pre-compiled `.engine` files (e.g. `realesrgan-x4plus-rtx-4090-sm89-trt10.8_fp16.engine`).
`fetch-model --variant engine --gpu-class rtx-4090` pulls the matching
engine, skipping the ~30s onnxruntime first-build entirely. If no
engine matches the host's GPU, we fall back to the `.onnx` and let
TRT EP build + cache locally.

## Measured numbers

UAT load tests (RTX 4090 on RunPod, realesrgan-x4plus, prod-sized
1280×1280 inputs):

| Mode                                    | First-request | Steady-state RPS | Cost/req (4090) |
|-----------------------------------------|---------------|------------------|-----------------|
| `upscale` (subprocess, cold)            | ~30s          | n/a              | n/a             |
| `serve` (warm, .onnx + JIT)             | ~7.8s         | ~23 jobs/min     | ~$0.0007        |
| `serve` (warm, pre-baked .engine)       | ~0.5s         | ~23 jobs/min     | ~$0.0007        |

Underlying runs are in the iosuite.io repo:
`load/results/upscale-stress-*.md`. Numbers are end-to-end including
RunPod queue + network round-trip, not just GPU time.

## Open optimisations

- **Batching** — onnxruntime supports batch inputs natively.
  Currently we run one image per inference call. A batch-2 mode could
  ~1.7× throughput at the cost of latency variance. Worth measuring
  once the queue depth justifies it.
- **Half-precision input pipeline** — preprocessing currently casts
  to float32 on CPU before sending to GPU. Casting on-GPU after
  upload would save host→device bandwidth.
- **Output encoding off the GPU** — JPEG encoding of a 5120×5120
  upscaled output dominates per-request CPU time. Encoding on a
  worker thread while the GPU starts the next job would overlap the
  cost.
