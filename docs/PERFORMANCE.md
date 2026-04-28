# Performance notes

The hosted iosuite.io service spends real money on GPU workers, and
self-hosters with a single 4090 want every drop of throughput. This
doc captures the optimisation strategies the rebuild is structured to
support.

## Strategies

### 1. CUDA execution provider via onnxruntime

`runtime/upscaler.py` requests the `CUDAExecutionProvider` when a GPU
is available, falls back to `CPUExecutionProvider` otherwise. ONNX
+ CUDA EP gives us a stable, cuDNN-free runtime path that doesn't
balloon the image with multi-gigabyte tensor libraries.

The TensorRT EP path was investigated and rejected during the
rebuild — see `Dockerfile` header for the math. Short version:
~5x warm-exec speedup needs ~1 GB of cuDNN runtime libraries, which
more than doubles cold-start image-pull time. At our throughput
profile (cold start dominates), that's a net loss.

### 2. Eager session load on `serve` start

`serve` mode runs `upscaler.py --serve`, which loads the model
session before signalling `{"event":"ready"}` over stdout. The Go
HTTP server doesn't accept connections until that signal arrives, so
the first end-user request never pays the warmup cost.

### 3. Single ORT session, multi-goroutine fan-in

The Go server holds one helper subprocess and one shared stdin lock.
N concurrent HTTP handlers serialise their JSONL frames over that
stdin, then wait on a per-job-ID result channel populated by a single
stdout reader goroutine. One session, no per-request spin-up, no GPU
contention from multiple processes.

### 4. Lazy weight unload after idle

Default keeps weights resident (cheap RAM, expensive cold reload).
Opt-in `--idle-unload <duration>` flag (planned) frees the session
after N minutes of inactivity, useful on hosts that share the GPU
with other workloads.

## Measured numbers

RunPod RTX 4090 (ADA_24 pool), realesrgan-x4plus FP16 ONNX,
64×64 → 256×256 test input, ONNX + CUDA EP:

| Mode                                    | Latency      | Notes                       |
|-----------------------------------------|--------------|-----------------------------|
| Cold start (full e2e)                   | ~46.0 s      | 45.2 s pull, 0.4 s exec     |
| Warm exec (CUDA EP, p50 over 5 jobs)    | ~423 ms      | walltime ~770 ms incl queue |

Cold start is dominated by image pull (1.6 GB). Warm exec is
dominated by Real-ESRGAN's compute itself; CUDA EP doesn't add
meaningful overhead. UAT load-test data on prod-sized 1280×1280
inputs is in the iosuite.io repo: `load/results/upscale-stress-*.md`.

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
- **Image slim** — the 1.6 GB image is dominated by ORT's wheel
  (~600 MB) and runpod SDK's transitive deps (~165 MB,
  fastapi[all]/boto3/paramiko/sentry-sdk). A `runpod --no-deps` +
  explicit minimal install could save ~80 MB, cutting cold-start
  pull time another ~5%.
