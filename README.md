# real-esrgan-serve

GPU-side worker for the [iosuite](https://github.com/ls-ads/iosuite)
ecosystem. Runs Real-ESRGAN inference locally (Go binary subprocesses
a Python helper) or as a RunPod serverless template that the iosuite
CLI deploys + tears down on demand.

The user-facing CLI is `iosuite`. Reach for this repo if you want to:

- Self-host on your own GPU box, with or without iosuite.
- Build a new image flavour (different precision, different EP).
- Contribute to the worker side: tiling, batching, provider templates.
- Ship a new variant of the model.

## Quick start (local)

Requirements: Linux x86_64, NVIDIA GPU + CUDA 12.x driver, Python
3.10+ with `onnxruntime-gpu` / `numpy` / `pillow`. Go 1.25+ if
building from source.

```bash
git clone https://github.com/ls-ads/real-esrgan-serve
cd real-esrgan-serve
make build               # → ./bin/real-esrgan-serve

# Pull a verified model artefact (caches under XDG cache dir).
./bin/real-esrgan-serve fetch-model --name realesrgan-x4plus --variant fp16

# One-shot upscale.
./bin/real-esrgan-serve upscale -i photo.jpg -o photo_4x.jpg

# Or run as a daemon (warm engine, JSON wire shape).
./bin/real-esrgan-serve serve --port 8311
```

CPU fallback works (slowly) for testing without a GPU:

```bash
./bin/real-esrgan-serve upscale -i photo.jpg -o photo_4x.jpg --gpu-id -1
```

## Quick start (Docker)

Three image flavours — pick what your host can satisfy:

```bash
make docker-cpu          # ~280 MB, no GPU dep, slow
make docker-cuda         # ~870 MB, NVIDIA + ORT CUDA EP
make docker-trt          # ~2 GB, NVIDIA + TensorRT direct execution
```

Run any of them:

```bash
docker run --rm --gpus all \
    -v $PWD/models:/models -v $PWD/imgs:/work \
    real-esrgan-serve:trt-dev \
    upscale -i /work/in.jpg -o /work/out.jpg --model realesrgan-x4plus
```

## Subcommands

| Command                          | What it does                                                  |
|----------------------------------|---------------------------------------------------------------|
| `real-esrgan-serve upscale`      | One-shot inference. Subprocesses the Python runtime helper.   |
| `real-esrgan-serve serve`        | Long-lived HTTP daemon. `POST /runsync` (JSON), `POST /upscale` (multipart). |
| `real-esrgan-serve fetch-model`  | Pull a verified `.onnx` / `.engine` artefact from GitHub Releases. |

`real-esrgan-serve <cmd> --help` prints the full flag surface.

The daemon's JSON envelope matches what iosuite serve and the RunPod
worker expect:

```
POST /runsync   Content-Type: application/json
{"input": {"images": [{"image_base64": "..."}],
           "tile": true, "output_format": "jpg"}}

→ {"status": "COMPLETED",
   "output": {"outputs": [{"image_base64": "...", "exec_ms": 612}]}}
```

`tile: true` slices inputs >1280² into 1024² tiles, infers per
tile, and stitches with linear-ramp blending in the overlap zones —
inputs up to 4096² are handled this way.

## Provider templates

Each provider lives under `providers/<name>/` with its own
Dockerfile, handler, and README. The handler subprocesses the same
Python runtime helper the local CLI uses; behaviour is identical
across local + cloud.

- [`providers/runpod/`](./providers/runpod/) — RunPod serverless
  template. Deployed by `iosuite endpoint deploy --tool real-esrgan`
  using the manifest at [`deploy/runpod.json`](./deploy/runpod.json).

## Deploy manifests

`deploy/runpod.json` is the source-of-truth for how iosuite deploys
this module to RunPod: image tag, container disk, GPU pool map per
class, FlashBoot default, CUDA pin, env vars. iosuite reads it at
deploy time — bumping any of those fields lands here, not in iosuite.

`deploy/benchmark.json` declares the matching benchmark suite for
`iosuite endpoint benchmark`: warmup count, measure count, request
template, metrics, and the input image to send.

Field reference: [`deploy/SCHEMA.md`](./deploy/SCHEMA.md).
Validator (CI gate): [`build/validate_manifest.py`](./build/validate_manifest.py).

## Performance

Cold-start numbers from a RunPod RTX 4090 ADA_24 deploy
(realesrgan-x4plus FP16, TensorRT direct execution):

| Mode                                    | Latency               | Notes                          |
|-----------------------------------------|-----------------------|--------------------------------|
| Cold start (full e2e, no FlashBoot)     | ~46 s                 | dominated by image pull        |
| Cold start with FlashBoot               | ~5 s                  | snapshot resume                |
| Warm exec (TRT)                         | ~19 ms                | p50 across 10 jobs (64×64 in)  |
| Tiled 2K → 8K (TRT, warm)               | ~70 s                 | 2048×1500 → 8192×6000          |

Performance discussion: [`docs/PERFORMANCE.md`](./docs/PERFORMANCE.md).

## Repo layout

```
real-esrgan-serve/
├── ARCHITECTURE.md          design rationale and contracts
├── cmd/real-esrgan-serve/   Go CLI entry point
├── internal/
│   ├── upscale/             one-shot subprocess flow
│   ├── server/              HTTP daemon mode (/runsync + /upscale)
│   ├── modelfetch/          GH-Releases-backed fetch + SHA-256 verify
│   └── runtime/             helper-locator + invocation primitives
├── runtime/upscaler.py      Python helper (ORT or TRT direct)
├── runtime/tiling.py        slice / infer / stitch for >1280² inputs
├── providers/runpod/        RunPod serverless template
├── deploy/                  iosuite-readable deploy + benchmark manifests
├── build/                   .pth → .onnx → .engine pipeline
├── models/MANIFEST.json     model artefact registry (URL + SHA-256)
└── docs/                    PERFORMANCE.md, PROVIDER-GUIDE.md, ...
```

## Building model artefacts

The `.onnx` / `.engine` files served via GitHub Releases are
reproducible from upstream Real-ESRGAN `.pth` weights. See
[`build/README.md`](./build/README.md) for the pipeline.
`make artifacts` runs the full pipeline locally; the release workflow
on tag push builds + publishes to GitHub Releases for CI.

## Documentation

- iosuite CLI reference (covers iosuite endpoint deploy / benchmark
  flow against this worker): <https://iosuite.io/cli-docs>
- Architecture: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- Manifest schema: [`deploy/SCHEMA.md`](./deploy/SCHEMA.md)

## License

Apache-2.0. See [`LICENSE`](./LICENSE) for the text and
[`NOTICE.md`](./NOTICE.md) for third-party attributions
(Real-ESRGAN weights, NVIDIA CUDA runtime, ONNX Runtime, TensorRT,
Pillow, et al). When forking or vendoring, preserve `LICENSE`,
`NOTICE.md`, and `third-party-licenses/`.
