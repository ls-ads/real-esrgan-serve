# real-esrgan-serve

GPU-side serving CLI for the [iosuite](https://github.com/ls-ads/iosuite) ecosystem.
Runs Real-ESRGAN inference locally (subprocess to a Python runtime
helper) or on a configured provider (RunPod first; vast.ai and others
follow the same provider-template shape).

The user-facing CLI is `iosuite` — this repo is what `iosuite` shells
out to under the hood. You probably want this repo if:

- you're self-hosting iosuite on your own GPU box
- you're contributing to the model side (perf, new providers, new variants)
- you want to deploy the RunPod template directly

For the broader architecture, read [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## What changed in this rebuild

The previous version of this repo linked directly to TensorRT via CGO
and shipped a Dockerfile based on `nvcr.io/nvidia/tensorrt`, which has
license terms that conflict with downstream Apache-2.0 redistribution.
It also carried committed `.onnx` and `.engine` blobs (~hundreds of
MB) in the repo.

The rebuild:

- Drops the CGO bridge — Go subprocesses to a small Python helper
  using onnxruntime instead
- Switches to `nvidia/cuda:12.x-runtime-ubuntu22.04` (CUDA EULA only)
- Moves model artefacts out of git into GitHub Releases, with
  `models/MANIFEST.json` + SHA-256 verification on fetch
- Folds in the standalone `runpod-real-esrgan` repo as
  `providers/runpod/`

## Quick start (local)

Requirements:

- Linux x86_64 with an NVIDIA GPU (CUDA 12.x driver)
- Python 3.10+ with `onnxruntime-gpu`, `numpy`, `pillow`
- Go 1.25+ (only if building from source)

```bash
git clone https://github.com/ls-ads/real-esrgan-serve
cd real-esrgan-serve
make build               # builds ./bin/real-esrgan-serve

# fetch a verified model artefact (caches under XDG cache dir)
./bin/real-esrgan-serve fetch-model --name realesrgan-x4plus --variant fp16

# upscale a single image (subprocess to runtime/upscaler.py)
./bin/real-esrgan-serve upscale -i photo.jpg -o photo_4x.jpg

# or run as a daemon for batch / hot-path workloads
./bin/real-esrgan-serve serve --port 8311 &
./bin/real-esrgan-serve upscale -i photo.jpg -o photo_4x.jpg
```

CPU fallback works (slowly) for testing without a GPU:

```bash
./bin/real-esrgan-serve upscale -i photo.jpg -o photo_4x.jpg --gpu-id -1
```

## Quick start (Docker)

```bash
docker build -t real-esrgan-serve:dev .
docker run --rm --gpus all \
    -v $PWD/models:/models \
    -v $PWD/imgs:/work \
    real-esrgan-serve:dev \
    upscale -i /work/in.jpg -o /work/out.jpg --model realesrgan-x4plus
```

## Subcommands

| Command                          | Purpose                                                        |
|----------------------------------|----------------------------------------------------------------|
| `real-esrgan-serve upscale`      | One-shot inference. Subprocesses to the Python runtime.        |
| `real-esrgan-serve serve`        | Long-lived HTTP daemon — keeps the ORT session warm.           |
| `real-esrgan-serve fetch-model`  | Pull a verified `.onnx` (or pre-compiled `.engine`) artefact.  |

Run `real-esrgan-serve <cmd> --help` for the full flag surface.

## Provider templates

Each provider lives under `providers/<name>/` with its own Dockerfile,
handler, and README. The handler subprocesses the same Python runtime
helper the local CLI uses, so behaviour matches across local + cloud.

Currently shipping:

- [`providers/runpod/`](./providers/runpod/) — RunPod serverless template

In progress:

- `providers/vast/` — vast.ai

## Performance

For the hosted iosuite.io service, the cost lever is concurrent
throughput on a single GPU. Strategies live in
[`docs/PERFORMANCE.md`](./docs/PERFORMANCE.md): TensorRT EP via
onnxruntime, eager engine load on `serve` start, single-session
multi-goroutine fan-in, lazy weight unload.

Cold-start numbers from the iosuite.io UAT load tests (RTX 4090,
realesrgan-x4plus, prod-sized 1280×1280 inputs):

| Mode                                    | First-request latency | Steady-state RPS |
|-----------------------------------------|-----------------------|------------------|
| `upscale` (subprocess, cold session)    | ~30s                  | n/a (one-shot)   |
| `serve` (warm session)                  | ~7.8s                 | ~23 jobs/min     |

## Repo layout

```
real-esrgan-serve/
├── ARCHITECTURE.md          design rationale and contracts
├── cmd/real-esrgan-serve/   Go CLI entry point
├── internal/
│   ├── upscale/             one-shot subprocess flow
│   ├── server/              HTTP daemon mode
│   ├── modelfetch/          GH-Releases-backed fetch + SHA-256 verify
│   └── runtime/             helper-locator + invocation primitives
├── runtime/upscaler.py      Python helper (onnxruntime / TRT EP)
├── providers/
│   └── runpod/              RunPod serverless template
├── build/                   .pth → .onnx → .engine pipeline (producer side)
├── models/MANIFEST.json     model artefact registry (URL + SHA-256)
└── docs/                    PERFORMANCE.md, PROVIDER-GUIDE.md, ...
```

## Building the model artefacts yourself

The `.onnx` and `.engine` files served via GitHub Releases are
reproducible from upstream Real-ESRGAN `.pth` weights. See
[`build/README.md`](./build/README.md) for the pipeline (Stage A on
CPU exports ONNX; Stage B on the target GPU compiles a TRT engine).
Run `make artifacts` for the full pipeline locally, or trigger the
release workflow on tag push for the CI-built matrix.

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

The Real-ESRGAN model weights are distributed under their own
[BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
terms; check `models/MANIFEST.json` for the upstream link and license
note for each artefact.
