# real-esrgan-serve architecture

This is the GPU-side of the iosuite ecosystem: the serving CLI, the
provider templates, and the model artifact contract. The user-facing
`iosuite` CLI wraps it; downstream tools (future `whisper-serve`,
`stable-diffusion-serve`, etc.) follow the same shape.

## Goals

1. **License-clean distribution.** The previous Dockerfile used
   `nvcr.io/nvidia/tensorrt`, whose terms conflict with downstream
   Apache-2.0 redistribution. The rebuild uses
   `nvidia/cuda:12.x-runtime-ubuntu22.04` (CUDA EULA only) and installs
   ONNX Runtime via pip — no nvcr.io image in the redistributable
   chain.
2. **No bundled C++ in this repo.** The previous codepath linked
   directly to TensorRT via CGO. The rebuild calls a Python helper as
   a subprocess. The Go binary stays pure Go, cross-compiles cleanly,
   and its only runtime requirement is "a working Python install with
   onnxruntime".
3. **Verified model artifacts, fetched on demand.** The repo carried
   committed `.onnx` and `.engine` blobs (~hundreds of MB). The rebuild
   removes all binary blobs and ships them as GitHub Releases, with
   SHA-256 verification baked into the fetch path.
4. **Optimised for concurrent throughput on a single GPU.** The
   primary cost lever for the hosted iosuite.io service. Server mode
   batches inflight requests, holds a warm engine, and reuses a single
   ORT session across N concurrent goroutines.
5. **Provider templates own the deploy story.** RunPod, vast.ai,
   etc. live under `providers/<name>/` with their own Dockerfile + any
   platform-specific handler. The runpod-real-esrgan repo merges in
   here.

## Tool surface

The Go CLI has three subcommands, all subprocess-friendly:

| Subcommand    | Purpose                                                      |
|---------------|--------------------------------------------------------------|
| `upscale`     | One-shot inference. Subprocesses to the Python runtime.      |
| `serve`       | HTTP daemon mode. Holds a warm ORT session for hot path.     |
| `fetch-model` | Pull a verified model artifact from GitHub Releases.         |

Default behaviour for `upscale` is "subprocess to Python, return".
`serve` is opt-in for users who batch many images and want to avoid
the cold-start cost of repeated process startup. The subprocess path
is always available — server mode is an optimisation, never a
requirement. Future tools (whisper-serve, etc.) follow the same shape;
the iosuite CLI doesn't have to know which tools have a daemon up.

### `upscale`

```
real-esrgan-serve upscale \
  --input  <file-or-dir>     # path; auto-detects file vs directory
  --output <file-or-dir>     # optional; auto-derived if omitted
  --model  <name>            # default: realesrgan-x4plus
  --gpu-id <int>             # default: 0
  --scale  <int>             # default: 4 (model native)
  --json-events              # emit progress as JSON to stdout (for iosuite CLI)
```

Subprocess flow:

1. CLI validates flags + image dimensions
2. Resolves model path (env > flag > config > default cache dir)
3. Spawns `python3 runtime/upscaler.py --image ... --model ... --output ...`
4. Captures stdout (JSON events) + stderr (logs)
5. Exits with the Python helper's exit code

### `serve`

```
real-esrgan-serve serve \
  --port <int>     # default: 8311
  --bind <addr>    # default: 127.0.0.1
  --model <name>   # which model to keep warm (default: realesrgan-x4plus)
  --concurrency <int>  # max in-flight requests; default: 1 per GPU
```

When running, accepts `POST /upscale` with multipart image. Hot path
keeps the ORT session warm across requests.

### `fetch-model`

```
real-esrgan-serve fetch-model \
  --name <name>      # required; e.g. realesrgan-x4plus
  --variant <type>   # fp16|fp32; default: fp16
  --dest <path>      # default: ~/.cache/real-esrgan-serve/models
```

Fetches the model from GitHub Releases. Verifies SHA-256 against a
manifest (`models/MANIFEST.json` in this repo). On hash mismatch:
fail loudly and delete the partial file.

## Runtime helper (`runtime/upscaler.py`)

A small standalone Python script. Single responsibility: take a
preprocessed image + a model file, run inference, write the output.
It's **separate from the Go binary**: ships in the same repo, gets
installed alongside (apt package, pip-installed wheel, or copied by
the install script).

Reasons:
- Python's onnxruntime ecosystem is the most mature for image-to-image
  inference. Going Go-native here would mean either CGO bindings
  (which we're explicitly avoiding) or a much weaker pure-Go ONNX
  implementation (immature).
- Subprocess boundary = process isolation. A model crash takes down
  the helper, not the Go CLI. iosuite's wrapper logic stays unchanged.
- Execution provider: `CUDAExecutionProvider` for GPU, `CPUExecutionProvider`
  fallback. The TensorRT path was investigated and rejected — it
  doubles image size (cuDNN ~1 GB) without meaningful warm-exec
  wins at our throughput.

## Provider templates

`providers/<name>/` is one directory per provider. Each contains:

- `Dockerfile` — image layout for that provider's runner
- `handler.py` (or equivalent) — the platform-specific request handler
- `README.md` — deploy instructions, GPU class recommendations, expected
  cold-start time, throughput notes

`providers/runpod/` is the first. Folded in from the previous
runpod-real-esrgan repo. The handler subprocesses the same
`runtime/upscaler.py` as the local CLI, so behaviour matches across
local and provider runs.

## Model artifact contract

- All artifacts live in GitHub Releases: `realesrgan-x4plus-fp16.onnx`
  and `realesrgan-x4plus-fp32.onnx`. ONNX-only — no per-GPU
  pre-compiled binaries to maintain.
- `models/MANIFEST.json` in the repo lists every supported artifact,
  its SHA-256, the GitHub Release URL, the model card.
- `fetch-model` reads the manifest, downloads, verifies, places.
- iosuite CLI defers to `real-esrgan-serve fetch-model` rather than
  re-implementing the fetch logic — single source of truth for
  artifact integrity.

### How artifacts are produced

The pipeline lives under [`build/`](./build/) and is reproducible
from upstream `.pth` weights:

- **`build/export_onnx.py`** — CPU-only PyTorch trace from the
  upstream Real-ESRGAN `.pth` to `.onnx`. Pinned upstream URL +
  SHA-256, fixed opset, fixed dynamic-shape spec → byte-identical
  output across runs. Runs in a deps-frozen container
  (`build/Dockerfile.export`); see `build/README.md` for why.
- **Manifest sync** (`build/update_manifest.py`) — recomputes
  SHA-256 + size for every artefact in `build/dist/` and writes
  back to `models/MANIFEST.json`. `--check` mode (used in CI) exits
  non-zero on drift without modifying anything.

Releases are cut by tagging `v*` — `.github/workflows/release.yml`
runs the export, uploads to the GitHub Release, and opens a PR with
the manifest update. See `build/README.md` for the full walkthrough.

## Cold-start optimisation (server mode)

Cold start on serverless GPU providers is dominated by image pull
(~1.6 GB → ~45 s on RunPod). Container start + ORT model load
together are <1 s on RTX 4090. Strategies:

1. **Eager session load on `serve` start** — first request pays no
   warmup cost beyond ORT's CUDA initialization (~hundreds of ms).
2. **Single ORT session, multiple goroutines** — the Go server holds
   one Python helper subprocess in serve mode and pipes inference
   requests over stdin/stdout. One session, N concurrent senders.
3. **Lazy weight unload after idle** — opt-in. Default keeps weights
   resident.

## Contract with iosuite CLI

The iosuite CLI subprocesses to `real-esrgan-serve` for every
operation. It never imports our Go packages. The contract is:

- Stable CLI surface: `upscale`, `serve`, `fetch-model` flag shapes
  are SemVer'd
- JSON events on stdout when `--json-events` is set (so iosuite can
  render progress, capture errors, run the embedded benchmark)
- Exit codes documented (0 = success; 1 = user error; 2 = runtime
  error; 3 = environment error like missing model)

This means iosuite ships independently of real-esrgan-serve. New tool
in the iosuite ecosystem? Build a `<tool>-serve` Go binary with the
same shape, ship it; iosuite already knows how to subprocess to it.

## Out of scope

- **Model training / fine-tuning.** This repo serves models. Training
  belongs elsewhere if it ever happens.
- **Image preprocessing UI.** Cropping, alignment, etc. — that's the
  iosuite CLI / web UI's job.
- **Multi-GPU orchestration on a single host.** The host running this
  binary is assumed to have one GPU (or we pin to one). Multi-GPU is
  a provider-side concern (RunPod assigns one worker = one GPU).

## Migration notes

The previous codebase had `cmd/{root,build,server}.go` with CGO to
TensorRT. The rebuild deletes `pkg/tensorrt/` entirely (CGO bridge),
deletes `pkg/server/` (it's being rewritten in pure Go subprocess
mode), drops the committed `.onnx` and `.engine` files, replaces the
Dockerfile, and lays out the structure described above. LICENSE,
go.mod (regenerated), and the README (rewritten) survive.
