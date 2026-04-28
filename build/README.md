# Artefact build pipeline

How the `.onnx` and `.engine` files this repo serves get produced.
Self-contained: clone the repo, run the pipeline, get the same SHA-256
hashes that ship in [`models/MANIFEST.json`](../models/MANIFEST.json)
(or update the manifest if you're cutting a new release).

The runtime side (`real-esrgan-serve`) only ever consumes artefacts.
This directory is the producer side — separate by design so the Go
binary stays pure-Go and free of PyTorch / TensorRT build deps.

## Pipeline stages

```
upstream .pth (BasicSR / Real-ESRGAN repo)
        │
        │ build/export_onnx.py    Stage A — CPU only, PyTorch + ONNX
        ▼
build/dist/realesrgan-x4plus_{fp16,fp32}.onnx
        │
        │ build/compile_engine.py Stage B — runs on target GPU hardware
        ▼
build/dist/realesrgan-x4plus-<gpu-class>-<sm-arch>-trt<ver>_fp16.engine

build/update_manifest.py
        │ updates models/MANIFEST.json with sha256 + bytes for each
        │ artefact in build/dist/, ready to commit alongside the
        │ release tag.
        ▼
models/MANIFEST.json (committed)  +  GitHub Release (tag v0.X.Y)
```

Stage A runs anywhere with PyTorch (CPU is fine — no inference, just
graph tracing). Stage B has to run on each target GPU class because
TensorRT engines are pinned to the SM architecture they were built
on. CI runs Stage B in a matrix across whichever GPU classes the
release supports; humans run it locally to add a class.

## Quick start (Docker — recommended)

```bash
# Stage A: export ONNX (containerised — see "Dependency archeology"
# below for why this isn't a plain `pip install`)
docker build -f build/Dockerfile.export -t res-export build/
docker run --rm -v $(pwd)/build/dist:/output res-export

# Stage B: compile a TRT engine on the GPU you're sitting at
make artifacts-engine

# Update the manifest with real hashes + sizes
make manifest
```

Or the convenience target that runs both stages + the manifest update:

```bash
make artifacts && make manifest
```

The output lands in `build/dist/`. That directory is `.gitignore`d
because the artefacts are large (hundreds of MB) and belong in GH
Releases, not git.

## Dependency archeology — why Stage A is containerised

> **TL;DR**: `basicsr` (Real-ESRGAN's model package) is unmaintained.
> Its declared dep ranges still claim modern PyTorch / NumPy / SciPy
> work, but in practice imports break on anything resolved against
> the current PyPI. The fix is `uv pip install --exclude-newer
> 2022-09-04T23:59:59Z` — time-travel resolution against the day
> basicsr last worked end-to-end. Plus a pinned upstream commit
> (`a4abfb29...`) and **post-export** FP16 conversion via
> `onnxconverter-common`. All three of these were learned the hard
> way; please don't change them casually.

`build/Dockerfile.export` encodes the working set:

1. Base: `ghcr.io/astral-sh/uv:python3.9-bookworm-slim` (Python 3.9
   is the version basicsr was last compatible with end-to-end).
2. Real-ESRGAN repo cloned + `git checkout
   a4abfb2979a7bbff3f69f58f58ae324608821e27` — pinning a tag wasn't
   safe because tags can be force-moved upstream.
3. `uv pip install --exclude-newer 2022-09-04T23:59:59Z -r
   requirements.txt && uv pip install --exclude-newer ... onnx
   onnxconverter-common` — both pip calls share the same resolution
   context, so onnx + onnxconverter-common pull versions that
   coexist with basicsr's own pins.
4. `system_site_packages=False` (default) — no host pollution.

The export script (`export_onnx.py`):

5. Loads `params_ema` (the official inference checkpoint, not the
   training-time `params` key).
6. Exports as **FP32** first, then runs
   `onnxconverter_common.float16.convert_float_to_float16` on the
   resulting `.onnx`. This produces FP16 graphs with consistent op
   dtypes that all our target TRT versions ingest cleanly.
   `model.half()` before export does NOT — older TRT chokes on
   auto-cast inputs to specific ops. We hit this in commits
   `e1de3fd` / `f1d5840` of the pre-rebuild history.
7. Opset 14 — works for both TRT 8.x and 10.x. Opset 17 also works
   on modern ORT but adds nothing the model uses.

If you ever need to bump any of `(uv version, exclude-newer date,
basicsr commit, opset)`, validate the resulting `.onnx` by running
it through `runtime/upscaler.py` against a known reference image
and confirming pixel-identical output. The whole point of the
container is reproducibility — silent drift is the failure mode
we're guarding against.

## Local Stage A (only if you've handled deps yourself)

```bash
cd build
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # likely will not work; see above
python export_onnx.py --variant both
```

The `requirements.txt` here is informational — it lists what versions
the Docker image ends up resolving to. `pip` won't reproduce that
resolution as reliably as `uv --exclude-newer`. If the local install
works for you, great; if it doesn't, use the container.

## Publishing a release

For a maintainer cutting `v0.X.Y`:

1. Run Stage A locally → push the resulting `.onnx` files
2. Trigger `.github/workflows/release.yml` with the tag → CI runs
   Stage B across the GPU matrix on hosted GPU runners (or your
   self-hosted GPU runners), publishes the artefacts to the GH
   release, runs `update_manifest.py`, opens a follow-up PR with
   the manifest changes
3. Merge the manifest PR → main is now consistent with the release

The manifest update is a separate PR rather than amending the tag
because the SHA-256s aren't known until Stage B finishes, and a
release tag is immutable.

## Adding a new GPU class

Run Stage B on a host with the new GPU, commit the manifest update,
no Stage A repeat needed. The `.onnx` is GPU-class-agnostic; only
the `.engine` is tied to a specific SM architecture.

## Remote builds via RunPod (no GPU required locally)

Most maintainers don't have an RTX 4090 / L40S / A100 / H100 sitting
on their desk. `build/remote_build.py` spins up a temp RunPod pod,
runs **the exact same `make artifacts-engine` target you'd run
locally** but on the remote GPU, and pulls the `.engine` back.
Typical run: 5–15 minutes, ~$0.05–0.30 of GPU time depending on
class.

> **Single source of truth**: the remote script doesn't carry its
> own compile flow. It tarballs your working tree
> (`git ls-files | tar`), pushes it to the pod, runs `make
> artifacts-engine` over SSH. If the remote build works, the local
> build is guaranteed to work — same code, different host. Any
> change you make to `compile_engine.py` or the Makefile takes
> effect for both paths immediately, no separate maintenance.

**Why this matters for trust**: any developer can re-run the
pipeline against your own RunPod account, produce the same
`.engine`, and verify it byte-for-byte against what we publish.
"Don't trust, verify" — without needing to own GPU hardware.
Because the remote runs the same Makefile target, your reproduction
isn't approximate; it's bit-identical (modulo TensorRT determinism,
which is true for fixed GPU + TRT version).

```bash
export RUNPOD_API_KEY=<your-key>
python build/remote_build.py --gpu-class rtx-4090

# or via Makefile:
make remote-build-engine GPU_CLASS=rtx-4090
```

Prerequisites:
- RunPod account with credit
- An ED25519 SSH public key registered on your RunPod profile
  (Settings → SSH Public Keys). The matching private key on your
  laptop at `~/.ssh/id_ed25519` (or `--ssh-key /path/to/key`).
- $RUNPOD_API_KEY env var (see Auditing below).
- `git` + `tar` locally (used to bundle the working tree).

What happens, in order:

1. Bundle the working tree: `git ls-files` lists every tracked path,
   piped to `tar` which reads the files' current working-tree
   contents. Untracked outputs (`bin/`, `build/dist/*.engine`) are
   automatically excluded.
2. Resolve `--gpu-class` → RunPod GPU type ID via the GraphQL API.
3. Spin up a Secure Cloud pod with the configured base image. The
   startup script installs `make` + `openssh-server` + `tensorrt`
   then waits for connections.
4. Wait for ssh to actually accept (RunPod's status flips before
   sshd is fully up, so the script polls).
5. SCP the tarball, untar into `/workspace/repo`.
6. SSH `cd /workspace/repo && make artifacts-engine` — same target
   you'd type locally. The compile script's `--auto-detect-gpu`
   picks up the pod's actual nvidia-smi output for the gpu_class
   string baked into the filename.
7. SCP every produced `*.engine` back to your local `build/dist/`,
   verify each one's SHA-256.
8. Terminate the pod (`finally` block — always runs even on error,
   so a bug or Ctrl-C never leaves an idle GPU billing you).

If you want to run a different Make target remotely (e.g. the full
`artifacts` flow including Stage A), pass `--make-target`:

```bash
python build/remote_build.py --gpu-class rtx-4090 --make-target artifacts
```

### Auditing

Every step logs the RunPod GraphQL request ID and the pod ID to
stderr. Cross-reference your RunPod billing dashboard's line items
to the `[remote] created pod <id>` log line — one pod = one billing
line. The pod's `--keep-pod` flag (debug only) lets you SSH in
manually if a build fails and you want to diagnose without the
pod auto-terminating.

### Maintainer convenience: `build/.with-iosuite-key`

If you (the iosuite maintainer) keep a `RUNPOD_API_KEY` in
`~/Projects/iosuite.io/.env`, the wrapper script
`build/.with-iosuite-key` reads only that one line, exports it into
its child process's env, and execs whatever follows. The value
never echoes to terminal or logs.

```bash
build/.with-iosuite-key make remote-build-engine GPU_CLASS=rtx-4090
build/.with-iosuite-key python build/remote_build.py --onnx ... --gpu-class l40s
```

Other developers should set `RUNPOD_API_KEY` directly via their own
mechanism (`direnv`, `.envrc`, secrets manager, etc.) — the wrapper
is intentionally specific to one personal-machine path and is not a
recommended pattern for general use.

### Cost estimates (rough)

| GPU class | Pod $/hr (Secure Cloud) | Compile time | One-time cost |
|-----------|-------------------------|--------------|---------------|
| rtx-4090  | ~$0.74                  | ~10 min      | ~$0.13        |
| l40s      | ~$0.99                  | ~7 min       | ~$0.12        |
| a100-40   | ~$1.89                  | ~6 min       | ~$0.19        |
| h100      | ~$3.39                  | ~4 min       | ~$0.23        |

Numbers approximate as of 2026-04. RunPod adjusts pricing; check
their dashboard for current rates. The script trades wall time for
cost — if you're queueing many compiles, parallelise by running the
script multiple times against different `--gpu-class`.

## Reproducibility checklist

- The upstream `.pth` URL + hash is pinned in `export_onnx.py`. A
  silent upstream change won't sneak into our artefacts.
- ONNX export uses fixed opset version (17) and a fixed dynamic-shape
  spec so two runs on the same Python+PyTorch+ONNX produce identical
  bytes.
- TensorRT engine builds embed the TRT version + SM arch + precision
  in the filename so consumers can verify they're running the right
  artefact for their GPU.
- `update_manifest.py` recomputes SHA-256 against the actual files
  on disk; mismatches against the committed manifest fail loudly.
