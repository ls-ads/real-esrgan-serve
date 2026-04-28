# Artefact build pipeline

How the `.onnx` files this repo serves get produced. Self-contained:
clone the repo, run the pipeline, get the same SHA-256 hashes that
ship in [`models/MANIFEST.json`](../models/MANIFEST.json) (or update
the manifest if you're cutting a new release).

The runtime side (`real-esrgan-serve`) only ever consumes artefacts.
This directory is the producer side — separate by design so the Go
binary stays pure-Go and free of PyTorch build deps.

## Pipeline

```
upstream .pth (BasicSR / Real-ESRGAN repo)
        │
        │ build/export_onnx.py    CPU only, PyTorch + ONNX
        ▼
build/dist/realesrgan-x4plus_{fp16,fp32}.onnx

build/update_manifest.py
        │ updates models/MANIFEST.json with sha256 + bytes for each
        │ artefact in build/dist/, ready to commit alongside the
        │ release tag.
        ▼
models/MANIFEST.json (committed)  +  GitHub Release (tag v0.X.Y)
```

CPU is fine — no inference happens, just graph tracing. The same
`.onnx` works on any GPU class because we do not pre-compile
per-GPU engines; runtime uses ONNX Runtime's CUDA EP everywhere.

## Quick start

```bash
# Export ONNX (containerised — see "Dependency archeology" below
# for why this isn't a plain `pip install`)
make artifacts

# Update the manifest with real hashes + sizes
make manifest
```

Equivalent without make:

```bash
docker build -f build/Dockerfile.export -t res-export build/
docker run --rm -v $(pwd)/build/dist:/output res-export
python build/update_manifest.py
```

The output lands in `build/dist/`. `*.onnx` is `.gitignore`d because
the artefacts are large (tens of MB) and belong in GH Releases, not
git.

## Dependency archeology — why the export is containerised

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
   dtypes that ORT's CUDA EP ingests cleanly. `model.half()` before
   export does NOT — it auto-casts inputs to specific ops in ways
   older runtimes choked on. We hit this in commits `e1de3fd` /
   `f1d5840` of the pre-rebuild history.
7. Opset 14 — works for the ORT versions we target. Opset 17 also
   works on modern ORT but adds nothing the model uses.

If you ever need to bump any of `(uv version, exclude-newer date,
basicsr commit, opset)`, validate the resulting `.onnx` by running
it through `runtime/upscaler.py` against a known reference image
and confirming pixel-identical output. The whole point of the
container is reproducibility — silent drift is the failure mode
we're guarding against.

## Local export (only if you've handled deps yourself)

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

1. Run `make artifacts` locally (or in CI on tag push) →
   `build/dist/*.onnx` produced.
2. CI workflow `.github/workflows/release.yml` uploads the artefacts
   to the GH release, runs `update_manifest.py`, opens a follow-up
   PR with the manifest changes.
3. Merge the manifest PR → main is now consistent with the release.

The manifest update is a separate PR rather than amending the tag
because the SHA-256s aren't known until the export finishes, and a
release tag is immutable.

## Reproducibility checklist

- The upstream `.pth` URL + hash is pinned in `export_onnx.py`. A
  silent upstream change won't sneak into our artefacts.
- ONNX export uses fixed opset and dynamic-shape spec so two runs
  on the same Python+PyTorch+ONNX produce identical bytes.
- `update_manifest.py` recomputes SHA-256 against the actual files
  on disk; mismatches against the committed manifest fail loudly
  in `--check` mode (CI).
