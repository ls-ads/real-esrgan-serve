# Tests

Three layers, with intentional separation:

| Layer | Where | Cost | Run via |
|---|---|---|---|
| Go unit tests | `internal/**/*_test.go` | free | `make test-go` |
| Python unit + property tests | `tests/test_*_unit.py`, `tests/test_image_sizing_property.py` | free | `make test-py` |
| Live RunPod integration | `tests/test_live_runpod.py` | ~$0.005/run on a 4090 | `make test-live` |

`make test` runs unit + property tests across both languages. Live
tests are opt-in.

## Running unit + property tests

```bash
make test
# or just one language:
make test-go
make test-py
```

`make test-py` runs inside the `real-esrgan-serve:cpu-dev` Docker image.
That image already has Python + numpy + Pillow + onnxruntime baked in;
the make target layers `pytest + hypothesis` on top at run time. No
host Python install needed — `python3 -m pytest` against the host
will fail unless you've installed `tests/requirements.txt` yourself.

If you've already got a venv with the test deps, you can shortcut:

```bash
python3 -m pytest tests/ -m "not live"
```

## Running live tests

Live tests submit real jobs to a deployed RunPod endpoint. Each one
costs roughly one GPU-second.

### One-time setup: deploy a long-lived test endpoint

The default `make deploy-runpod` tears the endpoint down on
completion, which is wrong for a test suite. Override that:

```bash
build/.with-iosuite-key python3 build/runpod_deploy.py \
    --image ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-dev \
    --gpu-class rtx-4090 \
    --endpoint-name real-esrgan-serve-tests \
    --warmup-jobs 0 \
    --keep-endpoint
```

Note the endpoint id from the deploy output (line starting with
`[deploy] endpoint id:`).

### Per-run

```bash
export RUNPOD_ENDPOINT_ID=<id from deploy output>
build/.with-iosuite-key make test-live
```

The wrapper loads `RUNPOD_API_KEY` from `~/Projects/iosuite.io/.env`
without echoing it. Without the wrapper, set both env vars manually:

```bash
RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... make test-live
```

### Tear down when done

The endpoint keeps billing until torn down. Use the RunPod console
or:

```bash
build/.with-iosuite-key python3 -c '
import json, os, urllib.request
key = os.environ["RUNPOD_API_KEY"]
eid = os.environ["RUNPOD_ENDPOINT_ID"]
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({
        "query": f"mutation {{ deleteEndpoint(id: \"{eid}\") }}"
    }).encode(),
    headers={"Authorization": f"Bearer {key}",
             "Content-Type": "application/json",
             "User-Agent": "real-esrgan-cleanup/0.1"},
)
print(urllib.request.urlopen(req, timeout=30).read().decode())
'
```

## What's covered

### Go (`internal/modelfetch`)

- `Manifest.Find`: variant matching, sm-arch vs gpu-class
  precedence, error paths (missing disambiguator, unknown variant).
- `verifyHash`: correct/wrong/missing-file.
- `resolveDest`: --dest > XDG_CACHE_HOME > $HOME/.cache.
- `loadManifest`: explicit-path success + invalid-JSON error.
- `download`: 200-OK round-trip and 404 surfacing via `httptest`.

### Python unit (runtime + handler)

- `_build_providers`: cpu/cuda/trt/auto + the strict-mode raise paths.
  Silent CPU fallback would re-introduce a class of production bugs
  the strict mode was added to prevent — these tests are the
  tripwire.
- `_preprocess` / `_postprocess_and_save`: shape, dtype, range
  invariants. Catches HW vs WH swaps and clip range bugs.
- `InputPayload`: pydantic validation (single-input requirement,
  format restriction).
- `_fetch_image_bytes`: all four input modes including the
  `data:image/png;base64,` prefix that web uploads send.
- `_gpu_sm_arch`: nvidia-smi parsing + missing-binary handling.
- `_resolve_model_path`: the strict-mode branches that determine
  whether a worker boots or fails. The trt-without-engine,
  trt-without-sm, and fetch-failure cases are the most failure-prone
  surfaces.

### Property-based (`hypothesis`)

- `_preprocess` shape invariant for any (w, h) in [64, 1280]² —
  catches HxW swaps that wouldn't show on square inputs.
- Round-trip pixel equality (preprocess → postprocess → reload) within
  ±1 per channel — flags clip-range and dtype-cast bugs.
- Below-engine-min preprocess still works — preprocess shouldn't
  silently tighten to the engine's optimisation profile.

### Live (`pytest -m live`)

- Output dimensions are exactly 4× input dimensions for a
  representative size set (boundary 64×64, small square, non-square
  to catch HW swaps, opt-profile point 720×720). The handler's
  `output_resolution` claim is cross-checked against the decoded
  PIL size of the returned base64 image.
- Diagnostics carry an active GPU provider — guards against silent
  CPU degradation.
- Oversize input (>1280) rejected before GPU work begins.
- Empty input produces a clean validation error.
- Warm-job latency is below a threshold — guards against the worker
  cycling between jobs (idle-timeout misconfig).

Live tests are NOT property-based. Hypothesis would generate dozens
of examples per test, each costing real money on the GPU.
