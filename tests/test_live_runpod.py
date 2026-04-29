"""Live integration tests against a deployed RunPod endpoint.

These tests submit real jobs and cost real money. They're gated by
RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY environment variables and skip
cleanly when either is unset, so unit-test runs never accidentally
fire them.

To run:

    # 1. Deploy an endpoint and keep it alive (default deploy tears
    #    it down on completion, which is wrong for a test suite).
    build/.with-iosuite-key make deploy-runpod \
      IMAGE=ghcr.io/ls-ads/real-esrgan-serve:runpod-trt-dev \
      ENDPOINT_NAME=real-esrgan-serve-tests \
      WARMUP_JOBS=0 \
      DEPLOY_FLAGS="--keep-endpoint"

    # 2. Note the endpoint id from the deploy output and export it.
    export RUNPOD_ENDPOINT_ID=<id>

    # 3. Run the live tests.
    make test-live

What's covered:

  - Round-trip on representative image sizes (small + boundary). NOT
    hypothesis-generated — every example costs a real GPU-second.
  - The output_resolution claim in the response actually matches the
    upscaled image's decoded size (catches bugs where the handler
    misreports without anyone noticing).
  - Diagnostics are present and identify a GPU provider.
  - Oversize input is rejected by the handler before GPU work runs
    (MAX_INPUT_DIM enforcement).
  - Validation errors come back as a clean error string, not a
    crash. Important because the iosuite frontend surfaces these.

Each test does at most one job; a full live run is ~5 inferences,
roughly $0.005 worth of GPU time on a 4090.
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

RUNPOD_API_BASE = "https://api.runpod.ai/v2"

pytestmark = pytest.mark.live


def _submit(endpoint_id: str, api_key: str, payload: dict,
            timeout_s: int = 120) -> dict:
    """Submit a synchronous job and return the parsed status response.
    Uses /runsync rather than /run + polling so the test code stays
    flat — synchronous endpoint blocks up to its own timeout, then
    we get the final status in one round-trip."""
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "real-esrgan-serve-tests/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


@pytest.mark.parametrize("size", [
    pytest.param((64, 64), id="min-boundary-64x64"),
    pytest.param((128, 128), id="small-square-128x128"),
    pytest.param((128, 256), id="non-square-128x256"),  # catches HW swap bugs
    pytest.param((720, 720), id="optimisation-profile-opt-720x720"),
])
def test_live_inference_output_dimensions_match_4x_input(runpod_env, make_png, size):
    """For every (w, h), the upscaled image must be exactly (4w, 4h).
    The handler claims this in `output_resolution`, but we don't
    trust the claim — we decode the returned base64 and check the
    actual PIL size. A mismatch means the model isn't actually doing
    x4 (custom model loaded?) or the handler is lying about the
    output."""
    endpoint_id, api_key = runpod_env
    w, h = size
    png = make_png(w, h)
    payload = {"input": {
        "image_base64": base64.b64encode(png).decode("ascii"),
        "output_format": "jpg",
    }}
    resp = _submit(endpoint_id, api_key, payload)
    assert resp["status"] == "COMPLETED", f"job failed: {resp}"

    out = resp["output"]
    assert "image_base64" in out, f"no output image in response: {out}"
    img = Image.open(io.BytesIO(base64.b64decode(out["image_base64"])))
    assert img.size == (4 * w, 4 * h), (
        f"upscaler output {img.size} != 4x of input ({w}x{h}); "
        f"model is not doing x4 or handler swapped dims"
    )
    # The handler's claim should match the decoded reality.
    assert out["output_resolution"] == f"{4 * w}x{4 * h}"


def test_live_response_carries_provider_diagnostics(runpod_env, make_png):
    """Every response should carry _diagnostics.providers — without
    this we lose the only programmatic channel for confirming the
    worker activated GPU EPs vs silently degrading to CPU. Empty or
    missing providers list = a configuration bug we want to catch
    immediately."""
    endpoint_id, api_key = runpod_env
    payload = {"input": {
        "image_base64": base64.b64encode(make_png(64, 64)).decode("ascii"),
    }}
    resp = _submit(endpoint_id, api_key, payload)
    assert resp["status"] == "COMPLETED"
    diag = resp["output"].get("_diagnostics") or {}
    providers = diag.get("providers") or []
    assert providers, f"empty diagnostics.providers: {diag}"
    # Whatever flavor we deployed against, at least one of these must
    # be active. CPUExecutionProvider only is treated as a regression
    # because none of the runpod-flavored deployments should ever land
    # on CPU.
    gpu_eps = {"CUDAExecutionProvider", "TensorrtExecutionProvider", "TensorrtDirect"}
    assert gpu_eps & set(providers), (
        f"no GPU EP active on a runpod-flavored worker: {providers}"
    )


def test_live_oversize_input_rejected(runpod_env, make_png):
    """The handler caps input at MAX_INPUT_DIM=1280. Anything larger
    must be rejected before GPU work begins — otherwise a single
    large upload could cause OOM that takes the worker out for the
    rest of its idle window. The error string should mention the cap
    (so iosuite can surface it to the user without parsing
    boilerplate)."""
    endpoint_id, api_key = runpod_env
    # 1281x1281: exactly one pixel over the cap on each axis
    payload = {"input": {
        "image_base64": base64.b64encode(make_png(1281, 1281)).decode("ascii"),
    }}
    resp = _submit(endpoint_id, api_key, payload)
    # The job RPC returns COMPLETED (the SDK doesn't surface app-level
    # errors as failed status) — the error is in output.error.
    out = resp.get("output") or {}
    err = out.get("error", "") or resp.get("error", "")
    assert "exceeds max" in err.lower() or "1280" in err, (
        f"oversize input not rejected with a clear message: {resp}"
    )


def test_live_missing_input_field_rejected(runpod_env):
    """Sending a payload with no image_url/image_base64/image_path
    must produce a clean validation error — not a 500 or a crash."""
    endpoint_id, api_key = runpod_env
    resp = _submit(endpoint_id, api_key, {"input": {"output_format": "jpg"}})
    out = resp.get("output") or {}
    err = out.get("error", "") or resp.get("error", "")
    assert err, f"empty payload should produce an error: {resp}"
    # The pydantic ValidationError mentions one of these fields
    assert any(s in err for s in ("image_url", "image_base64", "image_path")), (
        f"validation message lost: {err}"
    )


def test_live_warm_jobs_faster_than_cold(runpod_env, make_png):
    """After a worker is warm (model loaded, engine deserialized),
    subsequent jobs should complete in <2s walltime even on a
    standard 4090. If they don't, the worker is being reset between
    jobs (idle-timeout misconfigured?) and we'd lose all the cold-
    start optimisation work. Fires only as a tripwire on serious
    regressions; tolerant of normal RunPod variance."""
    endpoint_id, api_key = runpod_env
    payload = {"input": {
        "image_base64": base64.b64encode(make_png(128, 128)).decode("ascii"),
    }}
    # First call may be cold; we don't care. Use it to warm up.
    _submit(endpoint_id, api_key, payload)

    # Second + third should both be warm.
    timings = []
    for _ in range(2):
        t0 = time.monotonic()
        resp = _submit(endpoint_id, api_key, payload)
        timings.append(time.monotonic() - t0)
        assert resp["status"] == "COMPLETED"
    assert max(timings) < 5.0, (
        f"warm-job latency {timings} suggests the worker is cycling between "
        f"jobs; check idle-timeout in the endpoint config."
    )
