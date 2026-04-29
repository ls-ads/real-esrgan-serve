"""Property-based tests for image sizing.

The runtime has multiple places where image dimensions flow through:

  - _preprocess: PIL → NCHW float32. Shape contract: (1, 3, h, w).
  - The TRT engine: dynamic-shape profile (min 64x64, opt 720x720,
    max 1280x1280); set_input_shape rejects anything outside.
  - The handler: MAX_INPUT_DIM=1280 cap before GPU work begins.
  - The output: every model is x4 upscale. output_resolution must
    equal {4w}x{4h}.

Each surface is independently testable; getting any of them subtly
wrong means silent geometry bugs (transposed images, wrong aspect
ratios, off-by-one crops) that only show up on visual inspection.
Property-based testing here exercises a wide swath of (w, h) pairs
cheaply without us having to enumerate boundary cases by hand.

Live tests are NOT property-based here — each example would cost a
real RunPod inference. test_live_runpod.py covers a deterministic
boundary set instead.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from PIL import Image

from runtime import upscaler

# Hypothesis warns when a function-scoped fixture (tmp_path /
# make_png) is reused across generated examples. In our case the
# per-example reuse is fine — each example writes a fresh PNG to the
# same path or a new one — so the suppress_health_check on each
# @settings below silences it.


# Strategy: width/height pairs spanning the engine's full optimisation
# profile. Lower bound 64 (engine min), upper bound 1280 (handler cap +
# engine max). We use multiples of 4 so we don't have to reason about
# subpixel rounding in the round-trip — the upscaler itself is x4 so
# any input dim is fine, but the round-trip through PIL save/reload
# is bit-exact only when there's no JPEG re-encode involved.
_DIM = st.integers(min_value=64, max_value=1280)


@given(w=_DIM, h=_DIM)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_preprocess_shape_invariant(w, h, tmp_path, make_png):
    """For any (w, h) in the supported range, _preprocess emits exactly
    shape (1, 3, h, w) — H comes BEFORE W in NCHW, the easy place to
    flip and not notice on a square image. Mixed dims catch it."""
    p = tmp_path / "in.png"
    p.write_bytes(make_png(w, h))
    chw, gw, gh = upscaler._preprocess(p)
    assert chw.shape == (1, 3, h, w)
    assert (gw, gh) == (w, h)
    # Channel range invariant: must always be in [0, 1] for the model
    assert chw.min() >= 0.0
    assert chw.max() <= 1.0


@given(w=_DIM, h=_DIM)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_preprocess_postprocess_roundtrip_pixel_equality(w, h, tmp_path, make_png):
    """Round-trip a generated image: PIL load → preprocess → fake-pass
    (no model) → postprocess → PIL load → compare. The full pipeline
    has rounding (uint8↔float32) but should be lossless to ±1 per
    channel (the +0.5 then truncate convention). Drift wider than
    that is a bug — either the clip range is wrong or the dtype
    conversion is."""
    src_bytes = make_png(w, h)
    src = np.asarray(Image.open(io.BytesIO(src_bytes)).convert("RGB"))

    p_in = tmp_path / "in.png"
    p_in.write_bytes(src_bytes)
    p_out = tmp_path / "out.png"

    chw, _, _ = upscaler._preprocess(p_in)
    # Identity pass — the transformation we want to exercise is
    # JUST preprocess + postprocess, with no model in between.
    upscaler._postprocess_and_save(chw, p_out)
    rt = np.asarray(Image.open(p_out).convert("RGB"))

    assert rt.shape == src.shape
    # Allow ±1 per channel for the float→uint8 round (255.0 + 0.5 cast
    # introduces at most one ULP of error on each pixel).
    diff = np.abs(rt.astype(np.int16) - src.astype(np.int16))
    assert int(diff.max()) <= 1, (
        f"round-trip pixel diff {diff.max()} > 1 at ({w}x{h}) — "
        f"check _postprocess clip/scale logic"
    )


@given(w=st.integers(min_value=1, max_value=63))
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_preprocess_below_engine_min_still_loads(w, tmp_path, make_png):
    """The engine's min profile is 64x64, but _preprocess itself has
    no lower bound — it's a dumb image-to-tensor function. The TRT
    session is the one that rejects sub-64; preprocess should just
    work, and the rejection lands cleanly later. Ensures we don't
    accidentally tighten preprocess to the engine constraint."""
    p = tmp_path / "tiny.png"
    p.write_bytes(make_png(w, w))
    chw, _, _ = upscaler._preprocess(p)
    assert chw.shape == (1, 3, w, w)


def test_postprocess_output_dimensions_preserved(tmp_path):
    """Whatever NCHW shape comes in, the saved image's dims should
    match (W, H) of the input array — no aspect-ratio swap. This
    is the property-based reduction of the previous round-trip
    test, but explicitly stated so a regression that breaks shape
    fidelity is its own test failure."""
    # NCHW = (1, 3, 100, 200): 100 rows, 200 columns
    arr = np.full((1, 3, 100, 200), 0.5, dtype=np.float32)
    out = tmp_path / "out.png"
    upscaler._postprocess_and_save(arr, out)
    img = Image.open(out)
    assert img.size == (200, 100)  # PIL.Image.size = (W, H)
