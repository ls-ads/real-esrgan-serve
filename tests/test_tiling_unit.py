"""Unit + property tests for runtime/tiling.py.

The tile-based path is correctness-critical: a buggy slice/stitch
shows up as visible seams on every output above 1280². Tests here
guard the math (slice positions cover the full image, stitch
reconstructs the input under an identity infer fn) before the live
RunPod tests blow $1+ of GPU time chasing the same regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from PIL import Image

from runtime import tiling


# ───────────────────────────────────────────────────────────────────────
# slice_positions math
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dim,tile,expected",
    [
        (1024, 1024, [0]),  # exactly tile-sized → single tile
        (1023, 1024, [0]),  # smaller than tile → single tile
        (1, 1024, [0]),  # tiny image → single tile
        (1280, 1024, [0, 256]),  # default cap → two tiles, 768-wide overlap
        # 2048 needs 3 tiles: a single step of 992 (= tile - min_overlap)
        # leaves the last 32 px uncovered, so the algo bumps n_tiles to
        # 3 and spaces them evenly with 512-wide overlap.
        (2048, 1024, [0, 512, 1024]),
        (4096, 1024, [0, 768, 1536, 2304, 3072]),  # 5 evenly-spaced
    ],
)
def test_slice_positions_known_cases(dim, tile, expected):
    assert tiling.slice_positions(dim, tile, min_overlap=32) == expected


def test_slice_positions_first_is_zero_last_ends_at_dim():
    """Two invariants the stitch algorithm relies on: first tile
    starts at 0 (covers left edge) and last tile ends at dim
    (covers right edge). Without these, edge pixels would have
    weight 0 in the canvas."""
    for dim in [1281, 1500, 2000, 2049, 4097]:
        positions = tiling.slice_positions(dim, tile=1024, min_overlap=32)
        assert positions[0] == 0
        assert positions[-1] + 1024 == dim, (
            f"last tile at dim={dim}: positions={positions}, "
            f"end={positions[-1] + 1024} != dim={dim}"
        )


def test_slice_positions_cover_every_pixel():
    """Walk every dim from 1280 to 4096; for every (dim, tile=1024,
    overlap=32) pair, assert the union of tile ranges covers [0, dim]
    with no gaps. A gap of even 1 pixel would leave that column
    weight=0 in the stitch canvas."""
    for dim in range(1280, 4097, 17):  # step 17 to keep it fast
        positions = tiling.slice_positions(dim, tile=1024, min_overlap=32)
        covered = np.zeros(dim, dtype=bool)
        for p in positions:
            covered[p : p + 1024] = True
        assert covered.all(), f"gap in coverage for dim={dim}"


# ───────────────────────────────────────────────────────────────────────
# Stitch correctness — under identity infer, output == 4× nearest
# ───────────────────────────────────────────────────────────────────────


def _identity_4x(chw):
    """Synthetic infer fn: takes (1, 3, h, w), returns (1, 3, 4h, 4w)
    via nearest-neighbor 4× upsample. This is what we'd get from a
    'perfect' Real-ESRGAN: every block of 4×4 output pixels equals
    the corresponding input pixel.

    For tiling correctness we don't care about model-quality
    plausibility — we care that slice→infer→stitch reconstructs the
    upsampled-by-4 input pixel-for-pixel."""
    return np.repeat(np.repeat(chw, 4, axis=2), 4, axis=3).astype(np.float32)


def _reference_4x(img: Image.Image) -> Image.Image:
    """Ground-truth 4× nearest-neighbor upsample, in PIL — what the
    tiled path should produce when the infer fn is identity."""
    return img.resize((img.width * 4, img.height * 4), Image.NEAREST)


@pytest.mark.parametrize(
    "w,h",
    [
        (64, 64),  # smaller than tile — single-shot path inside upscale_tiled
        (1024, 1024),  # exactly tile-sized — single-shot path
        (1280, 720),  # 16:9 HD, fits a single tile
        (1281, 720),  # one px over tile in W — triggers slicing
        (2000, 1500),  # comfortable 2K
        (3840, 2160),  # 4K source
    ],
)
def test_upscale_tiled_reconstructs_under_identity(w, h):
    """The tile path's job: produce output identical to what the
    untiled path would produce. Under an identity-4× infer fn, that's
    exactly nearest-neighbor 4× upsample of the input. Comparing
    pixel-for-pixel is the strongest seam test we can run without
    GPU."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    out = tiling.upscale_tiled(img, _identity_4x)
    expected = _reference_4x(img)

    assert out.size == expected.size, f"size {out.size} != {expected.size}"
    a = np.asarray(out, dtype=np.int16)
    b = np.asarray(expected, dtype=np.int16)
    # Round-trip through float32 [0..1] and back to uint8 in
    # upscale_tiled introduces ≤1-LSB rounding noise. ±1 tolerance
    # is the right boundary — any larger drift is a real bug.
    diff = np.abs(a - b)
    assert diff.max() <= 1, f"max pixel diff {diff.max()} for {w}×{h} input"


# ───────────────────────────────────────────────────────────────────────
# Property: any input dim within range stays seamless under identity
# ───────────────────────────────────────────────────────────────────────


@given(
    w=st.integers(min_value=64, max_value=2048),
    h=st.integers(min_value=64, max_value=2048),
)
@example(w=1281, h=64)  # one-px-over-tile in W
@example(w=64, h=1281)  # one-px-over-tile in H
@example(w=1281, h=1281)  # both axes triggering slicing
@example(w=2048, h=64)  # extreme aspect, slicing on W
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_identity_reconstruction_property(w, h):
    """Hypothesis sweep: for arbitrary dims in the supported range, the
    tiled path under identity infer matches a plain 4× upsample.
    Catches off-by-one errors at tile boundaries the parametrized
    tests above might miss (e.g., for prime-numbered dims)."""
    rng = np.random.default_rng(seed=hash((w, h)) & 0xFFFFFFFF)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    out = tiling.upscale_tiled(img, _identity_4x)
    expected = _reference_4x(img)

    assert out.size == expected.size
    diff = np.abs(np.asarray(out, dtype=np.int16) - np.asarray(expected, dtype=np.int16))
    assert diff.max() <= 1, f"seam at {w}×{h}: max diff {diff.max()}"


# ───────────────────────────────────────────────────────────────────────
# needs_tiling: handler-side decision matches the tile path's behaviour
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "w,h,expected",
    [
        (64, 64, False),
        (1024, 1024, False),
        (1280, 720, True),  # 1280 > tile=1024 → needs tiling
        (1024, 1280, True),
        (1281, 1281, True),
        (4096, 4096, True),
    ],
)
def test_needs_tiling(w, h, expected):
    assert tiling.needs_tiling(w, h, tile=1024) is expected


# ───────────────────────────────────────────────────────────────────────
# Stitch behaviour at edges: weight 0 should never appear
# ───────────────────────────────────────────────────────────────────────


def test_stitch_corners_match_identity_input():
    """A single corner pixel landing in only one tile's blend zone is
    fine, but two adjacent fades that both go to 0 at the boundary
    would leave the corner with weight 0 — visible as a black or
    NaN'd pixel. Force the multi-tile case with a 2000×2000 input
    (slices into 2×2) and assert the output corners exactly match
    the input under identity-4× upsampling."""
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(1, 256, size=(2000, 2000, 3), dtype=np.uint8)  # avoid 0 to make leaks obvious
    img = Image.fromarray(arr, mode="RGB")
    out = np.asarray(tiling.upscale_tiled(img, _identity_4x), dtype=np.int16)
    inp = arr.astype(np.int16)
    # Corner output pixels should equal the corresponding corner input
    # pixel exactly (modulo ±1 rounding) — identity 4× repeats each
    # input pixel into a 4×4 output block, and the corners aren't in
    # any blend zone.
    for (oy, ox), (iy, ix) in [
        ((0, 0), (0, 0)),
        ((0, -1), (0, -1)),
        ((-1, 0), (-1, 0)),
        ((-1, -1), (-1, -1)),
    ]:
        assert np.abs(out[oy, ox] - inp[iy, ix]).max() <= 1, (
            f"corner ({oy},{ox}) drifted from input ({iy},{ix}): "
            f"{out[oy, ox]} vs {inp[iy, ix]}"
        )
