"""Tile-based upscaling for inputs larger than the engine's single-shot cap.

The engine profiles cap input at 1280×1280 (TRT INT32 element count + VRAM
on consumer cards). Bigger inputs go through this module instead: slice
into overlapping tiles, run inference per tile, blend overlap zones with
linear ramps so seams disappear.

Algorithm sketch:

    input ──slice──▶ N×N tiles (1024² each, ≥32 px overlap on shared edges)
                   │
                   ├─ each tile through the same inference path the
                   │  non-tiled code uses; output per tile is exactly
                   │  4× input on each axis (Real-ESRGAN-fixed factor).
                   │
                   └─stitch─▶ output canvas, with linear blend across
                              the FULL overlap region of each tile pair.
                              Two complementary linear ramps sum to 1.0
                              everywhere in the overlap → constant total
                              weight → no visible seams.

Memory ceiling: at 2048² input the working canvas is ~800 MB float32.
Larger inputs are gated by handler-side caps; streaming-strip processing
is a follow-up (see docs/IMAGE-IO.md § 6).

This module deliberately does not import onnxruntime or tensorrt — it
takes a callable that turns one preprocessed tile into one upscaled
tile, so the same code works against any inference backend (ORT, TRT,
or a unit-test stub).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PIL import Image

# Real-ESRGAN's fixed upscale factor. Output dims are exactly 4× input.
SCALE = 4

# Default tile geometry. Sized for the engine profile's `opt` shape — the
# 720² window is what tactic search optimised against, but 1024² stays
# inside the engine's max=(1, 3, 1280, 1280) and reduces tile count.
DEFAULT_TILE = 1024
DEFAULT_MIN_OVERLAP = 32

# Per-tile inference callable: (chw_uint8 numpy in [0..255]) → (chw_uint8 out at 4×).
# Helper passes a closure that wraps onnxruntime.session.run / TrtSession.run.
TileInferFn = Callable[[np.ndarray], np.ndarray]


def slice_positions(dim: int, tile: int, min_overlap: int) -> list[int]:
    """Compute tile starting positions covering [0, dim].

    Tiles are evenly distributed: the first sits at 0, the last ends
    flush with `dim`, and intermediates are spaced equally. Overlap
    between adjacent tiles is at least `min_overlap` (often more, since
    we round up the tile count).

    Examples (tile=1024, min_overlap=32):
      dim=1024 → [0]                        (single tile, no slicing)
      dim=1280 → [0, 256]                   (overlap = 768 px)
      dim=2048 → [0, 1024]                  (overlap = 0; min_overlap not honored)
      dim=2049 → [0, 512, 1025]             (overlap = 512 px each pair)

    For dim slightly above tile, the algorithm prefers FEWER tiles with
    LARGER overlap over more tiles with min overlap — keeps the per-edge
    blend region wide and reduces tile count.
    """
    if dim <= tile:
        return [0]
    step = max(1, tile - min_overlap)
    n_tiles = max(2, math.ceil((dim - tile) / step) + 1)
    if n_tiles == 1:
        return [0]
    return [i * (dim - tile) // (n_tiles - 1) for i in range(n_tiles)]


def _blend_mask(
    tile_h: int,
    tile_w: int,
    fade_top: int,
    fade_bottom: int,
    fade_left: int,
    fade_right: int,
) -> np.ndarray:
    """Return a (tile_h, tile_w) float32 mask in [0..1] with linear ramps
    on the specified edges.

    Each fade is the FULL overlap with the matching neighbour. With the
    neighbour's complementary ramp, the two masks sum to 1.0 everywhere
    in the overlap — perfect blending, no visible seams.
    """
    mask = np.ones((tile_h, tile_w), dtype=np.float32)
    if fade_top > 0:
        ramp = np.linspace(0.0, 1.0, fade_top, endpoint=True, dtype=np.float32)
        mask[:fade_top, :] *= ramp[:, None]
    if fade_bottom > 0:
        ramp = np.linspace(1.0, 0.0, fade_bottom, endpoint=True, dtype=np.float32)
        mask[-fade_bottom:, :] *= ramp[:, None]
    if fade_left > 0:
        ramp = np.linspace(0.0, 1.0, fade_left, endpoint=True, dtype=np.float32)
        mask[:, :fade_left] *= ramp[None, :]
    if fade_right > 0:
        ramp = np.linspace(1.0, 0.0, fade_right, endpoint=True, dtype=np.float32)
        mask[:, -fade_right:] *= ramp[None, :]
    return mask


def upscale_tiled(
    img: Image.Image,
    infer: TileInferFn,
    tile: int = DEFAULT_TILE,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> Image.Image:
    """Tile-based upscale of a PIL image. Returns a 4×-size PIL image.

    `infer` accepts NCHW float32 in [0..1] of shape (1, 3, h, w) and
    returns the same shape × 4 on the spatial axes (anything ORT or
    TrtSession is happy to consume).

    For images that fit in a single tile (≤ `tile` × `tile`), this is
    one inference call — no slicing or stitching, no quality loss vs.
    the non-tiled path.
    """
    img = img.convert("RGB")
    src_w, src_h = img.size

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr.transpose(2, 0, 1)[None, :, :, :]  # (1, 3, H, W)

    if src_w <= tile and src_h <= tile:
        # Single-shot path. Avoids the canvas allocation entirely.
        out_chw = infer(arr)  # (1, 3, 4H, 4W)
        return _to_pil(out_chw[0])

    xs = slice_positions(src_w, tile, min_overlap)
    ys = slice_positions(src_h, tile, min_overlap)

    out_h = src_h * SCALE
    out_w = src_w * SCALE
    canvas = np.zeros((3, out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            tile_in = arr[:, :, y : y + tile, x : x + tile]  # (1, 3, t_h, t_w)
            tile_in = np.ascontiguousarray(tile_in)
            tile_out = infer(tile_in)[0].astype(np.float32)  # (3, 4·t_h, 4·t_w)
            th, tw = tile_out.shape[-2:]

            # Fade widths on each edge equal the actual overlap with the
            # neighbour (in OUTPUT coordinates → multiply by SCALE).
            left_fade = ((xs[xi - 1] + tile) - x) * SCALE if xi > 0 else 0
            right_fade = ((x + tile) - xs[xi + 1]) * SCALE if xi < len(xs) - 1 else 0
            top_fade = ((ys[yi - 1] + tile) - y) * SCALE if yi > 0 else 0
            bottom_fade = ((y + tile) - ys[yi + 1]) * SCALE if yi < len(ys) - 1 else 0

            # Clamp to tile size — degenerate cases when overlap > tile dim.
            left_fade = min(left_fade, tw)
            right_fade = min(right_fade, tw)
            top_fade = min(top_fade, th)
            bottom_fade = min(bottom_fade, th)

            mask = _blend_mask(th, tw, top_fade, bottom_fade, left_fade, right_fade)

            ox, oy = x * SCALE, y * SCALE
            canvas[:, oy : oy + th, ox : ox + tw] += tile_out * mask[None, :, :]
            weight[oy : oy + th, ox : ox + tw] += mask

    # Edge-of-canvas pixels with weight 0 shouldn't happen given the
    # slice algorithm always covers [0, dim], but guard anyway so a
    # corner pixel doesn't NaN out.
    weight = np.where(weight > 0, weight, 1.0)
    canvas /= weight[None, :, :]
    return _to_pil(canvas)


def _to_pil(chw: np.ndarray) -> Image.Image:
    """(3, H, W) float in [0..1] → PIL RGB image."""
    arr = (chw.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def needs_tiling(width: int, height: int, tile: int = DEFAULT_TILE) -> bool:
    """True when input is large enough to require slicing. Helper uses
    this to decide whether to enter the tiled code path or pass through
    to the single-shot one. Match against the same `tile` value used in
    upscale_tiled to keep the decision consistent."""
    return width > tile or height > tile
