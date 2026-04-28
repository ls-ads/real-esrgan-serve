#!/usr/bin/env python3
"""Stage A: export Real-ESRGAN PyTorch weights to ONNX.

Designed to run INSIDE build/Dockerfile.export — the dep resolution
is the gnarly part of this pipeline, see that Dockerfile's header
for the full reasoning. Run locally only if you've already wrestled
basicsr/torch/numpy versions into a working set yourself.

Two ways to invoke:

  Containerised (recommended):
    docker build -f build/Dockerfile.export -t res-export build/
    docker run --rm -v $(pwd)/build/dist:/output res-export
    # produces /output/realesrgan-x4plus_fp16.onnx + _fp32.onnx

  Local (you've handled deps yourself):
    cd build && python export_onnx.py \\
        --input weights/RealESRGAN_x4plus.pth \\
        --output ./dist/

FP16 conversion happens AFTER export via onnxconverter-common, not
by calling model.half() before export. Pre-export half-precision
produces graphs that some TRT versions can't ingest cleanly; the
post-export converter keeps op dtypes consistent. This is the same
fix the previous version of this repo eventually landed on (see
commit e1de3fd in the pre-rebuild git history if you need archeology).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# Upstream weight pin. Verified at run time; mismatch deletes the
# partial file and exits non-zero. Update deliberately when bumping
# to a new upstream Real-ESRGAN release.
UPSTREAM_PTH_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
UPSTREAM_PTH_SHA256 = (
    "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
)

# ONNX opset. 14 is what the previous (working) export used; TensorRT
# 8.x/10.x both ingest it cleanly. Opset 17 works too with onnxruntime
# 1.18+ but adds nothing the model uses, so staying on 14 keeps wider
# TRT compatibility.
ONNX_OPSET = 14

# Real-ESRGAN inputs/outputs are dynamic on H/W. dynamic_axes lets
# onnxruntime + TRT handle variable-resolution inputs without
# rebuilding the engine per shape.
DYNAMIC_AXES = {
    "input":  {0: "batch_size", 2: "height", 3: "width"},
    "output": {0: "batch_size", 2: "height", 3: "width"},
}

# Dummy input for graph tracing. Shape is just a hint — dynamic_axes
# above keeps the resulting graph size-agnostic.
TRACE_INPUT_SHAPE = (1, 3, 64, 64)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_upstream(cache_dir: Path) -> Path:
    """Download (or reuse) the upstream .pth, verify SHA-256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "RealESRGAN_x4plus.pth"
    if dest.exists() and _sha256(dest) == UPSTREAM_PTH_SHA256:
        print(f"[A] using cached upstream: {dest}")
        return dest

    print(f"[A] fetching {UPSTREAM_PTH_URL}")
    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(UPSTREAM_PTH_URL) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    actual = _sha256(tmp)
    if actual != UPSTREAM_PTH_SHA256:
        tmp.unlink()
        sys.exit(
            f"[A] upstream SHA-256 mismatch — got {actual}, "
            f"expected {UPSTREAM_PTH_SHA256}. Refusing to proceed."
        )
    tmp.rename(dest)
    print(f"[A] verified + cached: {dest}")
    return dest


def export_fp32(pth_path: Path, output_path: Path, key: str) -> None:
    """torch → ONNX (FP32). FP16 is a separate post-step."""
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError as e:
        sys.exit(
            f"[A] missing dependency: {e}\n"
            "    The recommended path is the build/Dockerfile.export image —\n"
            "    its uv `--exclude-newer` resolution is what makes basicsr\n"
            "    actually install. Local installs are fragile.\n"
        )

    print(f"[A] loading weights: {pth_path}  (key={key})")
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4,
    )
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state[key], strict=True)
    model.eval()
    model.cpu()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(*TRACE_INPUT_SHAPE)

    print(f"[A] exporting FP32 → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=ONNX_OPSET,
            dynamic_axes=DYNAMIC_AXES,
            export_params=True,
            do_constant_folding=True,
        )

    print(f"[A] FP32 done: sha256={_sha256(output_path)}  bytes={output_path.stat().st_size}")


def convert_to_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Post-export FP16 conversion via onnxconverter-common.

    NOT model.half() before export — that produces graphs some TRT
    versions can't ingest. This path keeps op dtypes consistent."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        sys.exit(f"[A] need onnx + onnxconverter-common for FP16 conversion: {e}")

    print(f"[A] converting → FP16  ({fp32_path.name} -> {fp16_path.name})")
    model_fp32 = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(fp16_path))
    print(f"[A] FP16 done: sha256={_sha256(fp16_path)}  bytes={fp16_path.stat().st_size}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path,
                   help="path to upstream .pth (skipped if not given — script fetches)")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "dist",
                   help="output directory (default: build/dist/)")
    p.add_argument("--cache", type=Path,
                   default=Path(__file__).parent / ".cache",
                   help="upstream .pth download cache (default: build/.cache/)")
    p.add_argument("--key", choices=["params", "params_ema"], default="params_ema",
                   help="which state dict to load (RealESRGAN ships both — params_ema is the official inference target)")
    p.add_argument("--variant", choices=["fp16", "fp32", "both"], default="both",
                   help="which precision(s) to export (default: both)")
    args = p.parse_args()

    pth = args.input if args.input and args.input.exists() else _fetch_upstream(args.cache)

    fp32_path = args.output / "realesrgan-x4plus_fp32.onnx"
    fp16_path = args.output / "realesrgan-x4plus_fp16.onnx"

    # Always export FP32 first; FP16 is derived. Ordering matters
    # because convert_float_to_float16 reads from disk.
    if args.variant in ("fp32", "both"):
        export_fp32(pth, fp32_path, key=args.key)
    elif args.variant == "fp16":
        # Need FP32 on disk to convert from; produce it transiently
        # if the caller only asked for FP16.
        export_fp32(pth, fp32_path, key=args.key)

    if args.variant in ("fp16", "both"):
        convert_to_fp16(fp32_path, fp16_path)
        if args.variant == "fp16":
            # Caller asked only for fp16 — don't leave fp32 lying around
            fp32_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
