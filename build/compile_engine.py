#!/usr/bin/env python3
"""Stage B: compile a TensorRT engine from a Real-ESRGAN ONNX file.

Runs on the target GPU hardware. The resulting `.engine` is pinned
to that GPU's SM architecture + the host's TRT major.minor version,
so we encode both into the output filename:

    realesrgan-x4plus-<gpu-class>-<sm-arch>-trt<X.Y>_fp16.engine

e.g. `realesrgan-x4plus-rtx-4090-sm89-trt10.8_fp16.engine`.

`real-esrgan-serve fetch-model --variant engine --gpu-class rtx-4090`
on the consumer side picks the matching artefact from GH Releases;
if no engine exists for the host's GPU, the runtime falls back to
the `.onnx` and lets onnxruntime's TRT EP build + cache locally
(slower first request, same eventual perf).

Usage:
    python build/compile_engine.py \\
        --onnx build/dist/realesrgan-x4plus_fp16.onnx \\
        --auto-detect-gpu \\
        --output build/dist/

    python build/compile_engine.py \\
        --onnx build/dist/realesrgan-x4plus_fp16.onnx \\
        --gpu-class rtx-4090 \\
        --sm-arch sm89 \\
        --trt-version 10.8 \\
        --output build/dist/

The `--auto-detect-gpu` path runs nvidia-smi + tensorrt module
queries; the manual flags are escape hatches for headless CI where
detection might be wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

# Build memory limits — TRT may use a workspace up to this size for
# tactic search. 4GB fits comfortably on consumer 24GB cards while
# leaving room for the running process + other models. Increase for
# H100/A100 if perf tuning shows benefit.
DEFAULT_WORKSPACE_BYTES = 4 << 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Canonical short forms for nvidia-smi's verbose gpu_name output.
# nvidia-smi reports e.g. "NVIDIA GeForce RTX 4090"; sanitised that
# becomes "nvidia-geforce-rtx-4090", which is awkward to type and
# doesn't match the keys consumers query for via
# `fetch-model --gpu-class rtx-4090`. This map collapses each common
# long form to its short canonical form. Stays in sync with
# remote_build.py's GPU_CLASS_TO_TYPE keys.
#
# Unknown GPUs fall through to the sanitised long form so adding a
# new GPU class is a one-line PR (here + remote_build.py) rather
# than a silent breakage.
_CANONICAL_GPU_CLASS: dict[str, str] = {
    "nvidia-geforce-rtx-5090":         "rtx-5090",
    "nvidia-geforce-rtx-5080":         "rtx-5080",
    "nvidia-geforce-rtx-4090":         "rtx-4090",
    "nvidia-geforce-rtx-4080-super":   "rtx-4080-s",
    "nvidia-geforce-rtx-4080":         "rtx-4080",
    "nvidia-geforce-rtx-3090-ti":      "rtx-3090-ti",
    "nvidia-geforce-rtx-3090":         "rtx-3090",
    "nvidia-rtx-a6000":                "rtx-a6000",
    "nvidia-rtx-6000-ada-generation":  "rtx-6000",
    "nvidia-rtx-pro-6000-blackwell":   "rtx-pro-6000",
    "nvidia-l40s":                     "l40s",
    "nvidia-l40":                      "l40",
    "nvidia-l4":                       "l4",
    "nvidia-a40":                      "a40",
    "nvidia-a100-pcie-40gb":           "a100",
    "nvidia-a100-sxm4-40gb":           "a100-sxm",
    "nvidia-a100-sxm4-80gb":           "a100-sxm",
    "nvidia-h100-pcie":                "h100",
    "nvidia-h100-80gb-hbm3":           "h100-sxm",
    "nvidia-h100-nvl":                 "h100-nvl",
    "nvidia-h200":                     "h200",
    "nvidia-b200":                     "b200",
}


def _detect_gpu() -> tuple[str, str]:
    """Return (gpu_class, sm_arch). gpu_class is sanitised lowercase
    canonical short form (e.g. 'rtx-4090'); sm_arch is e.g. 'sm89'.

    Falls back to the sanitised long form for unknown GPUs so a new
    SKU produces a still-useful filename until the canonical map is
    updated."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=gpu_name,compute_cap",
             "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().split("\n")[0]
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        sys.exit(f"[B] nvidia-smi probe failed: {e}\n"
                 "    pass --gpu-class / --sm-arch manually instead")

    name, cap = (p.strip() for p in out.split(",", 1))
    sane = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]", "-", name.lower())).strip("-")
    canonical = _CANONICAL_GPU_CLASS.get(sane, sane)
    if canonical == sane and sane.startswith("nvidia-"):
        # Heuristic warning: an unrecognised nvidia-* probably wants
        # to be added to the canonical map for clean filenames.
        print(f"[B] note: unmapped GPU '{sane}' — using verbatim. "
              f"Consider adding to _CANONICAL_GPU_CLASS for cleaner output filenames.",
              file=sys.stderr)
    return canonical, "sm" + cap.replace(".", "")


def _detect_trt_version() -> str:
    """e.g. '10.8' — used in the output filename."""
    try:
        import tensorrt as trt
    except ImportError as e:
        sys.exit(f"[B] tensorrt python module missing: {e}\n"
                 "    install with: pip install -r build/requirements.txt")
    parts = trt.__version__.split(".")
    return f"{parts[0]}.{parts[1]}"


def compile_engine(
    onnx_path: Path,
    out_path: Path,
    workspace_bytes: int,
    fp16: bool,
) -> None:
    try:
        import tensorrt as trt
    except ImportError as e:
        sys.exit(f"[B] tensorrt python module missing: {e}")

    print(f"[B] building engine from {onnx_path.name}")
    print(f"[B] workspace = {workspace_bytes / (1<<30):.1f} GiB, fp16 = {fp16}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("    ", parser.get_error(i), file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimisation profile — Real-ESRGAN takes (1, 3, H, W) where H,W
    # vary. Min/opt/max here let TRT pick tactics that work across the
    # whole input range without rebuilding per resolution. The 1280x1280
    # ceiling matches our handler's input cap.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 64, 64),
        opt=(1, 3, 720, 720),    # the median iosuite.io upload size
        max=(1, 3, 1280, 1280),  # the handler's accepted maximum
    )
    config.add_optimization_profile(profile)

    print("[B] compiling — this can take 10–30 min on first build")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        sys.exit("[B] engine build failed (TRT logger should have printed why)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(engine_bytes)
    print(f"[B] done: {out_path.name}  sha256={_sha256(out_path)}  bytes={out_path.stat().st_size}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--onnx", type=Path, required=True, help="input .onnx (from Stage A)")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "dist",
                   help="output directory (default: build/dist/)")
    p.add_argument("--auto-detect-gpu", action="store_true",
                   help="probe nvidia-smi for GPU class + SM arch")
    p.add_argument("--gpu-class", help="e.g. rtx-4090 (overrides auto-detect)")
    p.add_argument("--sm-arch", help="e.g. sm89 (overrides auto-detect)")
    p.add_argument("--trt-version", help="e.g. 10.8 (overrides auto-detect)")
    p.add_argument("--workspace-bytes", type=int, default=DEFAULT_WORKSPACE_BYTES)
    p.add_argument("--fp32", action="store_true", help="disable FP16 (default: FP16 on)")
    args = p.parse_args()

    if not args.onnx.exists():
        sys.exit(f"[B] onnx not found: {args.onnx}")

    if args.auto_detect_gpu and not (args.gpu_class and args.sm_arch):
        gpu_class, sm_arch = _detect_gpu()
        args.gpu_class = args.gpu_class or gpu_class
        args.sm_arch = args.sm_arch or sm_arch

    if not (args.gpu_class and args.sm_arch):
        sys.exit("[B] need --gpu-class + --sm-arch (or --auto-detect-gpu)")
    args.trt_version = args.trt_version or _detect_trt_version()

    precision = "fp32" if args.fp32 else "fp16"
    out_name = (
        f"realesrgan-x4plus-{args.gpu_class}-{args.sm_arch}-"
        f"trt{args.trt_version}_{precision}.engine"
    )
    out_path = args.output / out_name

    compile_engine(args.onnx, out_path, args.workspace_bytes, fp16=not args.fp32)
    return 0


if __name__ == "__main__":
    sys.exit(main())
