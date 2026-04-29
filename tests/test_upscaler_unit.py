"""Unit tests for runtime/upscaler.py.

Covered:
  - _build_providers: provider-name → ORT providers list. The strict
    flag (cpu/cuda/trt) vs the silent-fallback (auto) distinction is
    the most failure-prone branch in this file — silent CPU fallback
    cost us an iteration cycle in production, so getting it right
    matters.
  - _preprocess + _postprocess_and_save: image I/O round-trip.
    Ensures the float-clip + uint8 cast doesn't silently mangle pixels
    that came in valid.

Not covered (intentional):
  - TrtSession: requires a real GPU and a built engine to exercise
    meaningfully. Mocking it would test the mock, not the integration.
    Live tests cover this surface end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from runtime import upscaler


# ────────────────────────────────────────────────────────────────────
# _build_providers — provider selection logic
# ────────────────────────────────────────────────────────────────────

ORT_FULL = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
ORT_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ORT_CPU_ONLY = ["CPUExecutionProvider"]


def _names(providers):
    """Strip provider-options dicts so we can assert on names alone."""
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def test_build_providers_cpu_explicit():
    got = upscaler._build_providers("cpu", gpu_id=0, available=ORT_FULL,
                                    trt_cache=Path("/tmp/x"))
    assert got == ["CPUExecutionProvider"]


def test_build_providers_cpu_when_gpu_id_negative():
    """gpu_id=-1 is the documented "force CPU" sentinel from the Go
    side; it must beat the provider= argument because the underlying
    GPU isn't selectable."""
    got = upscaler._build_providers("cuda", gpu_id=-1, available=ORT_FULL,
                                    trt_cache=Path("/tmp/x"))
    assert got == ["CPUExecutionProvider"]


def test_build_providers_cuda_strict_when_available():
    got = upscaler._build_providers("cuda", gpu_id=0, available=ORT_CUDA,
                                    trt_cache=Path("/tmp/x"))
    assert _names(got) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_build_providers_cuda_strict_raises_when_not_in_wheel():
    """The strict-mode contract: asking for cuda from a CPU-only ORT
    wheel must raise, not silently degrade. This is the safety net
    that protects against shipping a misconfigured runtime image."""
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider not in"):
        upscaler._build_providers("cuda", gpu_id=0, available=ORT_CPU_ONLY,
                                  trt_cache=Path("/tmp/x"))


def test_build_providers_trt_strict_raises_when_not_in_wheel():
    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider not"):
        upscaler._build_providers("trt", gpu_id=0, available=ORT_CUDA,
                                  trt_cache=Path("/tmp/x"))


def test_build_providers_auto_walks_trt_first_when_available():
    got = upscaler._build_providers("auto", gpu_id=0, available=ORT_FULL,
                                    trt_cache=Path("/tmp/x"))
    assert _names(got) == ["TensorrtExecutionProvider",
                           "CUDAExecutionProvider",
                           "CPUExecutionProvider"]


def test_build_providers_auto_falls_to_cuda_when_no_trt():
    got = upscaler._build_providers("auto", gpu_id=0, available=ORT_CUDA,
                                    trt_cache=Path("/tmp/x"))
    assert _names(got) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_build_providers_auto_to_cpu_when_only_cpu_available():
    got = upscaler._build_providers("auto", gpu_id=0, available=ORT_CPU_ONLY,
                                    trt_cache=Path("/tmp/x"))
    assert got == ["CPUExecutionProvider"]


# ────────────────────────────────────────────────────────────────────
# _preprocess / _postprocess_and_save — image I/O round-trip
# ────────────────────────────────────────────────────────────────────

def test_preprocess_returns_nchw_float32_normalized(tmp_image):
    """The exported ONNX expects (1, 3, H, W) float32 in [0, 1]. If
    _preprocess drifts on any of those (e.g. emits HWC, uint8, or
    [0, 255]), every inference is silently wrong — image looks like
    garbage but no error fires."""
    p = tmp_image(128, 96)  # non-square to catch H/W swaps
    chw, w, h = upscaler._preprocess(p)
    assert chw.shape == (1, 3, 96, 128)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() and chw.max() <= 1.0
    assert (w, h) == (128, 96)


def test_preprocess_grayscale_promoted_to_rgb(tmp_path, make_png):
    """Operators sometimes feed grayscale PNGs in. PIL's RGB conversion
    happens inside _preprocess, so we should still get 3 channels out."""
    grey = Image.new("L", (64, 64), 128)
    p = tmp_path / "grey.png"
    grey.save(p)
    chw, _, _ = upscaler._preprocess(p)
    assert chw.shape[1] == 3  # promoted to RGB


def test_postprocess_clips_out_of_range_values(tmp_path):
    """Inference can occasionally emit values slightly outside [0, 1]
    (FP16 numerics, clipping ops in the model). The postprocess must
    saturate rather than wrap, or pixels overflow uint8 modular and
    you get neon speckle. Test with deliberately out-of-range values."""
    # NCHW shape, deliberately covers <0 and >1.
    arr = np.array([[
        [[-0.5, 0.5, 1.5]],
        [[0.0, 0.5, 1.0]],
        [[0.5, 0.5, 0.5]],
    ]], dtype=np.float32)
    out = tmp_path / "out.png"
    # _postprocess_and_save expects NCHW: it slices [0] internally to
    # drop the batch dim. Pass the full 4-D array, not arr[0].
    upscaler._postprocess_and_save(arr, out)
    img = np.asarray(Image.open(out))  # HWC uint8
    # Channel R: -0.5 → 0, 0.5 → 128, 1.5 → 255
    assert img[0, 0, 0] == 0
    assert 126 <= img[0, 1, 0] <= 130  # ~128 ± rounding
    assert img[0, 2, 0] == 255


def test_run_inference_dispatches_to_trt_session_when_present(monkeypatch):
    """`_run_inference` is the dispatch point between ORT and TRT-direct.
    Forgetting to wrap the TrtSession output in a list would crash
    `_postprocess_and_save` (which expects result[0]). Verify the
    wrapping happens."""
    class FakeTrt:
        def run(self, chw):
            # Mimics TrtSession.run returning a single ndarray
            return np.zeros((1, 3, 4, 4), dtype=np.float32)

    # Patch isinstance check by injecting a fake TrtSession class
    monkeypatch.setattr(upscaler, "TrtSession", FakeTrt)
    chw = np.zeros((1, 3, 1, 1), dtype=np.float32)
    result = upscaler._run_inference(FakeTrt(), chw)
    # _postprocess_and_save consumes result[0] — must be subscriptable
    assert hasattr(result, "__getitem__")
    assert result[0].shape == (1, 3, 4, 4)
