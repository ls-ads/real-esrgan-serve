"""Unit tests for providers/runpod/handler.py.

Covered:
  - InputPayload validation (the single-input requirement, output
    format restriction).
  - _fetch_image_bytes: base64 / image_path / image_url branches.
  - _gpu_sm_arch: nvidia-smi parsing + missing-binary handling.
  - _resolve_model_path: the strict-mode branches that drive whether
    a worker boots or fails. Particularly important to test because
    silent fallback would re-introduce JIT cost the design forbids.
"""
from __future__ import annotations

import base64
import io
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image
from pydantic import ValidationError

import handler


# ────────────────────────────────────────────────────────────────────
# InputPayload validation
# ────────────────────────────────────────────────────────────────────

def test_input_payload_requires_at_least_one_image_source():
    """The pydantic model_validator rejects a payload with no input.
    Without this, the handler would happily start an inference call
    on a None image and fail with a confusing AttributeError 1000ms
    in instead of a clean 400-equivalent."""
    with pytest.raises(ValidationError, match="image_url|image_base64|image_path"):
        handler.InputPayload(output_format="jpg")


def test_input_payload_accepts_any_single_source():
    for kwargs in (
        {"image_url": "https://example.test/x.png"},
        {"image_base64": "AAAA"},
        {"image_path": "/workspace/in.jpg"},
    ):
        p = handler.InputPayload(**kwargs)
        assert p.output_format == "jpg"  # the default


def test_input_payload_rejects_unsupported_format():
    """Literal["png", "jpg"] should reject "tiff", "webp" etc."""
    with pytest.raises(ValidationError):
        handler.InputPayload(image_url="x", output_format="webp")


# ────────────────────────────────────────────────────────────────────
# _fetch_image_bytes — multi-mode input resolution
# ────────────────────────────────────────────────────────────────────

def _png_bytes() -> bytes:
    """A trivial valid PNG. Different from conftest.make_png because
    these tests don't care about content — only routing — and want a
    minimal payload."""
    img = Image.new("RGB", (8, 8), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_fetch_image_bytes_base64_raw():
    raw = _png_bytes()
    payload = handler.InputPayload(
        image_base64=base64.b64encode(raw).decode("ascii"),
    )
    got, label = handler._fetch_image_bytes(payload, job_id="t1")
    assert got == raw
    assert label == "image_base64"


def test_fetch_image_bytes_base64_with_data_url_prefix():
    """Data-URL prefix (`data:image/png;base64,...`) is what the iosuite
    frontend sends; the handler must strip it. A bug here = base64
    decode crashes on every web upload."""
    raw = _png_bytes()
    encoded = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    payload = handler.InputPayload(image_base64=encoded)
    got, _ = handler._fetch_image_bytes(payload, job_id="t1")
    assert got == raw


def test_fetch_image_bytes_image_path_absolute(tmp_path):
    raw = _png_bytes()
    p = tmp_path / "in.png"
    p.write_bytes(raw)
    payload = handler.InputPayload(image_path=str(p))
    got, label = handler._fetch_image_bytes(payload, job_id="t1")
    assert got == raw
    assert label == "in.png"


def test_fetch_image_bytes_image_path_relative_resolves_to_workspace(monkeypatch, tmp_path):
    """Relative paths must resolve to WORKSPACE, not CWD. RunPod workers
    have an unpredictable CWD; relying on it would make jobs flaky."""
    monkeypatch.setattr(handler, "WORKSPACE", tmp_path)
    raw = _png_bytes()
    (tmp_path / "in.png").write_bytes(raw)
    payload = handler.InputPayload(image_path="in.png")  # relative
    got, _ = handler._fetch_image_bytes(payload, job_id="t1")
    assert got == raw


def test_fetch_image_bytes_image_path_missing_raises(tmp_path):
    payload = handler.InputPayload(image_path=str(tmp_path / "nope.png"))
    with pytest.raises(FileNotFoundError):
        handler._fetch_image_bytes(payload, job_id="t1")


def test_fetch_image_bytes_image_url(monkeypatch):
    raw = _png_bytes()
    fake_response = mock.Mock()
    fake_response.content = raw
    fake_response.raise_for_status = lambda: None
    monkeypatch.setattr(handler.requests, "get", lambda url, timeout: fake_response)
    payload = handler.InputPayload(image_url="https://example.test/x.png")
    got, label = handler._fetch_image_bytes(payload, job_id="t1")
    assert got == raw
    assert label == "https://example.test/x.png"


# ────────────────────────────────────────────────────────────────────
# _gpu_sm_arch — nvidia-smi parsing
# ────────────────────────────────────────────────────────────────────

def test_gpu_sm_arch_parses_compute_cap(monkeypatch):
    """nvidia-smi outputs "8.9\n" (sometimes "8.9, 8.9" if multi-GPU);
    we want "sm89". The dot has to be stripped; if it isn't, the
    fetch-model lookup hits "sm8.9" which never matches the manifest
    and the worker boots into a bogus FileNotFoundError."""
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: "8.9\n")
    assert handler._gpu_sm_arch() == "sm89"


def test_gpu_sm_arch_handles_multi_gpu_output(monkeypatch):
    """Multi-GPU hosts return one line per GPU. We take the first."""
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: "9.0\n9.0\n")
    assert handler._gpu_sm_arch() == "sm90"


def test_gpu_sm_arch_missing_nvidia_smi_returns_none(monkeypatch):
    """CPU image / dev machine: nvidia-smi binary not present. Must
    return None, not raise — the caller branches on this to decide
    whether a TRT-mode boot is even possible."""
    def _missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "check_output", _missing)
    assert handler._gpu_sm_arch() is None


def test_gpu_sm_arch_subprocess_error_returns_none(monkeypatch):
    """Driver glitch / nvidia-smi exit !=0. Same shape as missing-
    binary case — None, not raise."""
    def _err(*a, **k):
        raise subprocess.CalledProcessError(1, ["nvidia-smi"])
    monkeypatch.setattr(subprocess, "check_output", _err)
    assert handler._gpu_sm_arch() is None


# ────────────────────────────────────────────────────────────────────
# _resolve_model_path — strict-mode branches
# ────────────────────────────────────────────────────────────────────

def test_resolve_model_path_uses_baked_env_var_first(monkeypatch, tmp_path):
    """When REAL_ESRGAN_MODEL points at an existing file, no fetch
    runs. This is the cpu/cuda image fast-path: zero-RTT boot."""
    baked = tmp_path / "baked.onnx"
    baked.write_bytes(b"x")
    monkeypatch.setenv("REAL_ESRGAN_MODEL", str(baked))
    # Subprocess must NOT be invoked — assert via a poison spy
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: pytest.fail("fetch-model should not run"))
    got = handler._resolve_model_path()
    assert got == baked


def test_resolve_model_path_trt_strict_no_sm_arch_raises(monkeypatch):
    """PROVIDER=trt + no GPU detected = hard fail. Falling back to
    .onnx would either crash later (TRT image has no ORT) or pay a
    JIT compile cost the design rules out."""
    monkeypatch.delenv("REAL_ESRGAN_MODEL", raising=False)
    monkeypatch.setattr(handler, "PROVIDER", "trt")
    monkeypatch.setattr(handler, "_gpu_sm_arch", lambda: None)
    with pytest.raises(RuntimeError, match="PROVIDER=trt requires a CUDA GPU"):
        handler._resolve_model_path()


def test_resolve_model_path_trt_fetch_failure_raises(monkeypatch, tmp_path):
    """PROVIDER=trt + fetch-model returns non-zero exit (e.g. no engine
    matches this sm-arch in the manifest). Must surface the subprocess
    stderr in the raised error so the boot-error diagnostic in the job
    response carries an actionable message."""
    monkeypatch.delenv("REAL_ESRGAN_MODEL", raising=False)
    monkeypatch.setattr(handler, "PROVIDER", "trt")
    monkeypatch.setattr(handler, "_gpu_sm_arch", lambda: "sm70")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    fake = mock.Mock(returncode=1, stderr="no engine for sm70", stdout="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="fetch-model exited 1.*no engine for sm70"):
        handler._resolve_model_path()


def test_resolve_model_path_trt_fetch_succeeds_no_engine_raises(monkeypatch, tmp_path):
    """fetch-model returns 0 but no .engine landed at the cache dir
    (manifest entry exists but downloaded asset is named differently).
    This is a packaging-bug condition; must raise, not silently use
    nothing."""
    monkeypatch.delenv("REAL_ESRGAN_MODEL", raising=False)
    monkeypatch.setattr(handler, "PROVIDER", "trt")
    monkeypatch.setattr(handler, "_gpu_sm_arch", lambda: "sm89")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    fake = mock.Mock(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="no matching .engine"):
        handler._resolve_model_path()


def test_resolve_model_path_trt_returns_engine_on_success(monkeypatch, tmp_path):
    """Happy path: subprocess "succeeds" and we plant a matching
    .engine file at the expected cache path. Verifies the glob
    pattern (`*<sm>*.engine`) and the chosen sort order."""
    monkeypatch.delenv("REAL_ESRGAN_MODEL", raising=False)
    monkeypatch.setattr(handler, "PROVIDER", "trt")
    monkeypatch.setattr(handler, "_gpu_sm_arch", lambda: "sm89")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = tmp_path / "real-esrgan-serve" / "models"

    def _fake_run(cmd, **k):
        # Simulate fetch-model writing the engine file
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "x4plus-l40s-sm89-trt10.1_fp16.engine").write_bytes(b"engine")
        return mock.Mock(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    got = handler._resolve_model_path()
    assert got.name == "x4plus-l40s-sm89-trt10.1_fp16.engine"
    assert got.parent == cache_dir


def test_resolve_model_path_cuda_returns_onnx(monkeypatch, tmp_path):
    monkeypatch.delenv("REAL_ESRGAN_MODEL", raising=False)
    monkeypatch.setattr(handler, "PROVIDER", "cuda")
    monkeypatch.setattr(handler, "MODEL_NAME", "realesrgan-x4plus")
    monkeypatch.setattr(handler, "MODEL_VARIANT", "fp16")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = tmp_path / "real-esrgan-serve" / "models"

    def _fake_run(cmd, **k):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "realesrgan-x4plus_fp16.onnx").write_bytes(b"onnx")
        return mock.Mock(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    got = handler._resolve_model_path()
    assert got.name == "realesrgan-x4plus_fp16.onnx"
