"""RunPod serverless handler for real-esrgan-serve.

Folded in from the previous runpod-real-esrgan repo and rewritten
against the new architecture. The previous version spawned the old
Go HTTP server (which carried the CGO TensorRT bridge) and POSTed
jobs to it. The new version cuts the middle layer:

  * On container start: warm up `runtime/upscaler.py --serve` once
    (eager onnxruntime session load with TRT EP if available).
  * Per RunPod job: write a JSONL frame to the helper's stdin, read
    one JSONL result frame from stdout.
  * Job input shape unchanged (image_url | image_base64 | image_path)
    so existing iosuite.io callers keep working.

The Go binary is not in the request path here — it ships with the
image so operators can `docker exec` and run `real-esrgan-serve
fetch-model`, `... upscale`, etc., for debugging.

Layout assumptions (set by providers/runpod/Dockerfile):
  /usr/local/bin/real-esrgan-serve
  /usr/share/real-esrgan-serve/runtime/upscaler.py
  /workspace                                       writable scratch
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import Literal, Optional

import requests
import runpod
from PIL import Image
from pydantic import BaseModel, ValidationError, model_validator
from runpod import RunPodLogger

log = RunPodLogger()

# ───────────────────────────────────────────────────────────────────────
# Tunables
# ───────────────────────────────────────────────────────────────────────
MAX_INPUT_DIM = 1280
RUNTIME_HELPER = Path(
    os.environ.get("REAL_ESRGAN_RUNTIME", "/usr/share/real-esrgan-serve/runtime/upscaler.py")
)
PYTHON_BIN = os.environ.get("PYTHON", sys.executable or "python3")
WORKSPACE = Path("/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)


# ───────────────────────────────────────────────────────────────────────
# Input schema (identical surface to the previous handler)
# ───────────────────────────────────────────────────────────────────────
class InputPayload(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    output_path: Optional[str] = None
    output_format: Literal["png", "jpg"] = "jpg"

    @model_validator(mode="after")
    def _need_one_input(self) -> "InputPayload":
        if not (self.image_url or self.image_base64 or self.image_path):
            raise ValueError(
                "Provide one of image_url, image_base64, or image_path."
            )
        return self


# ───────────────────────────────────────────────────────────────────────
# Model resolution: which .onnx do we serve?
# ───────────────────────────────────────────────────────────────────────
def _gpu_arch_tag() -> str:
    """Return e.g. "sm89" for an RTX 4090; "unknown" if nvidia-smi missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip().split("\n")[0]
        return "sm" + out.replace(".", "")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        log.warn(f"nvidia-smi probe failed: {e}")
        return "unknown"


def _trt_ep_loadable() -> bool:
    """ORT's TensorRT EP needs libcudnn.so.8 dlopen-able. Without it,
    loading a `.engine` file via TRT EP fails, ORT silently falls back
    to CUDA EP, and CUDA EP then can't parse the .engine (not ONNX).
    Probing here lets _resolve_model_path skip the engine path on
    images that don't ship cuDNN, avoiding a hard handler crash."""
    import ctypes
    try:
        ctypes.CDLL("libcudnn.so.8")
        return True
    except OSError:
        return False


def _resolve_model_path() -> Path:
    """Pick the model file to serve. Order:
       1. $REAL_ESRGAN_MODEL — explicit override
       2. Per-GPU pre-compiled .engine matching this host's GPU class
          (skips ~30s TRT engine compile on first request)
       3. .onnx fallback baked into the image (TRT JIT-compiles on
          first request, ~30s, then caches under XDG)
       4. fetch-model invocation as a last resort (network-bound,
          assumes a published release exists)
    """
    explicit = os.environ.get("REAL_ESRGAN_MODEL")
    if explicit and Path(explicit).exists():
        log.info(f"using REAL_ESRGAN_MODEL={explicit}")
        return Path(explicit)

    # Look for a pre-compiled .engine for this GPU class. Filename
    # convention is `realesrgan-x4plus-<gpu-class>-<sm>-trt<X.Y>_fp16.engine`
    # — see build/compile_engine.py. We match SM arch (most reliable
    # ID for engine compatibility) over the human GPU class string.
    arch = _gpu_arch_tag()  # e.g. "sm89"
    if arch != "unknown" and _trt_ep_loadable():
        engines = sorted(WORKSPACE.glob(f"*{arch}*.engine"))
        if engines:
            log.info(f"using pre-compiled engine for {arch}: {engines[0].name}")
            return engines[0]
    elif arch != "unknown":
        log.info(f"engine path skipped — libcudnn.so.8 not present, "
                 f"TRT EP would fail to load")

    onnx = WORKSPACE / "realesrgan-x4plus_fp16.onnx"
    if onnx.exists():
        log.info(f"no engine for {arch} — falling back to onnx (first request will JIT-compile a TRT engine)")
        return onnx

    log.warn("model not pre-cached in image — invoking fetch-model (network)")
    subprocess.check_call([
        "real-esrgan-serve", "fetch-model",
        "--name", "realesrgan-x4plus",
        "--variant", "fp16",
        "--dest", str(WORKSPACE),
    ])
    return onnx


# ───────────────────────────────────────────────────────────────────────
# Persistent helper subprocess (warm onnxruntime session)
# ───────────────────────────────────────────────────────────────────────
class WarmHelper:
    """Wraps the Python runtime helper in --serve mode and provides a
    blocking `upscale(input_path, output_path)` call. Replies are
    matched to requests via job_id so concurrent calls don't get
    interleaved (one helper, one ORT session, FIFO over its stdin).
    """

    def __init__(self, model: Path, gpu_id: int = 0) -> None:
        log.info(f"Warming helper: model={model.name}, gpu={gpu_id}")
        self._proc = subprocess.Popen(
            [
                PYTHON_BIN,
                str(RUNTIME_HELPER),
                "--serve",
                "--model", str(model),
                "--gpu-id", str(gpu_id),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
        self._stdin_lock = threading.Lock()
        self._pending: dict[str, Queue] = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_pump = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_pump.start()

        # Wait for the "ready" event before serving the first request
        ready = self._await_id("__ready__", timeout=120.0)
        if not ready or ready.get("event") != "ready":
            raise RuntimeError(f"helper did not signal ready: {ready}")
        log.info("Helper ready")

    def _read_loop(self) -> None:
        for line in self._proc.stdout:  # type: ignore[union-attr]
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.warn(f"helper non-json line: {line.rstrip()}")
                continue

            if msg.get("event") == "ready":
                # Bootstrap signal — route to the synthetic ready waiter
                if "__ready__" in self._pending:
                    self._pending["__ready__"].put(msg)
                continue

            job_id = msg.get("id")
            if job_id and job_id in self._pending:
                self._pending[job_id].put(msg)

    def _stderr_loop(self) -> None:
        for line in self._proc.stderr:  # type: ignore[union-attr]
            log.info(f"[helper] {line.rstrip()}")

    def _await_id(self, job_id: str, timeout: float) -> Optional[dict]:
        q: Queue = Queue()
        self._pending[job_id] = q
        try:
            return q.get(timeout=timeout)
        except Exception:  # noqa: BLE001 — Empty
            return None
        finally:
            self._pending.pop(job_id, None)

    def upscale(self, input_path: Path, output_path: Path, job_id: str, timeout: float = 120.0) -> dict:
        frame = json.dumps({
            "id": job_id,
            "input": str(input_path),
            "output": str(output_path),
        })
        # Subscribe before send to avoid race with a fast helper
        q: Queue = Queue()
        self._pending[job_id] = q
        try:
            with self._stdin_lock:
                self._proc.stdin.write(frame + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            try:
                return q.get(timeout=timeout)
            except Exception:  # noqa: BLE001
                return {"event": "error", "id": job_id, "msg": f"helper timed out after {timeout}s"}
        finally:
            self._pending.pop(job_id, None)


# Bootstrap once per container — warmed up before RunPod starts
# delivering jobs.
_HELPER = WarmHelper(model=_resolve_model_path(), gpu_id=int(os.environ.get("GPU_ID", "0")))


# ───────────────────────────────────────────────────────────────────────
# Per-job input fetching + handler
# ───────────────────────────────────────────────────────────────────────
def _fetch_image_bytes(payload: InputPayload, job_id: Optional[str]) -> tuple[bytes, str]:
    """Resolve `payload` to (img_bytes, source_label) for logging."""
    if payload.image_base64:
        b64 = payload.image_base64
        if ";base64," in b64:
            b64 = b64.split(";base64,", 1)[1]
        return base64.b64decode(b64), "image_base64"

    if payload.image_path:
        local = Path(payload.image_path)
        if not local.is_absolute():
            local = WORKSPACE / local
        if not local.exists():
            raise FileNotFoundError(f"image_path not found: {local}")
        return local.read_bytes(), local.name

    # image_url
    r = requests.get(payload.image_url, timeout=30)  # type: ignore[arg-type]
    r.raise_for_status()
    return r.content, str(payload.image_url)


def handler(job):
    job_id = job.get("id", "")
    try:
        try:
            payload = InputPayload.model_validate(job.get("input", {}))
        except ValidationError as e:
            log.error(f"validation: {e}", request_id=job_id)
            return {"error": str(e)}

        # Pull bytes + check input dimensions before paying GPU cost
        img_bytes, src_label = _fetch_image_bytes(payload, job_id)
        with Image.open(BytesIO(img_bytes)) as img:
            in_w, in_h = img.size
            if in_w > MAX_INPUT_DIM or in_h > MAX_INPUT_DIM:
                raise ValueError(
                    f"input {in_w}x{in_h} exceeds max {MAX_INPUT_DIM}x{MAX_INPUT_DIM}"
                )

        # Stage input + output paths for the helper. Use job_id in the
        # filename so concurrent jobs in the same container don't clobber.
        suffix = ".jpg" if payload.output_format == "jpg" else ".png"
        in_path = WORKSPACE / f"{job_id or 'unknown'}_in{Path(src_label).suffix or '.bin'}"
        out_path = WORKSPACE / f"{job_id or 'unknown'}_out{suffix}"
        in_path.write_bytes(img_bytes)

        result = _HELPER.upscale(in_path, out_path, job_id=job_id or src_label)
        if result.get("event") != "done":
            raise RuntimeError(result.get("msg", "helper returned unexpected event"))

        out_bytes = out_path.read_bytes()
        # Cleanup scratch files; failure here shouldn't sink the job
        try:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except OSError:
            pass

        # Optional: also write to a caller-specified output_path inside
        # the container (useful when network volume is mounted)
        if payload.output_path:
            dest = Path(payload.output_path)
            if not dest.is_absolute():
                dest = WORKSPACE / dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(out_bytes)
            return {
                "output_path": payload.output_path,
                "model": "realesrgan-x4plus",
                "input_resolution": f"{in_w}x{in_h}",
                "output_format": payload.output_format,
            }

        return {
            "image_base64": base64.b64encode(out_bytes).decode("ascii"),
            "model": "realesrgan-x4plus",
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{in_w * 4}x{in_h * 4}",
            "output_format": payload.output_format,
        }

    except Exception as e:  # noqa: BLE001
        log.error(str(e), request_id=job_id)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
