"""RunPod serverless handler for real-esrgan-serve.

Folded in from the previous runpod-real-esrgan repo. The previous
version spawned a Go HTTP server (with a CGO TensorRT bridge) and
POSTed jobs to it. The new version cuts the middle layer:

  * On container start: warm up `runtime/upscaler.py --serve` once
    (eager onnxruntime session load via CUDA EP).
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
# Tunables (env-driven so RunPod's endpoint config can flip them)
# ───────────────────────────────────────────────────────────────────────
MAX_INPUT_DIM = 1280
RUNTIME_HELPER = Path(
    os.environ.get("REAL_ESRGAN_RUNTIME", "/usr/share/real-esrgan-serve/runtime/upscaler.py")
)
PYTHON_BIN = os.environ.get("PYTHON", sys.executable or "python3")
WORKSPACE = Path("/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Execution provider chosen by the operator. Maps directly to
# upscaler.py's --provider flag.
#   cpu   — CPUExecutionProvider only
#   cuda  — CUDA EP (strict; fails if libcudnn missing)
#   trt   — TensorRT EP, JIT-compiles engine on first request,
#           caches under /tmp/real-esrgan-serve/trt-cache
#   auto  — trt → cuda → cpu, picking the first ORT successfully
#           initializes (silent fallback)
PROVIDER = os.environ.get("REAL_ESRGAN_PROVIDER", "auto").lower()

# Variant selection for fetch-model. cpu/cuda/auto pull the .onnx;
# trt pulls a pre-compiled .engine if one exists for this GPU's SM
# arch. Falls back to .onnx if no engine for the host's GPU.
MODEL_NAME = os.environ.get("REAL_ESRGAN_MODEL_NAME", "realesrgan-x4plus")
MODEL_VARIANT = os.environ.get("REAL_ESRGAN_MODEL_VARIANT", "fp16")


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
# Model resolution: fetch the right artefact at startup
# ───────────────────────────────────────────────────────────────────────
def _gpu_sm_arch() -> Optional[str]:
    """Read the host GPU's compute capability via nvidia-smi (e.g. "sm89").
    None on hosts without a GPU or nvidia-smi (we still ship a binary
    that runs on CPU)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip().split("\n")[0]
        if out:
            return "sm" + out.replace(".", "")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _resolve_model_path() -> Path:
    """Resolve the model artefact for this worker. Strategy:

       1. $REAL_ESRGAN_MODEL — operator override or image-baked path.
          The CPU and CUDA images bake the .onnx in and set this so the
          handler skips the fetch entirely. Also lets operators point
          at a custom model file.
       2. PROVIDER=trt — fetch the .engine matching this host's SM arch.
          STRICT: no .onnx fallback. The TRT image has no ORT to load a
          .onnx, and JIT-compiling at boot would add ~30-60 s to cold
          start (the metric we're optimising). A worker on an
          unsupported GPU class fails loudly rather than degrading;
          that's a maintenance signal to add the GPU to the build matrix.
       3. PROVIDER=cpu/cuda — fetch the .onnx for the configured variant.
          Used by non-Docker installs and as a fallback if the baked-in
          path was somehow not set.

    Cached under $XDG_CACHE_HOME/real-esrgan-serve/models so a worker
    that's paused/resumed reuses the file without re-downloading."""
    explicit = os.environ.get("REAL_ESRGAN_MODEL")
    if explicit and Path(explicit).exists():
        log.info(f"using REAL_ESRGAN_MODEL={explicit}")
        return Path(explicit)

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/var/cache")) / "real-esrgan-serve" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(*args: str) -> None:
        """Run fetch-model and surface stderr on failure. The default
        check_call swallows stderr, hiding the real error inside the
        boot diagnostic — that cost us an iteration cycle, so we
        capture and re-raise with the actual message."""
        cmd = ["real-esrgan-serve", "fetch-model", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"fetch-model exited {proc.returncode}: "
                f"stderr={proc.stderr.strip()!r} stdout={proc.stdout.strip()!r}"
            )

    if PROVIDER == "trt":
        sm = _gpu_sm_arch()
        if not sm:
            raise RuntimeError(
                "PROVIDER=trt requires a CUDA GPU (nvidia-smi must report a "
                "compute_cap), got nothing. The TRT image must run on GPU."
            )
        log.info(f"PROVIDER=trt + sm={sm}: fetching matching engine")
        _fetch("--name", MODEL_NAME, "--variant", "engine",
               "--sm-arch", sm, "--dest", str(cache_dir))
        engines = sorted(cache_dir.glob(f"*{sm}*.engine"))
        if not engines:
            raise RuntimeError(
                f"fetch-model succeeded for sm={sm} but no matching .engine "
                f"under {cache_dir}. Check MANIFEST.json and the release asset "
                f"naming."
            )
        log.info(f"using pre-built engine: {engines[0].name}")
        return engines[0]

    log.info(f"fetching .onnx variant={MODEL_VARIANT}")
    _fetch("--name", MODEL_NAME, "--variant", MODEL_VARIANT,
           "--dest", str(cache_dir))
    onnx = cache_dir / f"{MODEL_NAME}_{MODEL_VARIANT}.onnx"
    if not onnx.exists():
        # Fall back to whatever the manifest produced
        candidates = list(cache_dir.glob(f"{MODEL_NAME}*.onnx"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"fetch-model succeeded but no .onnx in {cache_dir}")
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

    def __init__(self, model: Path, gpu_id: int = 0,
                 provider: str = "auto") -> None:
        log.info(f"Warming helper: model={model.name}, gpu={gpu_id}, "
                 f"provider={provider}")
        self._proc = subprocess.Popen(
            [
                PYTHON_BIN,
                str(RUNTIME_HELPER),
                "--serve",
                "--model", str(model),
                "--gpu-id", str(gpu_id),
                "--provider", provider,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
        self._stdin_lock = threading.Lock()
        self._pending: dict[str, Queue] = {}
        # Ring buffer of stderr lines, drained continuously by the
        # pump thread. We use this on boot failure to surface what
        # the helper emitted to stderr (ORT EP-init error, ImportError,
        # etc.) — RunPod's worker logs aren't fetchable via API so
        # this is the only way to see helper stderr off the worker.
        self._stderr_lines: list[str] = []
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_pump = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_pump.start()

        # Longer timeout because TRT EP JIT-compiles a serialized
        # engine on first session creation (~30 s on RTX 4090). Engine
        # is cached under /tmp/real-esrgan-serve/trt-cache for the
        # rest of this worker's lifetime, but the first compile must
        # finish before we report ready.
        ready = self._await_id("__ready__", timeout=180.0)
        if not ready or ready.get("event") != "ready":
            self._proc.poll()
            # Pump may not have flushed the very last lines yet (no
            # more stdin → reader thread is at EOF read). Wait briefly.
            time.sleep(0.5)
            tail = "\n".join(self._stderr_lines[-100:])
            raise RuntimeError(
                f"helper did not signal ready. exit={self._proc.returncode} "
                f"stderr_tail=\n{tail}"
            )
        # Stash diagnostics from the ready event so handler can
        # include them in every job response. RunPod's worker logs
        # aren't reachable via API; piggy-backing on the response
        # payload is the only way to surface this programmatically.
        self.providers: list[str] = list(ready.get("providers") or [])
        self.requested_provider: str = ready.get("requested_provider") or ""
        self.model_name: str = ready.get("model") or ""
        log.info(f"Helper ready (active EPs: {self.providers})")

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
            stripped = line.rstrip()
            log.info(f"[helper] {stripped}")
            self._stderr_lines.append(stripped)
            # Cap retention so a chatty helper doesn't OOM us
            if len(self._stderr_lines) > 500:
                del self._stderr_lines[:200]

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
#
# CRITICAL: catch every exception. RunPod's worker logs aren't
# fetchable via API (verified — /v2/<id>/logs is console-only), so
# if the WarmHelper init crashes we'd see only a TimeoutError on
# the deploy side with no diagnosis. By stashing the exception in
# _BOOT_ERROR and letting the runpod SDK come up cleanly, the
# handler() function can echo the failure into every job response
# — that's the only programmatic channel back to the operator.
_HELPER: Optional["WarmHelper"] = None
_BOOT_ERROR: Optional[dict] = None
try:
    import traceback
    _model_path = _resolve_model_path()
    _HELPER = WarmHelper(
        model=_model_path,
        gpu_id=int(os.environ.get("GPU_ID", "0")),
        provider=PROVIDER,
    )
except BaseException as _boot_exc:  # noqa: BLE001 — really need everything
    _BOOT_ERROR = {
        "phase": "boot",
        "type": type(_boot_exc).__name__,
        "msg": str(_boot_exc),
        "traceback": traceback.format_exc(),
        "provider": PROVIDER,
        "model_name": MODEL_NAME,
        "model_variant": MODEL_VARIANT,
    }
    log.error(f"BOOT FAILURE — handler will return boot_error on each job: "
              f"{_BOOT_ERROR['type']}: {_BOOT_ERROR['msg']}")


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
    # Boot-error short-circuit: if module-level init failed, every
    # job returns the boot diagnostic immediately. Without this RunPod
    # would just see TimeoutError on the deploy side and we'd have no
    # clue why (worker logs are console-only).
    if _BOOT_ERROR is not None:
        log.error(f"boot_error short-circuit for job {job_id}",
                  request_id=job_id)
        return {
            "error": "boot_failed",
            "_diagnostics": {"boot_error": _BOOT_ERROR},
        }

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

        # Diagnostics piggy-backed on every response. RunPod doesn't
        # expose worker logs via API; this is the only programmatic
        # channel back to the caller. Cheap (a few bytes) and lets the
        # smoke-test verify GPU vs CPU EP without console scraping.
        diagnostics = {
            "providers": _HELPER.providers,
            "requested_provider": _HELPER.requested_provider,
            "model": _HELPER.model_name,
        }

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
                "_diagnostics": diagnostics,
            }

        return {
            "image_base64": base64.b64encode(out_bytes).decode("ascii"),
            "model": "realesrgan-x4plus",
            "input_resolution": f"{in_w}x{in_h}",
            "output_resolution": f"{in_w * 4}x{in_h * 4}",
            "output_format": payload.output_format,
            "_diagnostics": diagnostics,
        }

    except Exception as e:  # noqa: BLE001
        log.error(str(e), request_id=job_id)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
