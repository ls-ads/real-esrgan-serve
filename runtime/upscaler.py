#!/usr/bin/env python3
"""Real-ESRGAN inference helper — invoked as a subprocess by real-esrgan-serve.

This is the only place ONNX model loading lives. The Go CLI
subprocesses here for both:

  * `real-esrgan-serve upscale ...` — one-shot mode (this script
    runs once per image, exits)
  * `real-esrgan-serve serve` — daemon mode (the Go server spawns
    this once and feeds JSONL frames over stdin; see --serve flag)

Why Python and not Go: onnxruntime's Python bindings expose CUDA
EP + CPU fallback through a stable interface. The Go-CGO ONNX
binding option would re-tie the release to specific CUDA ABIs,
which is exactly the distribution friction the rebuild removes.

I/O contract:

  Args (one-shot mode):
    --input PATH       single image file
    --output PATH      where to write upscaled image (parent created)
    --model PATH       path to .onnx (resolved by Go side)
    --gpu-id INT       CUDA device index; -1 = CPU
    --json-events      emit progress as one JSON object per line on stdout

  Stdin (serve mode):
    one JSON object per line, e.g.
    {"id": "abc", "input": "/tmp/in.jpg", "output": "/tmp/out.jpg"}

  Stdout (json-events / serve mode):
    {"event": "ready"}                                 once after model load
    {"event": "progress", "id": "abc", "frac": 0.42}   periodic, 0.0..1.0
    {"event": "done", "id": "abc", "output": "..."}    on success
    {"event": "error", "id": "abc", "msg": "..."}      on failure (does NOT exit serve mode)

  Exit codes (one-shot mode):
    0  success
    1  user error (bad args / bad input image)
    2  runtime error (model load / inference failed)
    3  environment error (onnxruntime missing, CUDA EP unavailable when requested)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# We import lazily so `python3 upscaler.py --help` works on systems
# without onnxruntime installed (e.g. running --help from CI before
# the package is provisioned).


def _emit(json_events: bool, **payload) -> None:
    """Write one JSON event to stdout if json-events is on, flush."""
    if not json_events:
        return
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _die(code: int, msg: str, json_events: bool) -> None:
    """Print a clear human error, optionally also a JSON event, exit."""
    if json_events:
        _emit(True, event="error", msg=msg)
    else:
        print(f"upscaler: {msg}", file=sys.stderr)
    sys.exit(code)


_PROVIDER_CHOICES = ("auto", "cpu", "cuda", "trt")


def _build_providers(provider: str, gpu_id: int, available: list[str],
                     trt_cache: Path) -> list:
    """Translate a user-facing provider name (cpu|cuda|trt|auto) into
    an ORT providers list. `auto` walks trt → cuda → cpu, picking the
    first that's compiled into the wheel. Concrete picks are still
    subject to runtime EP-init success (see _load_session)."""
    cuda_opts = {"device_id": gpu_id}
    trt_opts = {
        "device_id": gpu_id,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(trt_cache),
    }
    if provider == "cpu" or gpu_id < 0:
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "provider=cuda requested but CUDAExecutionProvider not in "
                f"this ORT wheel. Available: {available}"
            )
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    if provider == "trt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "provider=trt requested but TensorrtExecutionProvider not "
                f"in this ORT wheel. Available: {available}"
            )
        # TRT first; CUDA EP next as a fallback for ops TRT doesn't
        # support; CPU last. ORT will JIT-compile a TRT engine on
        # first inference and cache under trt_engine_cache_path.
        return [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]
    # auto
    if "TensorrtExecutionProvider" in available:
        return [
            ("TensorrtExecutionProvider", trt_opts),
            ("CUDAExecutionProvider", cuda_opts),
            "CPUExecutionProvider",
        ]
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class TrtSession:
    """Direct TensorRT execution path. Loads a .engine produced by
    `build/compile_engine.py`, allocates GPU I/O via cuda-python, and
    executes via execute_async_v3.

    No ONNX Runtime in the request path. The TRT image deliberately
    omits onnxruntime-gpu (saves ~600 MB on the cold-start image
    pull); engines are pre-built per (gpu-class, sm-arch, trt-version)
    and fetched at worker boot, so first-request inference cost is
    pure execute, not JIT compile.

    Dynamic shape: the engine ships an optimisation profile spanning
    min=(1,3,64,64), opt=(1,3,720,720), max=(1,3,1280,1280) — see
    compile_engine.py. Per-request we set the actual input shape via
    set_input_shape, ask the engine for the resulting output shape,
    allocate a fresh output buffer (input buffer is re-allocated when
    the request is bigger than what we already have), and execute.
    """

    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-not-found]
            from cuda import cudart  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"TRT-direct path requires tensorrt + cuda-python: {e}"
            ) from e
        import numpy as np  # noqa: F401  imported here so np is loaded once

        self._trt = trt
        self._cudart = cudart

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with engine_path.open("rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self._engine = engine
        self._context = engine.create_execution_context()

        # Engine may carry one or two optimisation profiles (compile_engine.py
        # builds two: profile 0 = single-image / full size range, profile
        # 1 = batched / smaller size range). On each `run` we pick the
        # right profile based on the input shape and switch contexts via
        # set_optimization_profile_async if needed.
        self._num_profiles = engine.num_optimization_profiles
        self._current_profile = -1  # forces explicit set on first run

        # Resolve input/output tensor names. TRT 10 surfaces these via
        # get_tensor_name + get_tensor_mode rather than the deprecated
        # binding-index API.
        self._input_name = None
        self._output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if self._input_name is None or self._output_name is None:
            raise RuntimeError(
                f"engine missing input or output tensor "
                f"(input={self._input_name}, output={self._output_name})"
            )

        # Cache the I/O dtypes — TRT exposes them as trt.DataType which
        # we convert to numpy dtypes once so per-request hot path is just
        # numpy ops + memcpys.
        self._input_dtype = trt.nptype(engine.get_tensor_dtype(self._input_name))
        self._output_dtype = trt.nptype(engine.get_tensor_dtype(self._output_name))

        err, self._stream = cudart.cudaStreamCreate()
        _check_cudart(cudart, err, "cudaStreamCreate")

        # Per-shape device buffers; lazily (re)allocated when the
        # requested capacity grows. We never shrink — Real-ESRGAN
        # workloads on a given worker tend to cluster around a few input
        # sizes, so the high-water mark is what matters.
        self._d_input = None
        self._d_output = None
        self._d_input_capacity = 0
        self._d_output_capacity = 0

    def run(self, chw):  # numpy NCHW float32 → numpy NCHW (engine output dtype)
        """Execute one forward pass. Input is host-side NCHW float32
        (what _preprocess emits); we cast to the engine's input dtype,
        copy HtoD, run, copy DtoH, and return the host output array."""
        import numpy as np  # type: ignore[import-not-found]
        trt = self._trt
        cudart = self._cudart

        if chw.dtype != self._input_dtype:
            chw = chw.astype(self._input_dtype)
        chw = np.ascontiguousarray(chw)

        n, c, h, w = chw.shape
        # Pick the profile that this shape fits in. Profile 0 is the
        # single-image profile (full size range, batch=1 only); profile
        # 1 is the batched profile (smaller size range, batch up to
        # max_batch). Single-engine builds (legacy) only have profile 0.
        target_profile = self._select_profile(n, h, w)
        if target_profile != self._current_profile:
            self._context.set_optimization_profile_async(target_profile, self._stream)
            cudart.cudaStreamSynchronize(self._stream)  # profile switch is stream-bound
            self._current_profile = target_profile

        if not self._context.set_input_shape(self._input_name, (n, c, h, w)):
            raise RuntimeError(
                f"set_input_shape({n},{c},{h},{w}) rejected — outside the "
                f"engine's optimisation profile {target_profile} (min/opt/max). "
                f"Re-build the engine with a profile that covers this "
                f"resolution, or split the request so each image fits."
            )
        out_shape = tuple(self._context.get_tensor_shape(self._output_name))

        in_nbytes = int(chw.nbytes)
        out_nbytes = int(np.prod(out_shape)) * np.dtype(self._output_dtype).itemsize
        self._ensure_buffers(in_nbytes, out_nbytes)

        # Bind device addresses for this execution. set_tensor_address
        # is per-context state; safe to re-set every call.
        self._context.set_tensor_address(self._input_name, int(self._d_input))
        self._context.set_tensor_address(self._output_name, int(self._d_output))

        err, = cudart.cudaMemcpyAsync(
            int(self._d_input), chw.ctypes.data, in_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream,
        )
        _check_cudart(cudart, err, "cudaMemcpyAsync HtoD")

        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("execute_async_v3 returned False")

        host_out = np.empty(out_shape, dtype=self._output_dtype)
        err, = cudart.cudaMemcpyAsync(
            host_out.ctypes.data, int(self._d_output), out_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream,
        )
        _check_cudart(cudart, err, "cudaMemcpyAsync DtoH")
        err, = cudart.cudaStreamSynchronize(self._stream)
        _check_cudart(cudart, err, "cudaStreamSynchronize")
        return host_out

    def _select_profile(self, n: int, h: int, w: int) -> int:
        """Choose the optimisation profile that admits (n, h, w).

        Single-engine profile builds (legacy / one-profile) → 0.

        Two-profile engines (the standard build):
          n == 1 → profile 0. Profile 0 covers the full size range
            (1×64x64 to 1×1280x1280) and is tuned for opt=720×720,
            so it picks better tactics for single-image requests at
            arbitrary sizes than profile 1 (which has opt at the
            batched corner 4×512×512 — measurably slower at 720×720
            batch=1; ~20 % regression observed during validation).
          n > 1 → profile 1 if (n, h, w) fits its (max_n, max_h,
            max_w); else FAIL — the request shape is unsupported.

        Mixed-shape batches must be split upstream — handler.py's
        _process_batch does that. Reaching here with n>1 + an
        oversize shape is a caller bug."""
        if self._num_profiles < 2:
            return 0
        if n == 1:
            return 0
        prof1_max = self._engine.get_tensor_profile_shape(self._input_name, 1)[2]
        max_n, _, max_h, max_w = prof1_max
        if n > max_n or h > max_h or w > max_w:
            raise RuntimeError(
                f"batched request (n={n}, h={h}, w={w}) exceeds profile 1 max "
                f"(n={max_n}, h={max_h}, w={max_w}). Split the request into "
                f"smaller groups upstream."
            )
        return 1

    def _ensure_buffers(self, in_nbytes: int, out_nbytes: int) -> None:
        cudart = self._cudart
        if in_nbytes > self._d_input_capacity:
            if self._d_input is not None:
                cudart.cudaFree(self._d_input)
            err, self._d_input = cudart.cudaMalloc(in_nbytes)
            _check_cudart(cudart, err, "cudaMalloc input")
            self._d_input_capacity = in_nbytes
        if out_nbytes > self._d_output_capacity:
            if self._d_output is not None:
                cudart.cudaFree(self._d_output)
            err, self._d_output = cudart.cudaMalloc(out_nbytes)
            _check_cudart(cudart, err, "cudaMalloc output")
            self._d_output_capacity = out_nbytes

    def get_providers(self) -> list[str]:
        """Diagnostics surface — handler.py reads this and embeds it in
        every job response so callers can verify the worker really ran
        TRT-direct (not silently degraded). Mirrors ort.InferenceSession's
        get_providers() so callers don't branch."""
        return ["TensorrtDirect"]


def _check_cudart(cudart, err, where: str) -> None:
    """cuda-python returns (err, *result) tuples; non-zero err is fatal.
    We don't try to recover — a CUDA failure mid-request means the
    context is suspect; better to crash the helper and let the handler
    surface a clean boot_failed than to muddle on with corrupt state."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{where} failed: {err}")


def _load_session(model_path: Path, gpu_id: int, json_events: bool,
                  provider: str = "auto"):
    """Load the inference session for the chosen provider. `provider`
    is one of cpu | cuda | trt | auto.

    cpu/cuda go through ONNX Runtime (model_path is a .onnx).
    trt loads the engine directly via the TensorRT Python API
    (model_path is a .engine). The TRT image doesn't ship ORT, so the
    trt branch never imports onnxruntime — keeps the import lazy so the
    image stays clean.

    Strict mode: if the user asks for a specific GPU provider and ORT
    silently drops it at session init, this raises RuntimeError instead
    of degrading to CPU silently."""
    if not model_path.exists():
        _die(1, f"model file does not exist: {model_path}", json_events)

    if provider == "trt":
        # TRT-direct path. No ORT involved — the trt image deliberately
        # omits onnxruntime-gpu (~600 MB) and uses tensorrt + cuda-python
        # directly. model_path is a .engine pre-built for this host's GPU.
        if model_path.suffix != ".engine":
            raise RuntimeError(
                f"provider=trt requires a .engine file, got {model_path.name}. "
                f"The TRT image fetches the matching engine for this GPU's SM "
                f"arch at boot; check handler._resolve_model_path."
            )
        sess = TrtSession(model_path)
        print("[trt] direct execution path (no ORT); engine loaded",
              file=sys.stderr, flush=True)
        return sess

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        _die(3, f"onnxruntime not installed: {e}", json_events)

    available = ort.get_available_providers()
    trt_cache = Path("/tmp/real-esrgan-serve/trt-cache")
    trt_cache.mkdir(parents=True, exist_ok=True)
    providers = _build_providers(provider, gpu_id, available, trt_cache)

    session_options = ort.SessionOptions()
    # Reduce log noise; the Go side wants clean stdout for JSON events
    session_options.log_severity_level = 3

    sess = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )

    # Strict-mode verification: ORT silently drops EPs that fail to
    # initialize at session creation. If the user asked for a specific
    # GPU EP and it isn't in the active list, that's almost always a
    # missing system library (libcudnn / libnvinfer) — fail loudly so
    # the operator gets a real error in the response payload instead
    # of mysteriously slow CPU inference.
    requested = [p[0] if isinstance(p, tuple) else p for p in providers]
    actual = sess.get_providers()
    print(f"[ort] requested EPs: {requested}", file=sys.stderr, flush=True)
    print(f"[ort] active EPs:    {actual}", file=sys.stderr, flush=True)
    must_have = {
        "cuda": "CUDAExecutionProvider",
    }.get(provider)
    if must_have and must_have not in actual:
        raise RuntimeError(
            f"provider={provider} requested but {must_have} did not "
            f"initialize. Active EPs: {actual}. Likely cause: missing "
            f"system library (libcudnn / libcublas / libcudart). "
            f"Provider=auto will degrade silently; provider=cuda is "
            f"strict by design."
        )
    return sess


def _preprocess(image_path: Path):
    """Decode image → CHW float32 [0..1] tensor + (W, H)."""
    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:
        _die(3, f"missing dependency for image I/O: {e}", json_events=False)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # HWC -> CHW -> NCHW
    chw = arr.transpose(2, 0, 1)[None, :, :, :]
    return chw, w, h


def _run_inference(session, chw):
    """Run inference, casting input to match the model's input dtype.
    Returns a list whose [0] is the NCHW output array (matches ORT's
    session.run shape so _postprocess_and_save is path-agnostic).

    TrtSession casts internally and handles its own buffer mgmt. ORT
    needs us to coerce up front because the exported ONNX may be FP16
    (smaller, fast on tensor cores) or FP32 and _preprocess always
    emits FP32. Output is left in whatever dtype the model produced —
    _postprocess_and_save handles the uint8 cast at the end and
    clip/scale work correctly for either float dtype."""
    if isinstance(session, TrtSession):
        return [session.run(chw)]
    import numpy as np  # type: ignore[import-not-found]
    inp_meta = session.get_inputs()[0]
    if "float16" in inp_meta.type and chw.dtype != np.float16:
        chw = chw.astype(np.float16)
    return session.run(None, {inp_meta.name: chw})


def _postprocess_and_save(out_tensor, output_path: Path) -> None:
    """NCHW float [0..1] (float16 or float32) → PIL image → file."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    arr = (out_tensor[0].astype(np.float32).clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    # CHW -> HWC
    arr = arr.transpose(1, 2, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    # PIL infers format from suffix; .jpg encodes ~5x faster than .png
    # on a 5K-class output — see ARCHITECTURE.md performance notes.
    img.save(output_path)


def run_one_shot(args: argparse.Namespace) -> int:
    je = args.json_events
    model = Path(args.model)
    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        _die(1, f"input does not exist: {inp}", je)

    _emit(je, event="loading_model", path=str(model))
    t0 = time.monotonic()
    session = _load_session(model, args.gpu_id, je, provider=args.provider)
    _emit(je, event="model_loaded", elapsed_ms=int((time.monotonic() - t0) * 1000))

    _emit(je, event="preprocessing", input=str(inp))
    chw, w, h = _preprocess(inp)

    _emit(je, event="inferring", width=w, height=h)
    t0 = time.monotonic()
    try:
        result = _run_inference(session, chw)
    except Exception as e:  # noqa: BLE001
        _die(2, f"inference failed: {e}", je)
    _emit(je, event="inferred", elapsed_ms=int((time.monotonic() - t0) * 1000))

    _emit(je, event="postprocessing", output=str(out))
    _postprocess_and_save(result[0], out)

    _emit(je, event="done", output=str(out))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Daemon mode: load model once, then process JSONL jobs from stdin
    until EOF. Errors on a single job emit an event but do not exit.

    Two frame shapes accepted:

      Single image (legacy + ergonomic):
        {"id": "...", "input": "/path/to/in.jpg", "output": "/path/to/out.jpg"}
        → emits {"event": "done", "id": "...", "output": "/path/to/out.jpg"}

      Batched (same shape across all items):
        {"id": "...",
         "inputs":  ["in1.jpg", "in2.jpg", ...],
         "outputs": ["out1.jpg", "out2.jpg", ...]}
        → emits {"event": "done", "id": "...",
                 "results": [{"output": "out1.jpg"}, ...],
                 "batched": true}

    The batched path stacks the per-image NCHW tensors along axis 0
    (so all images must have the same H,W — the caller is expected
    to group by shape before sending), runs ONE forward pass, then
    slices the output back into per-image entries. That collapses
    N forward passes (with their N × set_input_shape + N × launch
    overhead) into one — the single biggest perf opportunity the
    sweep flagged.
    """
    import numpy as np  # type: ignore[import-not-found]

    model = Path(args.model)
    session = _load_session(model, args.gpu_id, json_events=True,
                            provider=args.provider)
    # Include active EPs and the requested provider in the ready
    # event so the handler can surface both in job responses —
    # RunPod's worker logs aren't reachable via API, so we route
    # diagnostics through the response payload instead of stderr.
    _emit(True, event="ready",
          providers=session.get_providers(),
          requested_provider=args.provider,
          model=model.name)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            _emit(True, event="error", msg=f"bad jsonl frame: {e}")
            continue

        job_id = job.get("id", "")

        # Branch on shape: `inputs` (plural) → batched; `input` → single.
        if "inputs" in job and "outputs" in job:
            try:
                _serve_one_batch(session, job, job_id, np)
            except Exception as e:  # noqa: BLE001
                _emit(True, event="error", id=job_id, msg=str(e))
            continue

        try:
            chw, w, h = _preprocess(Path(job["input"]))
            result = _run_inference(session, chw)
            _postprocess_and_save(result[0], Path(job["output"]))
            _emit(True, event="done", id=job_id, output=job["output"])
        except Exception as e:  # noqa: BLE001
            _emit(True, event="error", id=job_id, msg=str(e))
    return 0


def _serve_one_batch(session, job: dict, job_id: str, np) -> None:
    """Batched JSONL handler. All inputs must already share the same
    (H, W) — the handler-side router (handler.py:_process_one_image)
    is responsible for grouping. We re-validate here as a safety net
    and raise loudly if violated, rather than silently truncating."""
    inputs = [Path(p) for p in job["inputs"]]
    outputs = [Path(p) for p in job["outputs"]]
    if len(inputs) != len(outputs):
        raise ValueError(f"inputs/outputs length mismatch: "
                         f"{len(inputs)} vs {len(outputs)}")
    if not inputs:
        raise ValueError("empty batch")

    # Preprocess each, validate same shape, stack into NCHW.
    chws = []
    ref_shape = None
    for p in inputs:
        chw, _, _ = _preprocess(p)  # shape (1, 3, h, w)
        if ref_shape is None:
            ref_shape = chw.shape
        elif chw.shape != ref_shape:
            raise ValueError(
                f"batch contains heterogeneous shapes — first {ref_shape}, "
                f"then {chw.shape} at {p.name}. Group by shape upstream."
            )
        chws.append(chw)
    # Stack along the existing batch dim; each chw is already (1,3,H,W).
    batched = np.concatenate(chws, axis=0)

    # _run_inference returns a list of arrays — for both ORT and
    # TrtSession the first entry is the NCHW output. _postprocess
    # expects a 4-D NCHW tensor; we pass slices so each item lands
    # at its own output path.
    result = _run_inference(session, batched)
    out_tensor = result[0]  # shape (N, 3, 4H, 4W)

    per_item: list[dict] = []
    for i, out_path in enumerate(outputs):
        # _postprocess_and_save consumes `out_tensor[0]` (drops batch
        # dim). Pass a 1-element slice so the existing helper stays
        # path-agnostic between single + batched callers.
        single = out_tensor[i:i + 1]
        _postprocess_and_save(single, out_path)
        per_item.append({"output": str(out_path)})

    _emit(True, event="done", id=job_id, batched=True,
          results=per_item, batch_size=len(inputs))


def main() -> int:
    p = argparse.ArgumentParser(prog="upscaler.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--input", help="input image (one-shot mode)")
    p.add_argument("--output", help="output image (one-shot mode)")
    p.add_argument("--model", required=True, help="path to .onnx model file")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA device id; -1 = CPU")
    p.add_argument("--provider", choices=_PROVIDER_CHOICES, default="auto",
                   help="execution provider: cpu | cuda | trt | auto. "
                        "auto walks trt → cuda → cpu picking the first the "
                        "ORT wheel was built with. cpu/cuda/trt are STRICT "
                        "— a missing system lib raises rather than falling "
                        "back silently.")
    p.add_argument("--json-events", action="store_true", help="emit progress as JSONL on stdout")
    p.add_argument("--serve", action="store_true", help="daemon mode: read JSONL jobs from stdin")
    args = p.parse_args()

    if args.serve:
        return run_serve(args)

    if not args.input or not args.output:
        _die(1, "--input and --output are required in one-shot mode", args.json_events)
    return run_one_shot(args)


if __name__ == "__main__":
    sys.exit(main())
