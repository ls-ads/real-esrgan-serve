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


def _load_session(model_path: Path, gpu_id: int, json_events: bool):
    """Load the ONNX session with the best available provider."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
        _die(3, f"onnxruntime not installed: {e}", json_events)

    if not model_path.exists():
        _die(1, f"model file does not exist: {model_path}", json_events)

    available = ort.get_available_providers()
    providers: list = []
    if gpu_id >= 0 and "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {"device_id": gpu_id},
        ))
    providers.append("CPUExecutionProvider")

    session_options = ort.SessionOptions()
    # Reduce log noise; the Go side wants clean stdout for JSON events
    session_options.log_severity_level = 3

    sess = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )

    # Diagnostic: ORT silently drops EPs that fail to initialize at
    # session creation (e.g. CUDA EP can't load libcudnn / libcublas).
    # `requested` is what we asked for; `actual` is what ORT ended up
    # binding the session to. If actual==[CPUExecutionProvider] but we
    # passed gpu_id>=0, ORT fell back silently and the workload is
    # running on CPU even though the host has a GPU. Log loudly so the
    # operator sees this in worker logs.
    requested = [p[0] if isinstance(p, tuple) else p for p in providers]
    actual = sess.get_providers()
    print(f"[ort] requested EPs: {requested}", file=sys.stderr, flush=True)
    print(f"[ort] active EPs:    {actual}", file=sys.stderr, flush=True)
    if gpu_id >= 0 and "CUDAExecutionProvider" not in actual:
        print(f"[ort] WARNING: CUDA EP requested but not active — "
              f"running on CPU. Likely cause: libcudnn/libcublas/"
              f"libcudart not on the linker path.",
              file=sys.stderr, flush=True)

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
    """Run ORT inference, casting input to match the model's input dtype.
    Exported ONNX may be FP16 (smaller, fast on tensor cores) or FP32.
    _preprocess always emits FP32, so we coerce here once based on what
    the loaded model actually wants. Output is left in whatever dtype
    the model produced — _postprocess_and_save handles the uint8 cast
    at the end and clip/scale work correctly for either float dtype."""
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
    session = _load_session(model, args.gpu_id, je)
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
    until EOF. Errors on a single job emit an event but do not exit."""
    model = Path(args.model)
    session = _load_session(model, args.gpu_id, json_events=True)
    # Include active EPs in the ready event so the handler can surface
    # them in job responses — RunPod's worker logs aren't reachable via
    # API, so we route the diagnostic through the response payload
    # instead of relying on stderr scraping.
    _emit(True, event="ready", providers=session.get_providers())

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
        try:
            chw, w, h = _preprocess(Path(job["input"]))
            result = _run_inference(session, chw)
            _postprocess_and_save(result[0], Path(job["output"]))
            _emit(True, event="done", id=job_id, output=job["output"])
        except Exception as e:  # noqa: BLE001
            _emit(True, event="error", id=job_id, msg=str(e))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="upscaler.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--input", help="input image (one-shot mode)")
    p.add_argument("--output", help="output image (one-shot mode)")
    p.add_argument("--model", required=True, help="path to .onnx model file")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA device id; -1 = CPU")
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
