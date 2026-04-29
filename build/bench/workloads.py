"""Workload patterns for the benchmark harness.

A workload is a sequence of `JobSpec` objects that the runner submits
in order. The runner attaches client-side timing and writes one DB
row per spec.

Three patterns implemented:

  - cold_start: a single job on a fresh worker. The endpoint must be
    idle (workersMin=0 + recent idle timeout) so the job actually
    boots a cold worker. Captures first-image latency end-to-end.

  - batch_sweep: a series of jobs with batch sizes 1, 4, 16, 64
    (subject to VRAM ceiling). Each job's images are uniform-sized
    so per-image timing is comparable. Reveals the batching
    amortisation curve — at what batch size do request-overhead
    fixed costs become invisible relative to GPU work.

  - sustained_concurrent: N parallel client threads, each submitting
    jobs back-to-back for `duration_s`. Probes the steady-state
    throughput once the queue scheduler reaches equilibrium with the
    worker pool. Distinct from batch_sweep because the bottleneck is
    different — concurrent reveals the cost of multiple workers
    competing for the same GPU pool, not the within-worker batching
    benefit.

The patterns are independent of GPU class — the same workload runs
on every GPU we test, and the results table tracks (gpu_class,
flavor, workload) tuples.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Iterable

from PIL import Image, ImageDraw


@dataclass
class JobSpec:
    """One unit of work to submit. The runner records it as a single
    row in `jobs` and one row per item in `items`."""
    name: str
    batch_size: int
    image_w: int
    image_h: int
    telemetry: bool = False
    discard_output: bool = True   # benchmarks don't need the bytes back
    cold_start: bool = False      # first job of run, by convention

    def make_payload(self, png_bytes_for_size: Callable[[int, int], bytes]) -> dict:
        import base64
        png = png_bytes_for_size(self.image_w, self.image_h)
        encoded = base64.b64encode(png).decode("ascii")
        return {
            "input": {
                "images": [{"image_base64": encoded} for _ in range(self.batch_size)],
                "discard_output": self.discard_output,
                "telemetry": self.telemetry,
            }
        }


@dataclass
class Workload:
    """Top-level: a name + the sequence of JobSpecs it generates.

    `concurrency` > 1 means submit `concurrency` JobSpecs in parallel
    via threads. Most patterns use 1 (serial); sustained_concurrent
    uses >1.
    """
    name: str
    specs: list[JobSpec]
    concurrency: int = 1
    description: str = ""


# ────────────────────────────────────────────────────────────────────
# Image generator (deterministic, content doesn't matter — only size)
# ────────────────────────────────────────────────────────────────────

def png_for_size(w: int, h: int) -> bytes:
    """Synthesise a deterministic RGB PNG with a couple of shapes so
    the upscaler has actual edges. Cached by (w, h) so the harness
    doesn't re-encode the same image for every spec."""
    return _png_cache.setdefault((w, h), _make_png(w, h))


_png_cache: dict[tuple[int, int], bytes] = {}


def _make_png(w: int, h: int) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([w // 8, h // 8, 7 * w // 8, 7 * h // 8],
                fill="steelblue", outline="black", width=max(1, w // 64))
    d.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4],
              fill="gold", outline="black", width=max(1, w // 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ────────────────────────────────────────────────────────────────────
# Workload definitions
# ────────────────────────────────────────────────────────────────────

# Reference image size for non-sweep workloads. 720x720 is in the
# middle of the engine's optimisation profile — neither the smallest
# nor largest, so it's representative of the median request shape.
DEFAULT_IMAGE_W = 720
DEFAULT_IMAGE_H = 720


def workload_cold_start() -> Workload:
    """Single fresh request on a freshly-spawned worker. The RUNNER
    is responsible for forcing a cold spawn (idle the endpoint long
    enough that workersMin=0 lets the worker terminate)."""
    return Workload(
        name="cold_start",
        description="One job on a freshly-spawned worker; first-image latency.",
        specs=[
            JobSpec(name="cold_first",
                    batch_size=1, image_w=DEFAULT_IMAGE_W, image_h=DEFAULT_IMAGE_H,
                    telemetry=True, cold_start=True),
        ],
    )


def workload_batch_sweep(max_batch: int = 64,
                         image_w: int = DEFAULT_IMAGE_W,
                         image_h: int = DEFAULT_IMAGE_H) -> Workload:
    """Same image, increasing batch sizes. Reveals the request-overhead
    amortisation curve. Each batch runs after a small warm-up so the
    helper is loaded — we want to measure batch effects, not cold-start
    bleed-through. Top of the sweep capped at `max_batch` to fit in
    smaller-VRAM GPUs without OOM (24 GB cards top out around 64 at
    this resolution; 16 GB tier should be capped lower)."""
    sizes = [b for b in (1, 4, 16, 64) if b <= max_batch]
    specs = [JobSpec(name=f"warmup_b1",
                     batch_size=1, image_w=image_w, image_h=image_h,
                     telemetry=False, cold_start=True)]
    for b in sizes:
        specs.append(JobSpec(name=f"sweep_b{b}",
                             batch_size=b,
                             image_w=image_w, image_h=image_h,
                             telemetry=True))
    return Workload(
        name=f"batch_sweep_{image_w}x{image_h}",
        description=f"Batch sweep at {image_w}x{image_h}: {sizes}",
        specs=specs,
    )


def workload_sustained_concurrent(concurrency: int = 4,
                                  jobs_per_worker: int = 8,
                                  batch_size: int = 4,
                                  image_w: int = DEFAULT_IMAGE_W,
                                  image_h: int = DEFAULT_IMAGE_H) -> Workload:
    """`concurrency` parallel client threads each submit
    `jobs_per_worker` back-to-back jobs. Steady-state throughput
    measurement once RunPod's queue + worker scaling reaches
    equilibrium."""
    specs = [JobSpec(name=f"warmup",
                     batch_size=1, image_w=image_w, image_h=image_h,
                     telemetry=False, cold_start=True)]
    for i in range(concurrency * jobs_per_worker):
        specs.append(JobSpec(name=f"sustained_{i:03d}",
                             batch_size=batch_size,
                             image_w=image_w, image_h=image_h,
                             telemetry=(i % 4 == 0)))  # sample telemetry on 1/4
    return Workload(
        name=f"sustained_c{concurrency}_b{batch_size}",
        description=f"{concurrency} parallel clients × {jobs_per_worker} jobs × batch {batch_size}",
        specs=specs,
        concurrency=concurrency,
    )


def workload_image_size_sweep(sizes: list[int] = None,
                              include_batch_4: bool = True) -> Workload:
    """Sweep across image resolutions at batch=1 (and optionally batch=4
    where the engine admits it). Reveals how per-image cost scales with
    pixel count — informs "best tier for my resolution" decisions and
    quantifies the batched-engine win across sizes.

    Default sizes span the engine's optimisation profiles:
      256, 384, 512 — small (TRT often struggles to saturate small
                     inputs; cost/image is launch-overhead-dominated)
      720           — engine's batched profile max + single-mode opt
      1024, 1280    — single-mode only (above batched profile cap;
                     handler routes through primary engine)
    """
    sizes = sizes or [256, 384, 512, 720, 1024, 1280]
    specs = [JobSpec(name="warmup",
                     batch_size=1, image_w=512, image_h=512,
                     telemetry=False, cold_start=True)]
    for s in sizes:
        specs.append(JobSpec(name=f"size_{s}_b1",
                             batch_size=1, image_w=s, image_h=s,
                             telemetry=True))
        # Only add batch=4 for sizes the batched engine accepts (≤720).
        # Larger sizes fall back to per-image iteration on the primary
        # engine, which is what batch=1 already measures — no new info.
        if include_batch_4 and s <= 720:
            specs.append(JobSpec(name=f"size_{s}_b4",
                                 batch_size=4, image_w=s, image_h=s,
                                 telemetry=True))
    return Workload(
        name=f"image_size_sweep",
        description=f"Per-image cost across sizes {sizes}; batch=1 always, "
                    f"batch=4 for sizes ≤ 720 (batched-engine reach).",
        specs=specs,
    )


# Convenience registry the CLI dispatches against.
WORKLOADS: dict[str, Callable[..., Workload]] = {
    "cold_start": workload_cold_start,
    "batch_sweep": workload_batch_sweep,
    "sustained_concurrent": workload_sustained_concurrent,
    "image_size_sweep": workload_image_size_sweep,
}
