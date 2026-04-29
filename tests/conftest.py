"""Shared pytest configuration + fixtures.

Two responsibilities:

  1. sys.path setup so test files can import `runtime.upscaler` and
     the handler module without requiring the project to be installed
     as a package. Keeps the repo layout flat.
  2. The runpod-handler module imports `runpod` at module load. We
     don't want test environments to need the runpod SDK (and its
     transitive 80-package tree) just to unit-test pure logic. The
     `_pre_import_stubs` autouse fixture installs minimal stubs for
     the handler's import-time dependencies before any test imports
     it. Tests that need the real SDK can override.
"""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make `runtime/` and `providers/runpod/` importable as top-level modules.
# We don't reach for src-layout packaging because the runtime is invoked
# by path in production (Go subprocesses runtime/upscaler.py) — keeping
# them flat preserves the prod-shipping shape.
for sub in ("",  # repo root for runtime.upscaler
            "providers/runpod"):
    p = str(REPO_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def _install_runpod_stubs() -> None:
    """Install minimal stubs for `runpod` and `runpod.serverless` so
    handler.py's top-level imports work in test environments where the
    real runpod SDK isn't installed. The stubs intentionally don't
    implement behaviour — handler.py's bootstrap is gated by
    __name__ == '__main__', so the only things that get hit at import
    time are the symbols themselves (RunPodLogger class, serverless
    submodule reference)."""
    if "runpod" in sys.modules:
        return  # real SDK installed (e.g., inside the runpod images)
    runpod_stub = types.ModuleType("runpod")

    class _Logger:  # noqa: D401 — stub, no behaviour
        def info(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass

    runpod_stub.RunPodLogger = _Logger
    serverless_stub = types.ModuleType("runpod.serverless")
    serverless_stub.start = lambda *a, **k: None
    runpod_stub.serverless = serverless_stub
    sys.modules["runpod"] = runpod_stub
    sys.modules["runpod.serverless"] = serverless_stub


_install_runpod_stubs()


# ────────────────────────────────────────────────────────────────────
# Image fixtures
# ────────────────────────────────────────────────────────────────────

def _make_png(width: int, height: int) -> bytes:
    """Synthesize a deterministic RGB PNG of the given dims. Used by
    both unit-property tests and live tests so the bytes that flow
    into the runtime are identical regardless of test mode."""
    from PIL import Image, ImageDraw  # local import — keep tests collection fast
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    # A few deterministic shapes so the upscaler has actual edges to
    # work on (a uniform white image collapses to a degenerate input).
    d.rectangle([width // 8, height // 8, 7 * width // 8, 7 * height // 8],
                fill="steelblue", outline="black", width=max(1, width // 64))
    d.ellipse([width // 4, height // 4, 3 * width // 4, 3 * height // 4],
              fill="gold", outline="black", width=max(1, width // 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_png():
    """Returns a callable: (w, h) -> bytes (PNG)."""
    return _make_png


@pytest.fixture
def tmp_image(tmp_path, make_png):
    """Returns a callable: (w, h) -> Path to a freshly-saved PNG.
    Useful for tests that pass a path into the runtime helper."""
    counter = {"n": 0}

    def _make(w: int, h: int) -> Path:
        counter["n"] += 1
        p = tmp_path / f"img_{counter['n']}_{w}x{h}.png"
        p.write_bytes(make_png(w, h))
        return p
    return _make


# ────────────────────────────────────────────────────────────────────
# Live-test gating: skip unless a real RunPod endpoint is configured.
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def runpod_env():
    """Returns (endpoint_id, api_key) or skips if either is missing.
    Live tests should depend on this fixture rather than reading env
    directly so the skip message is consistent."""
    endpoint = os.environ.get("RUNPOD_ENDPOINT_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not endpoint or not api_key:
        pytest.skip(
            "live test requires RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY. "
            "Deploy an endpoint first (`make deploy-runpod ... --keep-endpoint`) "
            "and export the resulting endpoint id."
        )
    return endpoint, api_key


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: integration test that hits a real RunPod endpoint "
        "(requires RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY; costs money).",
    )
