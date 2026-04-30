#!/usr/bin/env python3
"""Validate every deploy/*.json manifest against the documented schema.

Run from the repo root:

    python3 build/validate_manifest.py

CI gate. Failure is non-zero exit + a clear message naming the file
and field. Stdlib-only — no jsonschema dep so the gate stays fast and
cross-platform without an install step.

The schema is hand-written (no JSON Schema doc) because:
  - The shape is small and stable.
  - Custom validation gives clearer error messages than jsonschema's
    structured errors.
  - Avoids a dependency on the jsonschema package in CI.

If we ever grow more than ~3 manifest types, replacing this with a
jsonschema-based validator becomes the right call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "1"

# Allowed values for endpoint.min_cuda_version. Mirrors RunPod's REST
# enum (per their openapi spec, sampled 2026-04). null is allowed for
# tools whose images don't pin a specific driver.
ALLOWED_CUDA = {
    None, "11.8", "12.0", "12.1", "12.2", "12.3", "12.4",
    "12.5", "12.6", "12.7", "12.8", "12.9", "13.0",
}


def fail(path: Path, msg: str) -> None:
    print(f"[validate-manifest] {path}: {msg}", file=sys.stderr)
    sys.exit(1)


def require(path: Path, obj: dict, key: str, want_type) -> object:
    if key not in obj:
        fail(path, f"missing required field {key!r}")
    val = obj[key]
    # Allow None for optional-typed (caller wraps want_type in tuple)
    if not isinstance(val, want_type):
        fail(path, f"field {key!r} expected {want_type.__name__}, got {type(val).__name__}")
    return val


def validate_runpod(path: Path) -> None:
    with path.open() as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            fail(path, f"invalid JSON: {e}")

    sv = require(path, obj, "schema_version", str)
    if sv != SCHEMA_VERSION:
        fail(path, f"schema_version is {sv!r}, validator only knows {SCHEMA_VERSION!r}")

    require(path, obj, "tool", str)
    require(path, obj, "image", str)
    if ":" not in obj["image"]:
        fail(path, f"image {obj['image']!r} missing tag (must be repo:tag)")

    ep = require(path, obj, "endpoint", dict)
    require(path, ep, "container_disk_gb", int)
    require(path, ep, "workers_max_default", int)
    require(path, ep, "idle_timeout_s_default", int)
    require(path, ep, "flashboot_default", bool)
    if "min_cuda_version" not in ep:
        fail(path, "endpoint.min_cuda_version is required (use null to opt out)")
    cuda = ep["min_cuda_version"]
    if cuda not in ALLOWED_CUDA:
        fail(path, f"endpoint.min_cuda_version {cuda!r} not in {sorted(c for c in ALLOWED_CUDA if c)}")

    pools = require(path, obj, "gpu_pools", dict)
    if not pools:
        fail(path, "gpu_pools must declare at least one entry")
    for cls, pool in pools.items():
        if not isinstance(cls, str) or not cls:
            fail(path, f"gpu_pools key {cls!r} must be a non-empty string")
        if not isinstance(pool, str) or not pool:
            fail(path, f"gpu_pools[{cls!r}] must be a non-empty string")

    env = require(path, obj, "env", list)
    for i, e in enumerate(env):
        if not isinstance(e, dict):
            fail(path, f"env[{i}] must be an object")
        require(path, e, "key", str)
        require(path, e, "value", str)


ALLOWED_AGGS = {"mean", "p50", "p95", "p99", "max", "min"}


def validate_benchmark(path: Path) -> None:
    with path.open() as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            fail(path, f"invalid JSON: {e}")

    sv = require(path, obj, "schema_version", str)
    if sv != SCHEMA_VERSION:
        fail(path, f"schema_version is {sv!r}, validator only knows {SCHEMA_VERSION!r}")

    require(path, obj, "tool", str)
    warmup = require(path, obj, "warmup", int)
    measure = require(path, obj, "measure", int)
    if warmup < 0:
        fail(path, f"warmup must be >= 0, got {warmup}")
    if measure <= 0:
        fail(path, f"measure must be > 0, got {measure}")

    res_rel = require(path, obj, "input_resource", str)
    res_path = path.parent.parent / res_rel  # repo root / <relative>
    if not res_path.is_file():
        fail(path, f"input_resource {res_rel!r} not found at {res_path}")

    require(path, obj, "request_template", dict)

    metrics = require(path, obj, "metrics", list)
    if not metrics:
        fail(path, "metrics must declare at least one entry")
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            fail(path, f"metrics[{i}] must be an object")
        require(path, m, "name", str)
        require(path, m, "from", str)
        agg = require(path, m, "agg", str)
        if agg not in ALLOWED_AGGS:
            fail(path, f"metrics[{i}].agg {agg!r} not in {sorted(ALLOWED_AGGS)}")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    deploy = repo / "deploy"
    if not deploy.is_dir():
        print(f"[validate-manifest] no deploy/ directory at {deploy}", file=sys.stderr)
        return 1

    runpod = deploy / "runpod.json"
    if not runpod.exists():
        print(f"[validate-manifest] missing deploy/runpod.json", file=sys.stderr)
        return 1
    validate_runpod(runpod)
    print(f"[validate-manifest] OK: {runpod.relative_to(repo)}")

    bench = deploy / "benchmark.json"
    if bench.exists():
        validate_benchmark(bench)
        print(f"[validate-manifest] OK: {bench.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
