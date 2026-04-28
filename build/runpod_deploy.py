#!/usr/bin/env python3
"""Deploy real-esrgan-serve to a RunPod serverless endpoint and run a
cold-start + warm-latency smoke test.

Repeatable + auditable for any maintainer with a RunPod account:

  1. Validate $RUNPOD_API_KEY (silent — never echoed).
  2. Resolve the requested GPU class via gpuTypes query.
  3. Create or update the serverless endpoint with our image,
     workersMin=0 (so we can force cold starts on demand),
     workersMax configurable.
  4. Wait for the endpoint to be in a runnable state.
  5. Submit one cold-start job. Time it end-to-end.
  6. Submit N warm jobs in series. Time each, compute p50/p95.
  7. Print a summary that's machine-parseable on stdout (JSON line)
     and human-readable on stderr.
  8. Optionally tear down the endpoint (--cleanup); default is to
     leave it running so iosuite.io / other consumers can use it.

Usage:
    export RUNPOD_API_KEY=<your-key>
    python build/runpod_deploy.py \\
        --image ghcr.io/ls-ads/real-esrgan-serve:runpod-test \\
        --gpu-class rtx-4090 \\
        --endpoint-name real-esrgan-serve-test \\
        --warmup-jobs 5

Maintainer convenience:
    build/.with-iosuite-key python build/runpod_deploy.py ...
    # or via Makefile:
    build/.with-iosuite-key make e2e-runpod GPU_CLASS=rtx-4090

The "cold start" we measure is end-to-end from the user's perspective:
submission timestamp → first byte of response. That's the metric that
matters for actual UX. RunPod's response also includes its own
delayTime + executionTime breakdown which we report alongside.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# Serverless endpoints use GPU POOLS (grouped by VRAM + generation),
# not specific GPU types. e.g. ADA_24 covers RTX 4090, RTX 4080
# SUPER, RTX 4080. So we map our user-facing kebab-case GPU class
# to its containing pool.
#
# This is different from remote_build.py which spins up Pods (those
# DO take a specific GPU type by displayName). Both files map from
# the same kebab-case input — just to different RunPod identifiers.
#
# Pool IDs come from
# https://docs.runpod.io/references/gpu-types#gpu-pools — keep this
# in sync if RunPod adds new pools.
GPU_CLASS_TO_POOL: dict[str, str] = {
    "rtx-5090":    "BLACKWELL_96",   # 32GB Blackwell
    "rtx-5080":    "BLACKWELL_96",
    "rtx-4090":    "ADA_24",          # 24GB Ada
    "rtx-4080":    "ADA_24",
    "rtx-4080-s":  "ADA_24",
    "rtx-3090":    "AMPERE_24",       # 24GB Ampere
    "rtx-3090-ti": "AMPERE_24",
    "l40s":        "ADA_48_PRO",      # 48GB Ada professional
    "l40":         "ADA_48_PRO",
    "l4":          "ADA_24",
    "a40":         "AMPERE_48",       # 48GB Ampere
    "a100":        "AMPERE_80",       # 80GB Ampere
    "a100-sxm":    "AMPERE_80",
    "h100":        "HOPPER_141",      # 80/141GB Hopper
    "h100-sxm":    "HOPPER_141",
    "h100-nvl":    "HOPPER_141",
    "h200":        "HOPPER_141",
    "b200":        "BLACKWELL_180",
}


# A 64x64 RGB PNG generated deterministically at import time via
# stdlib struct + zlib. Built in-process so the script has no PIL
# dependency on the operator's machine and zero risk of base64
# alignment breakage from string-literal line splits (a previous
# revision hit exactly that bug). The test exercises the full
# upscale path; pixel values don't matter for latency measurement.
def _make_test_png_b64(width: int = 64, height: int = 64) -> str:
    import struct
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    # Each scanline is one filter byte (0 = None) + width*3 RGB bytes.
    row = b"\x00" + (b"\x40\x80\xc0" * width)
    idat = zlib.compress(row * height, 6)
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


_TEST_PNG_64x64_B64 = _make_test_png_b64()


# ─────────────────────────────────────────────────────────────────────
# RunPod GraphQL client — same shape as remote_build.py
# ─────────────────────────────────────────────────────────────────────

class RunPodClient:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            sys.exit("RUNPOD_API_KEY not set. Either:\n"
                     "  export RUNPOD_API_KEY=<your-key>\n"
                     "or use build/.with-iosuite-key wrapper.")
        self._api_key = api_key

    def query(self, query: str, variables: dict | None = None) -> dict:
        body = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(
            RUNPOD_GRAPHQL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "real-esrgan-serve-runpod-deploy/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            sys.exit(f"runpod http {e.code}: {e.read().decode(errors='replace')}")
        if "errors" in resp:
            sys.exit(f"runpod errors: {resp['errors']}")
        return resp["data"]

    def find_gpu_type_id(self, human_name: str) -> str:
        data = self.query("query { gpuTypes { id displayName } }")
        for g in data["gpuTypes"]:
            if g["displayName"] == human_name:
                return g["id"]
        avail = ", ".join(sorted(g["displayName"] for g in data["gpuTypes"]))
        sys.exit(f"GPU '{human_name}' not found.\nAvailable: {avail}")

    def find_endpoint(self, name: str) -> dict | None:
        data = self.query("query { myself { endpoints { id name templateId } } }")
        for e in data["myself"]["endpoints"]:
            if e["name"] == name:
                return e
        return None

    def find_template(self, name: str) -> dict | None:
        # RunPod calls these "podTemplates" but they're used by
        # serverless endpoints too via `isServerless: true`.
        data = self.query("query { myself { podTemplates { id name imageName } } }")
        for t in data["myself"]["podTemplates"]:
            if t["name"] == name:
                return t
        return None

    def find_registry_auth_for(self, registry: str) -> str | None:
        """Look up the RunPod container-registry credential matching
        the given registry hostname (e.g. 'ghcr.io'). Returns the
        auth id if found, None otherwise. Lets the deploy work with
        private images without baking a token into this repo."""
        try:
            data = self.query(
                "query { myself { containerRegistryCreds { id name registryAuth } } }"
            )
        except SystemExit:
            # Some RunPod accounts don't have this field exposed.
            return None
        for cred in (data.get("myself", {}).get("containerRegistryCreds") or []):
            # `registryAuth` is an opaque identifier, often the
            # registry hostname or a Docker auth blob. Most accounts
            # have a friendly `name` like "ghcr-pull" that we match
            # case-insensitively.
            label = (cred.get("name") or "") + " " + (cred.get("registryAuth") or "")
            if registry.lower() in label.lower():
                return cred["id"]
        return None

    def save_template(
        self,
        name: str,
        image: str,
        container_disk_gb: int,
        env: list[dict] | None = None,
        existing_id: str | None = None,
        registry_auth_id: str | None = None,
    ) -> str:
        """RunPod serverless endpoints reference a template (image +
        disk + env). saveTemplate creates or updates by id."""
        env = env or []
        env_json = json.dumps(env)
        id_field = f'id: "{existing_id}",' if existing_id else ""
        auth_field = f'containerRegistryAuthId: "{registry_auth_id}",' \
            if registry_auth_id else ""
        # `dockerArgs` and `volumeInGb` are required by the schema
        # even though they're optional in concept here:
        #   - dockerArgs: empty string lets the image's CMD run as-is
        #   - volumeInGb: 0 = no persistent volume (serverless workers
        #     get fresh containers; persistent volumes are a paid feature
        #     that doesn't help cold-start measurement anyway)
        mutation = f"""
        mutation saveTemplate {{
            saveTemplate(input: {{
                {id_field}
                {auth_field}
                name: "{name}",
                imageName: "{image}",
                containerDiskInGb: {container_disk_gb},
                volumeInGb: 0,
                dockerArgs: "",
                isServerless: true,
                env: {env_json}
            }}) {{ id name }}
        }}
        """
        data = self.query(mutation)
        return data["saveTemplate"]["id"]

    def save_endpoint(
        self,
        name: str,
        template_id: str,
        gpu_type_id: str,
        workers_min: int,
        workers_max: int,
        idle_timeout_s: int,
        existing_id: str | None = None,
    ) -> str:
        """saveEndpoint creates or updates by id. Endpoint references
        a template (image + disk) and adds GPU/scaling config."""
        id_field = f'id: "{existing_id}",' if existing_id else ""
        mutation = f"""
        mutation saveEndpoint {{
            saveEndpoint(input: {{
                {id_field}
                name: "{name}",
                templateId: "{template_id}",
                gpuIds: "{gpu_type_id}",
                workersMin: {workers_min},
                workersMax: {workers_max},
                idleTimeout: {idle_timeout_s},
                scalerType: "QUEUE_DELAY",
                scalerValue: 4
            }}) {{ id name }}
        }}
        """
        data = self.query(mutation)
        return data["saveEndpoint"]["id"]

    def delete_endpoint(self, endpoint_id: str) -> None:
        try:
            # Workers must be drained before delete; set both to 0.
            self.query(f"""
                mutation {{
                    saveEndpoint(input: {{
                        id: "{endpoint_id}",
                        workersMin: 0,
                        workersMax: 0
                    }}) {{ id }}
                }}
            """)
            time.sleep(2)
            self.query(f'mutation {{ deleteEndpoint(id: "{endpoint_id}") }}')
            print(f"[deploy] deleted endpoint {endpoint_id}", file=sys.stderr)
        except SystemExit:
            print(f"[deploy] WARNING: failed to delete endpoint {endpoint_id} — "
                  f"check RunPod console!", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────
# Job submission via the v2 REST API
# ─────────────────────────────────────────────────────────────────────

def submit_job(endpoint_id: str, api_key: str, payload: dict, timeout_s: int = 600) -> dict:
    """POST a job, poll status until COMPLETED or FAILED. Returns the
    final status response (which includes timing fields)."""
    submit_url = f"{RUNPOD_API_BASE}/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "real-esrgan-serve-runpod-deploy/0.1",
    }

    submit_t0 = time.monotonic()
    req = urllib.request.Request(
        submit_url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        submit_resp = json.loads(r.read())
    job_id = submit_resp["id"]

    status_url = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req = urllib.request.Request(status_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            status = json.loads(r.read())
        if status["status"] in ("COMPLETED", "FAILED"):
            wall_time_s = time.monotonic() - submit_t0
            status["_walltime_s"] = wall_time_s
            return status
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not complete within {timeout_s}s")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.1f} s"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--image", required=True,
                   help="container image to deploy (e.g. ghcr.io/ls-ads/real-esrgan-serve:runpod-test)")
    p.add_argument("--gpu-class", required=True,
                   choices=sorted(GPU_CLASS_TO_POOL.keys()),
                   help="GPU class — picks the RunPod GPU pool the endpoint runs on")
    p.add_argument("--endpoint-name", default="real-esrgan-serve-test",
                   help="serverless endpoint name (will create or update)")
    p.add_argument("--workers-max", type=int, default=2,
                   help="max concurrent workers. Default 2 because RunPod throttles "
                        "single-worker endpoints when the GPU pool is busy — having "
                        "two slots gives the scheduler a second chance to find "
                        "capacity. The cold-start measurement still hits the FIRST "
                        "(cold) worker; the second only spawns under load.")
    p.add_argument("--idle-timeout", type=int, default=5,
                   help="seconds before idle worker shuts down — short = easier to force cold")
    p.add_argument("--container-disk-gb", type=int, default=10,
                   help="container disk allocation")
    p.add_argument("--warmup-jobs", type=int, default=5,
                   help="number of warm-state jobs to time after the cold one")
    p.add_argument("--cleanup", action="store_true",
                   help="delete the endpoint after the smoke test (default: keep)")
    p.add_argument("--registry-auth-id",
                   help="RunPod container registry auth id for private images "
                        "(create one at runpod.io/console/user/settings → "
                        "Container Registry Auth). Auto-detected for ghcr.io if "
                        "you have one named 'ghcr*'. Public images don't need this.")
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    rp = RunPodClient(api_key)

    gpu_pool = GPU_CLASS_TO_POOL[args.gpu_class]
    print(f"[deploy] gpu={args.gpu_class} → pool {gpu_pool}", file=sys.stderr)

    # Auto-detect registry auth for ghcr.io if image is on ghcr and
    # no explicit --registry-auth-id was given. Public images skip
    # this entirely; private images need the auth or the worker
    # spawn hangs forever on `docker pull`.
    auth_id = args.registry_auth_id
    if not auth_id and args.image.startswith("ghcr.io/"):
        auto = rp.find_registry_auth_for("ghcr")
        if auto:
            auth_id = auto
            print(f"[deploy] auto-selected ghcr registry auth: {auth_id}",
                  file=sys.stderr)
        else:
            print("[deploy] no ghcr registry auth on RunPod account — "
                  "if image is private, worker spawn will hang on docker pull. "
                  "Either make the package public or create a registry auth at "
                  "runpod.io/console/user/settings → Container Registry Auth, "
                  "then pass --registry-auth-id <id>.",
                  file=sys.stderr)

    # Step 1: template — image + disk + env. Endpoints reference these.
    template_name = args.endpoint_name + "-tmpl"
    existing_tmpl = rp.find_template(template_name)
    if existing_tmpl:
        print(f"[deploy] updating template: {existing_tmpl['id']} ({existing_tmpl.get('imageName')} → {args.image})",
              file=sys.stderr)
    else:
        print(f"[deploy] creating template: {template_name}", file=sys.stderr)
    template_id = rp.save_template(
        name=template_name,
        image=args.image,
        container_disk_gb=args.container_disk_gb,
        existing_id=existing_tmpl["id"] if existing_tmpl else None,
        registry_auth_id=auth_id,
    )
    print(f"[deploy] template id: {template_id}", file=sys.stderr)

    # Step 2: endpoint — points at the template + GPU/scaling.
    existing = rp.find_endpoint(args.endpoint_name)
    if existing:
        print(f"[deploy] updating endpoint: {existing['id']}", file=sys.stderr)
    else:
        print(f"[deploy] creating endpoint: {args.endpoint_name}", file=sys.stderr)
    endpoint_id = rp.save_endpoint(
        name=args.endpoint_name,
        template_id=template_id,
        gpu_type_id=gpu_pool,
        workers_min=0,
        workers_max=args.workers_max,
        idle_timeout_s=args.idle_timeout,
        existing_id=existing["id"] if existing else None,
    )
    print(f"[deploy] endpoint id: {endpoint_id}", file=sys.stderr)

    # If we just updated, give RunPod a beat to register the new image.
    if existing:
        time.sleep(5)

    # ─── Smoke test ──────────────────────────────────────────────
    payload = {
        "input": {
            "image_base64": _TEST_PNG_64x64_B64,
            "output_format": "jpg",
        }
    }

    print(f"\n[deploy] === COLD START === (workers idle, image cold)", file=sys.stderr)
    cold = submit_job(endpoint_id, api_key, payload)
    if cold["status"] != "COMPLETED":
        print(f"[deploy] cold-start job failed: {json.dumps(cold, indent=2)}", file=sys.stderr)
        return 2

    print(f"  walltime:   {fmt_ms(cold['_walltime_s'] * 1000)}", file=sys.stderr)
    print(f"  delayTime:  {fmt_ms(cold.get('delayTime', 0))}    "
          "(queue + worker spawn + image pull + container start)", file=sys.stderr)
    print(f"  execTime:   {fmt_ms(cold.get('executionTime', 0))}    "
          "(handler init + ORT load + first inference)", file=sys.stderr)

    print(f"\n[deploy] === WARM x{args.warmup_jobs} === (worker stays alive)", file=sys.stderr)
    warm_walltimes_ms: list[float] = []
    warm_exec_ms: list[float] = []
    for i in range(args.warmup_jobs):
        warm = submit_job(endpoint_id, api_key, payload)
        if warm["status"] != "COMPLETED":
            print(f"  warm[{i}] FAILED: {warm.get('error')}", file=sys.stderr)
            continue
        wt = warm["_walltime_s"] * 1000
        et = warm.get("executionTime", 0)
        warm_walltimes_ms.append(wt)
        warm_exec_ms.append(et)
        print(f"  warm[{i}]: walltime={fmt_ms(wt)}  exec={fmt_ms(et)}", file=sys.stderr)

    # Summary — JSON to stdout for tooling, table to stderr for humans
    summary = {
        "endpoint_id": endpoint_id,
        "endpoint_name": args.endpoint_name,
        "image": args.image,
        "gpu_class": args.gpu_class,
        "cold_start": {
            "walltime_ms": cold["_walltime_s"] * 1000,
            "delay_ms": cold.get("delayTime", 0),
            "exec_ms": cold.get("executionTime", 0),
        },
        "warm": {
            "n": len(warm_walltimes_ms),
            "walltime_ms": {
                "mean": statistics.mean(warm_walltimes_ms) if warm_walltimes_ms else None,
                "p50":  statistics.median(warm_walltimes_ms) if warm_walltimes_ms else None,
                "p95":  (
                    sorted(warm_walltimes_ms)[int(0.95 * len(warm_walltimes_ms))]
                    if len(warm_walltimes_ms) >= 2 else None
                ),
            },
            "exec_ms": {
                "mean": statistics.mean(warm_exec_ms) if warm_exec_ms else None,
                "p50":  statistics.median(warm_exec_ms) if warm_exec_ms else None,
            },
        },
    }
    print(json.dumps(summary, indent=2))  # stdout: machine-readable

    print("\n[deploy] === SUMMARY ===", file=sys.stderr)
    print(f"  cold start (full e2e):      {fmt_ms(summary['cold_start']['walltime_ms'])}",
          file=sys.stderr)
    print(f"    of which queue + spawn:   {fmt_ms(summary['cold_start']['delay_ms'])}",
          file=sys.stderr)
    print(f"    of which exec on worker:  {fmt_ms(summary['cold_start']['exec_ms'])}",
          file=sys.stderr)
    if warm_exec_ms:
        print(f"  warm walltime mean / p50:    {fmt_ms(summary['warm']['walltime_ms']['mean'])} / "
              f"{fmt_ms(summary['warm']['walltime_ms']['p50'])}",
              file=sys.stderr)
        print(f"  warm exec mean / p50:        {fmt_ms(summary['warm']['exec_ms']['mean'])} / "
              f"{fmt_ms(summary['warm']['exec_ms']['p50'])}",
              file=sys.stderr)

    if args.cleanup:
        rp.delete_endpoint(endpoint_id)
    else:
        print(f"\n[deploy] endpoint left running ({endpoint_id}). "
              f"Use --cleanup to tear down, or RunPod console.",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
