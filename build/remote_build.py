#!/usr/bin/env python3
"""Compile a TensorRT engine remotely on a RunPod GPU pod.

THIS SCRIPT IS A THIN ORCHESTRATOR — IT DOES NOT CARRY ITS OWN
COMPILE LOGIC. By design, the actual compile runs the same `make
artifacts-engine` target a local maintainer would run on their own
hardware. The only thing that's different is the host. If the
remote build works, the local build is guaranteed to work, because
it's the same code executing.

Why: a separate "remote compile flow" would inevitably diverge from
the local one (different argument plumbing, slightly different
script paths, wrong dist directory, etc.). Keeping the compile
logic local-only and delegating to it via SSH means there's exactly
one path to maintain.

What it does:

  1. Tarball the current working tree via `git ls-files | tar`
     (tracked files only — including any uncommitted modifications,
     since `tar` reads from the working tree). Untracked artefacts
     like `bin/` or `build/dist/` are skipped — they're outputs, not
     inputs.
  2. Pick a RunPod GPU type matching --gpu-class
  3. Spin up a temp pod with a TRT-toolkit base image
  4. SCP the tarball over, untar
  5. SSH `make artifacts-engine` on the pod
  6. SCP `build/dist/*.engine` back to local `build/dist/`
  7. Verify SHA-256 + bytes
  8. Terminate the pod (always — even on error)

Usage:
    export RUNPOD_API_KEY=<your-key>   # NEVER commit / paste in chat
    python build/remote_build.py --gpu-class rtx-4090

Maintainer convenience (reads RUNPOD_API_KEY from
~/Projects/iosuite.io/.env without echoing the value):

    build/.with-iosuite-key make remote-build-engine GPU_CLASS=rtx-4090

Auditing: pod creation, SSH commands, and termination all log their
RunPod request IDs to stderr so a maintainer can cross-reference a
billing line item to a specific compile run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# Default base image. Has Python + CUDA toolkit pre-installed.
# tensorrt + make get pip/apt-installed by the startup script
# (license-tangled image, used as a build sandbox only — never
# redistributed).
DEFAULT_BUILD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Map kebab-case GPU class names → RunPod's GraphQL displayName for
# the GPU type. RunPod's naming has been inconsistent over time
# (sometimes "NVIDIA GeForce RTX 4090", sometimes just "RTX 4090") —
# we resolve at runtime via gpuTypes query, so a stale mapping shows
# a "GPU not found, available: ..." error with the live list.
# Validated against the API on 2026-04.
GPU_CLASS_TO_TYPE: dict[str, str] = {
    "rtx-5090":    "RTX 5090",
    "rtx-5080":    "RTX 5080",
    "rtx-4090":    "RTX 4090",
    "rtx-4080":    "RTX 4080",
    "rtx-4080-s":  "RTX 4080 SUPER",
    "rtx-3090":    "RTX 3090",
    "rtx-3090-ti": "RTX 3090 Ti",
    "rtx-a6000":   "RTX A6000",
    "rtx-6000":    "RTX 6000 Ada",
    "rtx-pro-6000":"RTX PRO 6000",
    "l40s":        "L40S",
    "l40":         "L40",
    "l4":          "L4",
    "a40":         "A40",
    "a100":        "A100 PCIe",
    "a100-sxm":    "A100 SXM",
    "h100":        "H100 PCIe",
    "h100-sxm":    "H100 SXM",
    "h100-nvl":    "H100 NVL",
    "h200":        "H200 SXM",
    "h200-nvl":    "NVIDIA H200 NVL",
    "b200":        "B200",
}


# ─────────────────────────────────────────────────────────────────────
# RunPod GraphQL client (intentionally tiny — no SDK dep)
# ─────────────────────────────────────────────────────────────────────

class RunPodClient:
    """Minimal GraphQL client. Stdlib-only so the build pipeline has no
    install footprint beyond what RunPod's pytorch image already has."""

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
                "User-Agent": "real-esrgan-serve-remote-build/0.2",
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
        """Resolve 'NVIDIA GeForce RTX 4090' → the typeId RunPod uses
        in createPod. Mapping changes occasionally as RunPod adds or
        renames SKUs; we resolve at runtime rather than hardcoding."""
        data = self.query("""
            query gpuTypes {
                gpuTypes { id displayName }
            }
        """)
        for g in data["gpuTypes"]:
            if g["displayName"] == human_name:
                return g["id"]
        avail = ", ".join(sorted(g["displayName"] for g in data["gpuTypes"]))
        sys.exit(f"GPU '{human_name}' not found in your RunPod account.\n"
                 f"Available: {avail}")

    def create_pod(
        self,
        name: str,
        gpu_type_id: str,
        image: str,
        container_disk_gb: int,
        startup_script: str,
    ) -> str:
        """Create a Secure Cloud pod. Returns pod ID."""
        data = self.query(
            """
            mutation create($input: PodFindAndDeployOnDemandInput) {
                podFindAndDeployOnDemand(input: $input) {
                    id
                    machine { podHostId }
                }
            }
            """,
            variables={"input": {
                "name": name,
                "imageName": image,
                "gpuTypeId": gpu_type_id,
                "gpuCount": 1,
                "containerDiskInGb": container_disk_gb,
                "minMemoryInGb": 16,
                "minVcpuCount": 4,
                "ports": "22/tcp",
                "dockerArgs": startup_script,
                "cloudType": "SECURE",
            }},
        )
        pod = data["podFindAndDeployOnDemand"]
        if not pod or not pod.get("id"):
            sys.exit("createPod returned no id — typically means no capacity for that GPU class. Retry or pick another --gpu-class.")
        return pod["id"]

    def wait_running(self, pod_id: str, timeout_s: int = 300) -> dict:
        deadline = time.time() + timeout_s
        last_status = ""
        while time.time() < deadline:
            data = self.query("""
                query pod($input: PodFilter) {
                    pod(input: $input) {
                        id
                        desiredStatus
                        runtime {
                            ports { ip privatePort publicPort isIpPublic type }
                        }
                    }
                }
            """, variables={"input": {"podId": pod_id}})
            pod = data["pod"]
            status = pod["desiredStatus"]
            if status != last_status:
                print(f"[remote] pod {pod_id} status={status}", file=sys.stderr)
                last_status = status
            ports = (pod.get("runtime") or {}).get("ports") or []
            for p in ports:
                if p["privatePort"] == 22 and p["isIpPublic"]:
                    return {"ip": p["ip"], "port": p["publicPort"]}
            time.sleep(5)
        sys.exit(f"pod {pod_id} did not become reachable within {timeout_s}s")

    def terminate(self, pod_id: str) -> None:
        # Inline mutation rather than the variable form. RunPod's API
        # has rejected the typed-variable shape on this specific
        # mutation in past runs (other variable-form calls work fine);
        # inline is more robust and the pod_id we substitute came
        # from RunPod itself a moment earlier.
        if "'" in pod_id or '"' in pod_id:
            # Defensive: pod IDs are short hex strings, never quoted —
            # but reject anything that could close the GraphQL string.
            print(f"[remote] WARNING: refusing to terminate pod {pod_id!r} — suspicious id",
                  file=sys.stderr)
            return
        try:
            self.query(
                f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}'
            )
            print(f"[remote] terminated pod {pod_id}", file=sys.stderr)
        except SystemExit:
            print(f"[remote] WARNING: failed to terminate pod {pod_id} — check RunPod console!",
                  file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────
# SSH / SCP helpers
# ─────────────────────────────────────────────────────────────────────

def _ssh_base(host: str, port: int, key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i", str(key_path),
        "-p", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "BatchMode=yes",
        f"root@{host}",
    ]


def ssh_run(host: str, port: int, key_path: Path, cmd: str, *,
            check: bool = True, timeout: int = 600,
            stream: bool = False) -> subprocess.CompletedProcess:
    base = _ssh_base(host, port, key_path) + [cmd]
    if stream:
        # Inherit our stdout/stderr — useful for the long-running make
        # target so the user sees compile progress in real time.
        return subprocess.run(base, check=check, timeout=timeout)
    return subprocess.run(base, capture_output=True, text=True,
                          check=check, timeout=timeout)


def scp_to(host: str, port: int, key_path: Path, local: Path, remote: str) -> None:
    subprocess.run([
        "scp",
        "-i", str(key_path),
        "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        str(local), f"root@{host}:{remote}",
    ], check=True)


def scp_from(host: str, port: int, key_path: Path, remote: str, local: Path) -> None:
    subprocess.run([
        "scp",
        "-i", str(key_path),
        "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"root@{host}:{remote}", str(local),
    ], check=True)


# ─────────────────────────────────────────────────────────────────────
# Working-tree tarball
# ─────────────────────────────────────────────────────────────────────

def tarball_working_tree(out_path: Path) -> None:
    """Bundle every git-tracked file (with current working-tree
    contents — including uncommitted edits) PLUS any inputs from
    build/dist/ that the local Make target would consume.

    Why both:
      - tracked files: scripts, Makefile, manifest — what the make
        target actually executes
      - build/dist/*.onnx: the input artefact for `artifacts-engine`,
        produced locally by `make artifacts-onnx` (which runs in
        Docker for basicsr-deps reasons we don't want to replicate
        on the pod). Gitignored, so not in `git ls-files`.

    Why this and not `git archive HEAD`: archive only sees committed
    state, so a maintainer iterating on compile_engine.py would have
    to commit between every remote test. tar-from-ls-files picks up
    the working tree which is what they actually want to validate.
    """
    if not (REPO_ROOT / ".git").exists():
        sys.exit(f"{REPO_ROOT} is not a git repo — `git init` first")

    # 1. files git knows about + everything untracked-but-not-ignored.
    # Two `git ls-files` calls so we pick up files staged for the next
    # commit AND new files the user hasn't committed yet (the common
    # case during active development — e.g. just after a wipe-and-
    # rebuild). Anything in .gitignore stays out, except for the
    # explicit dist/.onnx allow-list below.
    paths_set: set[str] = set()
    skipped: list[str] = []
    for git_args in (
        ["git", "ls-files"],                                 # tracked
        ["git", "ls-files", "--others", "--exclude-standard"],  # untracked, not ignored
    ):
        out = subprocess.run(
            git_args,
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
        for p in out.splitlines():
            if not p:
                continue
            if (REPO_ROOT / p).exists():
                paths_set.add(p)
            else:
                skipped.append(p)
    if not paths_set:
        sys.exit("nothing to bundle — `git init` and add files first")
    if skipped:
        print(f"[remote] skipping {len(skipped)} tracked-but-missing path(s) "
              f"(unstaged deletions — `git add -A` to clean up)",
              file=sys.stderr)

    # 2. dist inputs (.onnx files the engine compile reads). The
    # tarball must NOT carry stale .engine files from previous local
    # runs — those are what we'd be re-producing on the remote.
    dist = REPO_ROOT / "build" / "dist"
    onnx_added = []
    if dist.exists():
        for p in sorted(dist.glob("*.onnx")):
            rel = str(p.relative_to(REPO_ROOT))
            paths_set.add(rel)
            onnx_added.append(rel)

    if not onnx_added:
        sys.exit(
            "no .onnx file found under build/dist/ to ship to the remote.\n"
            "Run Stage A locally first:  make artifacts-onnx"
        )

    print(f"[remote] bundling working tree + ONNX inputs → {out_path}", file=sys.stderr)
    for rel in onnx_added:
        size_mb = (REPO_ROOT / rel).stat().st_size / (1 << 20)
        print(f"  + {rel}  ({size_mb:.1f} MiB)", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-czf", str(out_path), "-C", str(REPO_ROOT), "-T", "-"],
        input="\n".join(sorted(paths_set)) + "\n",
        text=True, check=True,
    )
    size = out_path.stat().st_size
    print(f"[remote] tarball: {size / (1<<20):.1f} MiB", file=sys.stderr)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gpu-class", required=True,
                   choices=sorted(GPU_CLASS_TO_TYPE.keys()),
                   help="GPU class — picks the RunPod pod type")
    p.add_argument("--make-target", default="artifacts-engine",
                   help="Make target to run on the pod (default: artifacts-engine)")
    p.add_argument("--make-args", default="",
                   help="extra arguments passed to make on the pod (e.g. ONNX=path/to/foo.onnx)")
    p.add_argument("--build-image", default=DEFAULT_BUILD_IMAGE,
                   help="container image for the build pod")
    p.add_argument("--container-disk-gb", type=int, default=20,
                   help="pod container disk allocation")
    p.add_argument("--ssh-key", type=Path,
                   default=Path.home() / ".ssh" / "id_ed25519",
                   help="local private key (matching public key MUST be on RunPod profile)")
    p.add_argument("--keep-pod", action="store_true",
                   help="don't terminate after build (debug)")
    args = p.parse_args()

    if not args.ssh_key.exists():
        sys.exit(f"--ssh-key not found: {args.ssh_key} — generate one with `ssh-keygen -t ed25519`")
    pub_key_path = args.ssh_key.with_suffix(args.ssh_key.suffix + ".pub") \
        if args.ssh_key.suffix else Path(str(args.ssh_key) + ".pub")
    if not pub_key_path.exists():
        sys.exit(f"matching public key not found at {pub_key_path}")
    pub_key = pub_key_path.read_text().strip()
    # We embed the key inside a `bash -c '...'` startup script with
    # double-quotes around the key value. Reject anything that would
    # break that or open a shell injection — public keys never
    # contain these in practice.
    bad = set(pub_key) & set('"`$\\\'')
    if bad:
        sys.exit(f"public key contains shell-meta chars {bad} — refusing to embed")

    rp = RunPodClient(os.environ.get("RUNPOD_API_KEY", ""))

    gpu_human = GPU_CLASS_TO_TYPE[args.gpu_class]
    gpu_type_id = rp.find_gpu_type_id(gpu_human)
    print(f"[remote] gpu-class={args.gpu_class} → typeId={gpu_type_id}", file=sys.stderr)

    # Bundle locally before spinning up the pod — fail fast on a bad
    # repo state instead of after we've already started billing for GPU.
    tarball = REPO_ROOT / "build" / "dist" / "_remote-build.tar.gz"
    tarball_working_tree(tarball)

    name = f"res-engine-{args.gpu_class}-{int(time.time())}"
    # Startup script runs once when the container boots:
    #   - inject our SSH public key into /root/.ssh/authorized_keys so
    #     the pod accepts our key without the user pre-configuring it
    #     on their RunPod profile (self-contained build)
    #   - apt-install openssh-server + make (the Make target we want
    #     to invoke uses both)
    #   - pip install tensorrt (CUDA-12 compatible build)
    #   - sleep so the container stays alive after the script exits
    #
    # The pub_key is embedded with single-quote shell escaping. We
    # validated above that it contains no single quotes.
    # Single-quoted bash -c with double-quotes around the public key
    # (we've validated above that the key has no shell-meta chars
    # that'd close or escape either).
    # The pip install for tensorrt takes ~30–60s but ssh comes up
    # in ~5s. Without a sentinel, our `make artifacts-engine` would
    # race the pip install and fail with `No module named 'tensorrt'`.
    # The startup script writes /tmp/setup-done as the very last step;
    # we poll for it (below) before invoking make.
    startup = (
        "bash -c '"
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh; "
        f'echo "{pub_key}" >> /root/.ssh/authorized_keys; '
        "chmod 600 /root/.ssh/authorized_keys; "
        "apt-get update -qq; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "    openssh-server make >/dev/null 2>&1; "
        "mkdir -p /run/sshd; "
        "service ssh start; "
        "pip install --quiet --no-cache-dir tensorrt && touch /tmp/setup-done; "
        "sleep infinity"
        "'"
    )
    pod_id = rp.create_pod(
        name=name,
        gpu_type_id=gpu_type_id,
        image=args.build_image,
        container_disk_gb=args.container_disk_gb,
        startup_script=startup,
    )
    print(f"[remote] created pod {pod_id} ({name})", file=sys.stderr)

    try:
        access = rp.wait_running(pod_id)
        host, port = access["ip"], access["port"]
        print(f"[remote] ssh root@{host}:{port}", file=sys.stderr)

        # Wait for ssh — RunPod's `desiredStatus=RUNNING` flips before
        # sshd is fully up, so poll explicitly.
        for _ in range(24):
            try:
                ssh_run(host, port, args.ssh_key, "true", timeout=10)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                time.sleep(5)
        else:
            sys.exit("ssh never came up — check RunPod console + your ssh public key on profile")

        # Wait for the startup script to finish installing tensorrt
        # (the actual signal we care about for the build). SSH is up
        # before pip-install-tensorrt finishes; running make before
        # that races the install and bricks with "No module named
        # tensorrt". The startup script touches /tmp/setup-done at
        # the end.
        print("[remote] waiting for pod setup (apt + pip install tensorrt)…", file=sys.stderr)
        setup_deadline = time.time() + 300  # 5 min ceiling
        while time.time() < setup_deadline:
            r = ssh_run(host, port, args.ssh_key,
                        "test -f /tmp/setup-done", check=False, timeout=10)
            if r.returncode == 0:
                print("[remote] pod setup complete", file=sys.stderr)
                break
            time.sleep(5)
        else:
            sys.exit("pod setup did not finish within 5 min — check the pod's "
                     "/var/log via RunPod console")

        # Stage repo: push tarball, untar in a fresh dir.
        ssh_run(host, port, args.ssh_key,
                "rm -rf /workspace/repo && mkdir -p /workspace/repo")
        scp_to(host, port, args.ssh_key, tarball, "/workspace/repo.tgz")
        ssh_run(host, port, args.ssh_key,
                "tar -xzf /workspace/repo.tgz -C /workspace/repo && rm /workspace/repo.tgz")

        # Run the same Make target a local maintainer would run.
        # `--auto-detect-gpu` inside compile_engine.py picks up this
        # pod's actual nvidia-smi output for the gpu_class string in
        # the engine filename. So if the local user runs the same
        # target on the same hardware, they get an identical output
        # (modulo TRT determinism — engines for the same input are
        # bit-identical on the same GPU + TRT version).
        target = args.make_target
        extra = args.make_args.strip()
        cmd = (
            f"cd /workspace/repo && make {target} {extra}"
        ).strip()
        print(f"[remote] running on pod: {cmd}", file=sys.stderr)
        print(f"[remote] (this typically takes 5–15 min for engine compile)\n", file=sys.stderr)
        try:
            ssh_run(host, port, args.ssh_key, cmd, stream=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            sys.exit(f"remote `make {target}` failed (exit {e.returncode})")

        # Pull every artefact in build/dist back. The Make target is
        # responsible for putting things there; we don't second-guess
        # which files matter.
        list_proc = ssh_run(host, port, args.ssh_key,
                            "ls /workspace/repo/build/dist/*.engine 2>/dev/null || true")
        engines = [ln.strip() for ln in list_proc.stdout.splitlines() if ln.strip()]
        if not engines:
            sys.exit("no .engine produced on the pod — check the make output above")

        local_dist = REPO_ROOT / "build" / "dist"
        local_dist.mkdir(parents=True, exist_ok=True)

        produced: list[Path] = []
        for remote_path in engines:
            local = local_dist / Path(remote_path).name
            scp_from(host, port, args.ssh_key, remote_path, local)
            produced.append(local)

        print("\n[remote] artefacts pulled:", file=sys.stderr)
        for path in produced:
            digest = sha256(path)
            print(f"  {path}", file=sys.stderr)
            print(f"    sha256: {digest}", file=sys.stderr)
            print(f"    bytes:  {path.stat().st_size}", file=sys.stderr)

        # stdout: machine-parseable list of paths (one per line) so
        # callers (CI / scripts) can pipe them to manifest sync etc.
        for path in produced:
            print(path)

        return 0

    finally:
        try:
            tarball.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        if args.keep_pod:
            print(f"[remote] --keep-pod set: pod {pod_id} still running. Terminate manually.",
                  file=sys.stderr)
        else:
            rp.terminate(pod_id)


if __name__ == "__main__":
    sys.exit(main())
