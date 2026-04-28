#!/usr/bin/env python3
"""Update models/MANIFEST.json against the real artefacts in build/dist/.

Reads every file in build/dist/, computes SHA-256 + size, finds the
matching entry in MANIFEST.json (matched by `filename`), updates the
hash + bytes in place. Entries with no matching file in dist/ are
left alone — typical when running this on a host that only built
some variants.

Usage:
    python build/update_manifest.py
    python build/update_manifest.py --check   # exit non-zero on drift, no writes

The `--check` mode is what CI uses post-release to detect manifest
drift before opening a PR. A clean run = manifest already matches
disk = no PR needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "models" / "MANIFEST.json"
DIST = REPO_ROOT / "build" / "dist"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true",
                   help="exit non-zero on drift; do not modify the manifest")
    args = p.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text())
    changed = False
    seen: set[str] = set()

    for entry in manifest.get("models", []):
        name = entry.get("filename")
        if not name:
            continue
        path = DIST / name
        if not path.exists():
            continue  # nothing built for this entry on this host
        seen.add(name)

        sha = _sha256(path)
        size = path.stat().st_size

        if entry.get("sha256") != sha or entry.get("bytes") != size:
            print(f"[manifest] {name}")
            print(f"    sha256: {entry.get('sha256')!r:80} -> {sha!r}")
            print(f"    bytes:  {entry.get('bytes')!r:>20} -> {size}")
            entry["sha256"] = sha
            entry["bytes"] = size
            changed = True

    extras = sorted(p.name for p in DIST.iterdir()
                    if p.is_file() and p.name not in seen and p.suffix == ".onnx")
    if extras:
        print(f"\n[manifest] artefacts in dist/ with no manifest entry: {extras}")
        print("    add a manifest entry for each before publishing.")
        if args.check:
            return 1

    if not changed:
        print("[manifest] up to date — no changes.")
        return 0

    if args.check:
        print("\n[manifest] drift detected (run without --check to apply).")
        return 1

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[manifest] wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
