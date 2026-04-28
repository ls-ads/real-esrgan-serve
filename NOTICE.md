# Third-party notices and licensing

This project (`real-esrgan-serve`) is licensed under the MIT License
(see [`LICENSE`](./LICENSE)). It redistributes and depends on a number
of third-party works, each governed by its own license. This document
catalogues those dependencies, their licenses, and any obligations
that flow with their use or redistribution.

If you find an attribution missing or incorrect, please open an issue
or PR.

---

## 1. Model weights

The runtime serves Real-ESRGAN model weights produced by upstream:

| Component | Source | License | Obligations |
|---|---|---|---|
| `RealESRGAN_x4plus.pth` (input to our build) | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.1.0 release | BSD-3-Clause | Preserve copyright, license, attribution; no endorsement claim |
| `realesrgan-x4plus_{fp16,fp32}.onnx` (our export of the above) | This repo's GitHub Releases | BSD-3-Clause (inherited from upstream weights) | Same as above |

The exact upstream URL + SHA-256 are pinned in `build/export_onnx.py`
and re-verified at every build, so the upstream `.pth` cannot
silently change.

When redistributing the `.onnx` files (e.g. in a Docker image or a
fork of this repo), include this NOTICE and the upstream BSD-3-Clause
text. The upstream license text is reproduced at
[`third-party-licenses/real-esrgan.txt`](./third-party-licenses/real-esrgan.txt).

---

## 2. Build-time Python dependencies (Stage A export, NOT shipped)

These are used only inside `build/Dockerfile.export` to convert the
upstream `.pth` to `.onnx`. They are **not** present in the runtime
image or the published artefacts.

| Package | Version | License |
|---|---|---|
| basicsr | 1.4.2 | Apache 2.0 |
| realesrgan | 0.3.0 | BSD-3-Clause |
| torch | 1.12.1 | BSD-3-Clause |
| torchvision | 0.13.1 | BSD-3-Clause |
| numpy | 1.23.3 | BSD-3-Clause |
| opencv-python | 4.6.0.66 | Apache 2.0 |
| onnx | 1.12.0 | Apache 2.0 |
| onnxconverter-common | 1.13.0 | MIT |

Even though these aren't redistributed, the BSD/Apache/MIT terms
require attribution preservation if you fork or vendor the build
container.

---

## 3. Runtime Python dependencies (shipped in the image)

| Package | Version | License |
|---|---|---|
| onnxruntime-gpu | 1.18.1 | MIT |
| numpy | 1.26.4 | BSD-3-Clause |
| Pillow | 10.4.0 | HPND (permissive, BSD-style) |
| runpod | 1.7.4 (RunPod provider only) | MIT |
| pydantic | 2.9.2 (RunPod provider only) | MIT |
| requests | 2.32.3 (RunPod provider only) | Apache 2.0 |

**`onnxruntime-gpu` carries an additional notice:** the Microsoft-
published wheel bundles GPU support libraries (CUDA runtime, cuBLAS,
cuFFT, etc.) that originate with NVIDIA and are governed by the
NVIDIA Software License Agreement / CUDA Toolkit EULA. Microsoft has
arranged the wheel-level redistribution; downstream redistribution
(e.g. in a Docker image) is permitted under those terms.
See [https://onnxruntime.ai/docs/install/](https://onnxruntime.ai/docs/install/)
and the wheel's `LICENSE-nvidia` for the operative text.

`runpod` (the Python SDK) drags in a long transitive dependency tree
(fastapi, boto3, paramiko, sentry-sdk, etc.). pip's metadata reports
all of them as MIT, Apache 2.0, BSD, HPND, or PSF — none are GPL,
AGPL, or LGPL. A full rebuild that runs `pip-licenses` or
`license-checker` against the image is the canonical way to refresh
this list.

---

## 4. Go runtime dependencies (compiled into the binary)

| Module | License |
|---|---|
| github.com/spf13/cobra | Apache 2.0 |
| github.com/spf13/pflag | BSD-3-Clause |
| github.com/inconshreveable/mousetrap | Apache 2.0 |

The Go binary statically links these into `/usr/local/bin/real-esrgan-serve`.
Apache 2.0 + BSD-3-Clause require attribution preservation but no
source disclosure.

---

## 5. Docker base images

| Image | Where used | License / EULA |
|---|---|---|
| `nvidia/cuda:12.4.1-base-ubuntu22.04` | base runtime + provider images | NVIDIA CUDA EULA + Ubuntu 22.04 (a mix of GPL/LGPL/MIT/BSD; see Ubuntu's package licenses) |
| `golang:1.25-alpine` | first stage of base image; Go binary build only | BSD-3-Clause (Go) + MIT (Alpine) |
| `ghcr.io/astral-sh/uv:python3.9-bookworm-slim` | build/Dockerfile.export only — not in runtime image | Apache 2.0 / MIT (uv) + Debian licenses (mixed) |

**NVIDIA CUDA EULA highlights:**

- Permits redistribution of the CUDA runtime libraries inside container
  images that run on end-user GPU hardware.
- Forbids reverse-engineering or modifying the CUDA libraries.
- Requires the EULA text to remain accessible (it ships in
  `/NGC-DL-CONTAINER-LICENSE` in the base image; we leave it in
  place).

Full EULA: https://docs.nvidia.com/cuda/eula/index.html

---

## 6. Provider SDKs

| SDK | License | Notes |
|---|---|---|
| RunPod Python SDK (`runpod`) | MIT | Optional. Only present in `providers/runpod/` images. Consumers using a different provider don't inherit it. |

---

## 7. Tooling we invoke at release time (not redistributed)

- GitHub Actions workflows in `.github/workflows/` use Apache 2.0
  `actions/checkout`, `actions/setup-python`, `actions/upload-artifact`,
  `actions/download-artifact`. None of these ship in published
  artefacts.
- `gh` CLI used in the workflow is MIT.

---

## Distribution checklist

Before publishing a release or container image, verify:

- [ ] `models/MANIFEST.json` lists the correct license text URL for
      each model artefact (the `license_url` field).
- [ ] This `NOTICE.md` is included in any source tarball or container
      image at a discoverable path (e.g. `/usr/share/doc/real-esrgan-serve/`
      or repo root).
- [ ] The repo's own `LICENSE` (MIT) is preserved.
- [ ] The upstream Real-ESRGAN `LICENSE` text is preserved at
      [`third-party-licenses/real-esrgan.txt`](./third-party-licenses/real-esrgan.txt)
      — this is required by BSD-3-Clause.
- [ ] If you have replaced the model weights with a different model,
      update the `license` and `license_url` fields in
      `models/MANIFEST.json` and add a corresponding entry to this
      file.

---

## Maintenance

When bumping any dependency, re-validate that:

1. The new version's license is still permissive (MIT / BSD-x /
   Apache 2.0 / HPND / PSF / Python-2.0). **Do not introduce
   GPL, AGPL, LGPL, MPL, or proprietary-licensed dependencies
   without documenting the implications here first.**
2. If the dependency is redistributed in the runtime image (Section 3),
   update the version in this document.
3. If the dependency carries a NOTICE file (Apache 2.0 requirement),
   the NOTICE text is preserved in the runtime image (typically at
   `/usr/share/doc/<package>/NOTICE`).

A simple sanity check:

```bash
# Inside the runtime image
docker run --rm --entrypoint bash real-esrgan-serve:runpod-test -c \
  "pip3 show $(pip3 list --format=freeze | cut -d= -f1) 2>/dev/null | \
   grep -E '^(Name|License):' | paste - - | sort -u"
```

Any output of `License: GPL` / `License: LGPL` / `License: AGPL` /
`License: UNKNOWN` warrants investigation before the image is
published.
