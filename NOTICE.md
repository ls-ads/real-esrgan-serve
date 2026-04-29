# Third-party notices and licensing

This project (`real-esrgan-serve`) is licensed under the Apache
License 2.0 (see [`LICENSE`](./LICENSE)). It redistributes and
depends on a number of third-party works, each governed by its own
license. This document catalogues those dependencies, their
licenses, and any obligations that flow with their use or
redistribution.

When you redistribute this project (in source, binary, or container
form), Apache 2.0 §4(d) requires you to also redistribute this
NOTICE file alongside your derivative work.

If you find an attribution missing or incorrect, please open an issue
or PR.

---

## 0. real-esrgan-serve

Copyright 2026 Andrew Damon-Smith

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

---

## 1. Model weights

The runtime serves Real-ESRGAN model weights produced by upstream:

| Component | Source | License | Obligations |
|---|---|---|---|
| `RealESRGAN_x4plus.pth` (input to our build) | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) v0.1.0 release | BSD-3-Clause | Preserve copyright, license, attribution; no endorsement claim |
| `realesrgan-x4plus_{fp16,fp32}.onnx` (our export of the above) | This repo's GitHub Releases | BSD-3-Clause (inherited from upstream weights) | Same as above |
| `realesrgan-x4plus-<gpu-class>-<sm-arch>-trt<X.Y>_fp16.engine` (TensorRT-compiled engines for specific GPU classes) | This repo's GitHub Releases | BSD-3-Clause (inherited from the .onnx; engine is a hardware-specific re-encoding of the same weights) | Same as above |

The exact upstream URL + SHA-256 are pinned in `build/export_onnx.py`
and re-verified at every build, so the upstream `.pth` cannot
silently change.

When redistributing the `.onnx` or `.engine` files (e.g. in a Docker
image or a fork of this repo), include this NOTICE and the upstream
BSD-3-Clause text. The upstream license text is reproduced at
[`third-party-licenses/real-esrgan.txt`](./third-party-licenses/real-esrgan.txt).

The `.onnx` is baked into the cpu and cuda images; the trt image
fetches the matching `.engine` for its GPU's SM arch at boot. Either
way, the BSD-3-Clause obligations attach to the binary artefact and
flow with whichever image you redistribute.

---

## 2. Build-time Python dependencies (Stage A export, NOT shipped)

These are used only inside `build/Dockerfile.export` to convert the
upstream `.pth` to `.onnx`. They are **not** present in any runtime
image or published artefact.

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

A separate Stage B compiles `.engine` files on a remote RunPod GPU
pod via `build/remote_build.py`. The TRT compile bundles
`tensorrt-cu12==10.1.0` (NVIDIA proprietary) on the build pod;
nothing from the build pod is redistributed — only the resulting
`.engine` artefact is published, and the engine itself is the
upstream BSD-3-Clause weights re-encoded for a specific (GPU, TRT)
pair.

Even though Stage A/B deps aren't redistributed, the BSD/Apache/MIT
terms require attribution preservation if you fork or vendor the
build container.

---

## 3. Runtime Python dependencies (shipped in the image)

Three runtime image flavors (`cpu`, `cuda`, `trt`) carry different
Python dep sets. Each one is layered with provider-specific Python
deps in the `runpod` variants (Section 6). Versions below match the
images this NOTICE was last audited against.

### 3a. Common to all flavors

| Package | Version | License |
|---|---|---|
| numpy | 1.26.4 | BSD-3-Clause |
| Pillow | 10.4.0 | HPND (permissive, BSD-style) |

### 3b. `cpu` flavor adds

| Package | Version | License |
|---|---|---|
| onnxruntime | 1.20.1 | MIT |

The CPU wheel ships ORT's MIT-licensed shared objects only — no
CUDA or TensorRT plugin libs.

### 3c. `cuda` flavor adds

| Package | Version | License |
|---|---|---|
| onnxruntime-gpu | 1.20.1 | MIT |

The `onnxruntime-gpu` 1.20.1 wheel ships ORT's CUDA-EP plugin
(`libonnxruntime_providers_cuda.so`) and TensorRT-EP plugin (we
don't activate it on the cuda flavor) — both MIT-licensed by
Microsoft. The wheel does **not** bundle copies of cuDNN, cuBLAS,
cuFFT, cuRAND, or libcudart — those come from the base image (see
Section 5).

### 3d. `trt` flavor adds

| Package | Version | License |
|---|---|---|
| `tensorrt` (Python module, provided by apt `python3-libnvinfer`) | 10.1.0 | NVIDIA Software License (proprietary) |
| cuda-python | 12.6.0 | NVIDIA Software License Agreement |

The trt image runs inference through the TensorRT Python API
directly (no ORT). Both the `tensorrt` Python module (installed via
apt's `python3-libnvinfer`, registered to pip metadata as
proprietary) and `cuda-python` are NVIDIA-proprietary. They are
redistributable inside container images that run on end-user GPU
hardware under NVIDIA's TensorRT and CUDA SLAs respectively (see
Section 5).

---

## 4. Go runtime dependencies (compiled into the binary)

| Module | License |
|---|---|
| github.com/spf13/cobra | Apache 2.0 |
| github.com/spf13/pflag | BSD-3-Clause |
| github.com/inconshreveable/mousetrap | Apache 2.0 |
| github.com/cpuguy83/go-md2man/v2 | MIT (cobra's man-page generator; only present in compiled binary if cobra's docgen is invoked) |
| github.com/russross/blackfriday/v2 | BSD-2-Clause (transitive via go-md2man) |

The Go binary statically links these into
`/usr/local/bin/real-esrgan-serve`. Apache 2.0, BSD, and MIT all
require attribution preservation but no source disclosure.

---

## 5. Docker base images and the NVIDIA libraries we redistribute

| Image | Where used | License / EULA |
|---|---|---|
| `ubuntu:22.04` | base of the cpu flavor | Mix of GPL/LGPL/MIT/BSD per package; no NVIDIA components |
| `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | base of the cuda flavor | NVIDIA CUDA EULA + NVIDIA cuDNN SLA + Ubuntu base |
| `nvidia/cuda:12.4.1-base-ubuntu22.04` | base of the trt flavor | NVIDIA CUDA EULA + Ubuntu base |
| `golang:1.25-alpine` | first build stage of every flavor (Go binary build only — not in runtime image) | BSD-3-Clause (Go) + MIT (Alpine) |
| `ghcr.io/astral-sh/uv:python3.9-bookworm-slim` | `build/Dockerfile.export` only — Stage A export, not in runtime image | Apache 2.0 / MIT (uv) + Debian licenses (mixed) |

### 5a. NVIDIA libraries shipped in the `cuda` flavor

The `cudnn-runtime` base bundles NVIDIA's full CUDA-X runtime stack.
We redistribute it as-is (no modification). All components are
governed by their respective NVIDIA SLAs/EULAs.

| Component | Version (audit) | Governing license |
|---|---|---|
| cuda-compat-12-4 | 550.54.15-1 | NVIDIA CUDA EULA |
| cuda-cudart-12-4 | 12.4.127-1 | NVIDIA CUDA EULA |
| cuda-libraries-12-4 (meta) | 12.4.1-1 | NVIDIA CUDA EULA |
| cuda-nvrtc-12-4 | 12.4.127-1 | NVIDIA CUDA EULA |
| cuda-nvtx-12-4 | 12.4.127-1 | NVIDIA CUDA EULA |
| cuda-opencl-12-4 | 12.4.127-1 | NVIDIA CUDA EULA + Khronos OpenCL header license |
| libcublas-12-4 | 12.4.5.8-1 | NVIDIA CUDA EULA |
| libcudnn9-cuda-12 | 9.1.0.70-1 | NVIDIA cuDNN Software License Agreement |
| libcufft-12-4 | 11.2.1.3-1 | NVIDIA CUDA EULA |
| libcufile-12-4 | 1.9.1.3-1 | NVIDIA CUDA EULA |
| libcurand-12-4 | 10.3.5.147-1 | NVIDIA CUDA EULA |
| libcusolver-12-4 | 11.6.1.9-1 | NVIDIA CUDA EULA |
| libcusparse-12-4 | 12.3.1.170-1 | NVIDIA CUDA EULA |
| libnccl2 | 2.21.5-1+cuda12.4 | BSD-3-Clause (NCCL is BSD; bundled by NVIDIA) |
| libnvjitlink-12-4 | 12.4.127-1 | NVIDIA CUDA EULA |

### 5b. NVIDIA libraries shipped in the `trt` flavor

The trt flavor uses the slimmer `cuda:12.4.1-base` and adds cuDNN +
the TensorRT runtime explicitly. ABI-pinned to TRT 10.1.0.27 + CUDA
12.4 (engines compiled against this combination).

| Component | Version (audit) | Governing license |
|---|---|---|
| cuda-compat-12-4 | 550.54.15-1 | NVIDIA CUDA EULA |
| cuda-cudart-12-4 | 12.4.127-1 | NVIDIA CUDA EULA |
| libcudnn9-cuda-12 | 9.21.1.3-1 | NVIDIA cuDNN Software License Agreement |
| libnvinfer10 | 10.1.0.27-1+cuda12.4 | NVIDIA TensorRT Software License Agreement |
| libnvinfer-plugin10 | 10.1.0.27-1+cuda12.4 | NVIDIA TensorRT SLA |
| libnvinfer-vc-plugin10 | 10.1.0.27-1+cuda12.4 | NVIDIA TensorRT SLA |
| libnvonnxparsers10 | 10.1.0.27-1+cuda12.4 | NVIDIA TensorRT SLA |
| python3-libnvinfer | 10.1.0.27-1+cuda12.4 | NVIDIA TensorRT SLA |

### NVIDIA EULA / SLA highlights

NVIDIA's CUDA EULA, cuDNN SLA, and TensorRT SLA share these
redistribution terms:

- **Permitted**: redistribution of the runtime libraries inside
  container images that run on end-user GPU hardware.
- **Forbidden**: reverse-engineering, modification, removing or
  obscuring proprietary notices.
- **Required**: keep the EULA text accessible alongside the binaries.
  The NGC base images ship `/NGC-DL-CONTAINER-LICENSE` at root; we
  leave it in place. Pointers to canonical NVIDIA SLA URLs are at
  [`third-party-licenses/nvidia-cuda-eula.txt`](./third-party-licenses/nvidia-cuda-eula.txt),
  [`third-party-licenses/nvidia-cudnn-sla.txt`](./third-party-licenses/nvidia-cudnn-sla.txt), and
  [`third-party-licenses/nvidia-tensorrt-sla.txt`](./third-party-licenses/nvidia-tensorrt-sla.txt).

Canonical URLs:
- CUDA EULA: https://docs.nvidia.com/cuda/eula/index.html
- cuDNN SLA: https://docs.nvidia.com/deeplearning/cudnn/sla/index.html
- TensorRT SLA: https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html

---

## 6. RunPod provider runtime (shipped in `providers/runpod/Dockerfile`)

The runpod-flavored images add the RunPod Python SDK on top of the
base flavor. The SDK pulls a deep transitive tree (FastAPI, boto3,
paramiko, sentry-sdk, etc.). Direct deps:

| Package | Version | License |
|---|---|---|
| runpod | 1.7.4 | MIT |
| pydantic | 2.9.2 | MIT |
| requests | 2.32.3 | Apache 2.0 |

Notable transitive deps (non-permissive or worth flagging):

| Package | Version | License | Notes |
|---|---|---|---|
| paramiko | 4.0.0 | **LGPL-2.1** | Direct dep of `runpod`. See compliance note below. |
| certifi | 2026.4.22 | **MPL-2.0** | CA bundle, not modified by us. License-text retention only. |
| tqdm | 4.67.3 | **MPL-2.0 AND MIT** | Progress bar lib. Same compliance posture as certifi. |
| cryptography | 43.0.3 | Apache-2.0 OR BSD-3-Clause | Permissive |
| boto3 / botocore / s3transfer | 1.42.97 / 1.42.97 / 0.16.1 | Apache-2.0 | Permissive |
| fastapi / starlette / uvicorn / httpx | various | MIT / BSD-3 / BSD-3 / BSD-3 | Permissive |
| sentry-sdk | 2.58.0 | MIT | Permissive |

The remaining ~80 transitive packages all carry permissive licenses
(MIT, BSD-2/3, Apache-2.0, ISC, HPND, PSF, Unlicense). A spot check
against the live image is the canonical refresh — see Maintenance.

### paramiko (LGPL-2.1) compliance

paramiko is a direct hard dependency of `runpod==1.7.4` and cannot
be removed without replacing the SDK. LGPL-2.1's obligations,
satisfied by our distribution model:

1. **Source availability**: paramiko's source is on GitHub
   (https://github.com/paramiko/paramiko) and PyPI; LGPL-2.1
   §6(a)/(b) is satisfied by the upstream's continuous availability.
   The text + URL is preserved at
   [`third-party-licenses/paramiko-LGPL-2.1.txt`](./third-party-licenses/paramiko-LGPL-2.1.txt).
2. **Library replaceability**: paramiko ships as a Python package
   at `/usr/local/lib/python3.10/dist-packages/paramiko/` —
   recipients can replace it with any LGPL-2.1-compatible build by
   overwriting that directory in the running container.
3. **License notice retention**: shipped via the third-party-licenses
   file noted above.
4. **No modification**: we don't patch paramiko; we install the
   upstream wheel verbatim.

### MPL-2.0 compliance (certifi, tqdm)

We don't modify either package. MPL-2.0 §3.1's "Source Code"
obligation is satisfied by upstream availability:

- certifi: https://github.com/certifi/python-certifi
- tqdm: https://github.com/tqdm/tqdm

If you fork this repo and modify either package, you'd need to
publish your modified source under MPL-2.0 in turn.

---

## 7. Tooling we invoke at release time (not redistributed)

- GitHub Actions workflows in `.github/workflows/` use Apache 2.0
  `actions/checkout`, `actions/setup-python`, `actions/upload-artifact`,
  `actions/download-artifact`. None of these ship in published
  artefacts.
- `gh` CLI used in the workflow is MIT.
- `build/.with-iosuite-key` is a maintainer-local wrapper for
  loading `RUNPOD_API_KEY`; it ships in the repo but not in any image.

---

## Distribution checklist

Before publishing a release or container image, verify:

- [ ] `models/MANIFEST.json` lists the correct license text URL for
      each model artefact (the `license_url` field) — for both
      `.onnx` and `.engine` variants.
- [ ] This `NOTICE.md` is included in any source tarball or container
      image at a discoverable path (each runtime image copies it to
      `/usr/share/doc/real-esrgan-serve/`).
- [ ] The repo's own `LICENSE` (Apache-2.0) is preserved alongside
      NOTICE.
- [ ] The upstream Real-ESRGAN `LICENSE` text is preserved at
      [`third-party-licenses/real-esrgan.txt`](./third-party-licenses/real-esrgan.txt)
      — required by BSD-3-Clause.
- [ ] For the `cuda` and `trt` flavors: NVIDIA's `/NGC-DL-CONTAINER-LICENSE`
      (inherited from the NGC base image) is still at root in the
      shipped image. Don't `rm` it during image stripping.
- [ ] For the `trt` flavor: the TensorRT SLA pointer file at
      `/usr/share/doc/real-esrgan-serve/third-party-licenses/nvidia-tensorrt-sla.txt`
      is present.
- [ ] For the runpod variants: paramiko's LGPL-2.1 text + source
      pointer at `/usr/share/doc/real-esrgan-serve/third-party-licenses/paramiko-LGPL-2.1.txt`
      is present.
- [ ] If you have replaced the model weights with a different model,
      update the `license` and `license_url` fields in
      `models/MANIFEST.json` and add a corresponding entry to this
      file.

---

## Maintenance

When bumping any dependency, re-validate that:

1. The new version's license is still permissive (MIT / BSD-x /
   Apache 2.0 / HPND / PSF / Python-2.0 / ISC / Unlicense), MPL-2.0
   (compliance via upstream source pointer), or LGPL-2.1+ via
   replaceable dynamic linkage. **Do not introduce GPL, AGPL, or
   proprietary-licensed dependencies (other than the NVIDIA
   components catalogued in Section 5) without documenting the
   implications here first.**
2. If the dependency is redistributed in a runtime image
   (Sections 3, 5, 6), update the version in this document.
3. If the dependency carries a NOTICE file (Apache 2.0 §4(d)), the
   NOTICE text is preserved in the runtime image (typically at
   `/usr/share/doc/<package>/NOTICE`).

A live refresh of pip-installed licenses against any runtime image:

```bash
docker run --rm --entrypoint /bin/bash real-esrgan-serve:runpod-cuda-dev -c '
  cd /usr/local/lib/python3.10/dist-packages
  for d in */; do
    meta=$(find "$d" -maxdepth 2 -name METADATA 2>/dev/null | head -1)
    [ -z "$meta" ] && continue
    pkg=$(echo "$d" | sed "s|/$||")
    expr=$(grep -m1 "^License-Expression:" "$meta" 2>/dev/null | sed "s/^License-Expression: //")
    lic=$(grep -m1 "^License:" "$meta" 2>/dev/null | sed "s/^License: //")
    classifier=$(grep "^Classifier: License" "$meta" 2>/dev/null | head -1 | sed "s/^Classifier: License :: //")
    printf "%-40s %s\n" "$pkg" "${expr:-${classifier:-${lic:-(unknown)}}}"
  done | sort -u
'
```

A live refresh of apt-installed packages of interest:

```bash
docker run --rm --entrypoint /bin/bash real-esrgan-serve:trt-dev -c \
  "dpkg -l | grep -iE 'cuda|cudnn|nvinfer' | awk '{print \$2,\$3}'"
```

Any output of `License: GPL` / `License: AGPL` / `License: UNKNOWN`
(except setuptools, which is MIT-licensed but ships the bare
copyright header that triggers the UNKNOWN classifier) warrants
investigation before the image is published.
