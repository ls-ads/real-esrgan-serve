# real-esrgan-serve runtime image — license-clean + cold-start-optimised.
#
# COLD-START IS THE PRIMARY OPTIMISATION TARGET HERE.
# RunPod (and other serverless GPU providers) pull the full image to
# every node on every cold start. A 4 GB image over a typical
# multi-GB/s registry pull is still ~10–30 s of pure waiting before
# the model has even been touched. The previous shape sat around
# ~4 GB; this one targets <1 GB.
#
# Layer ordering matters: slow-changing layers (apt, pip) come first
# so RunPod's image puller resumes partial pulls cleanly when only
# the final layers (Go binary, Python script) change between
# releases. NEVER reshuffle this without re-checking the published
# image-layer sizes.
#
# Sizing breakdown (approximate, ubuntu22.04 + cuda 12.4 base):
#   nvidia/cuda:base            ~150 MB   (libcuda.so.1 + nvidia-smi)
#   python3 + pip               ~ 80 MB
#   numpy + pillow (pip wheels) ~ 30 MB
#   onnxruntime-gpu 1.18 wheel  ~600 MB   (CUDA EP libs)
#   Go binary (CGO=0, stripped) ~  7 MB
#   runtime/upscaler.py         < 1 MB
#   ───────────────────────────────────
#                       Total   ~870 MB
#
# We run ONNX-only via ORT's CUDAExecutionProvider — no TensorRT,
# no cuDNN, no per-GPU engine compilation. Earlier iterations
# explored a TRT-EP path for ~5x warm-exec speedup, but enabling it
# requires libcudnn (~1 GB on its own — cuDNN 9 is split into
# per-arch precompiled kernel libraries) and lifts the image past
# 3.5 GB, more than doubling cold-start image-pull time. The
# trade-off doesn't pay back at our throughput; if it ever does,
# it's a contained re-enable (cuDNN install + ORT bump + restore
# the engine COPY in providers/runpod/Dockerfile).
#
# Why pre-baked model on top of an `runtime` layer: the .onnx
# (~33 MB FP16 / ~67 MB FP32) is pre-fetched in the provider-
# specific Dockerfile (e.g. providers/runpod/Dockerfile), NOT here,
# so the base image stays useful for non-RunPod consumers who fetch
# their own weights. Provider images add the model on top, which is
# the right caching shape.
#
# Build:
#   make docker
#   # equivalently: docker build -t real-esrgan-serve:dev .
# Run (one-shot upscale):
#   docker run --rm --gpus all \
#     -v $PWD/imgs:/work \
#     real-esrgan-serve:dev upscale -i /work/in.jpg -o /work/out.jpg \
#     --model realesrgan-x4plus
# ─────────────────────────────────────────────────────────────────────

# --- Stage 1: build the Go binary ───────────────────────────────────
FROM golang:1.25-alpine AS gobuild
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/      ./cmd/
COPY internal/ ./internal/
COPY models/   ./models/
ARG VERSION=dev
# CGO=0 + -s -w + trimpath = smallest, fully static, reproducible binary.
RUN CGO_ENABLED=0 go build \
        -trimpath \
        -ldflags "-s -w -X main.version=${VERSION}" \
        -o /out/real-esrgan-serve \
        ./cmd/real-esrgan-serve

# --- Stage 2: runtime (slim) ────────────────────────────────────────
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Minimal Python install. We drop python3-numpy and python3-pil
# from apt (saves ~50 MB vs the previous shape) — both come in via
# pip wheels below. apt's Pillow also links a wider set of system
# codecs we don't use, so the pip wheel is leaner.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# onnxruntime-gpu provides the CUDA EP we use for inference. Pillow
# + numpy are pulled in explicitly because we dropped python3-numpy
# /pil from apt above; --no-deps keeps the layer from picking up
# unused GPU libs (e.g. TensorRT's ~2 GB pip wheel).
RUN pip3 install --no-cache-dir --no-deps \
        onnxruntime-gpu==1.18.1 \
        numpy==1.26.4 \
        Pillow==10.4.0 \
    && rm -rf /root/.cache /tmp/*

# Late-arriving artefacts (small, change every build) ────────────────
COPY --from=gobuild /out/real-esrgan-serve /usr/local/bin/real-esrgan-serve
COPY runtime/upscaler.py /usr/share/real-esrgan-serve/runtime/upscaler.py
RUN chmod +x /usr/share/real-esrgan-serve/runtime/upscaler.py

# Ship licensing notices alongside the binary so redistribution
# (Docker pull → run) inherently carries the attributions BSD-3-Clause
# and Apache 2.0 require. Standard FHS path; surface them via
# `docker run --rm <image> cat /usr/share/doc/real-esrgan-serve/NOTICE.md`.
COPY LICENSE                       /usr/share/doc/real-esrgan-serve/LICENSE
COPY NOTICE.md                     /usr/share/doc/real-esrgan-serve/NOTICE.md
COPY third-party-licenses/         /usr/share/doc/real-esrgan-serve/third-party-licenses/

# Cache dirs. /var/cache/real-esrgan-serve/ holds fetched .onnx
# weights. Mount a persistent volume here in providers (RunPod
# Network Volume, K8s PVC, etc.) so an out-of-image fetch survives
# container restarts.
ENV XDG_CACHE_HOME=/var/cache \
    REAL_ESRGAN_RUNTIME=/usr/share/real-esrgan-serve/runtime/upscaler.py

ENTRYPOINT ["real-esrgan-serve"]
CMD ["--help"]
