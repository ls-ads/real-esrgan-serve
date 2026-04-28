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
# Sizing breakdown (approximate, ubuntu22.04 + cuda 12.4):
#   nvidia/cuda:base            ~150 MB   (libcuda.so.1 + nvidia-smi)
#   python3 + pip + apt-cleaned ~100 MB
#   numpy + pillow (apt)        ~ 50 MB
#   onnxruntime-gpu wheel       ~500 MB   (bundles cuDNN, cuBLAS, etc.)
#   Go binary (CGO=0, stripped) ~ 10 MB
#   runtime/upscaler.py         < 1 MB
#   ───────────────────────────────────
#                       Total   ~810 MB
#
# Why nvidia/cuda:base and NOT :runtime: the :runtime variant ships
# the full CUDA runtime (~3.5 GB worth of libs the wheel doesn't
# need). onnxruntime-gpu's pip wheel bundles its own copies of
# cuDNN + cuBLAS; we only need libcuda.so.1 from the host (provided
# by nvidia-container-toolkit at runtime) plus a working python.
# Validated against onnxruntime-gpu 1.18.x; if a future ORT release
# starts demanding system CUDA libs, fall back to :runtime here.
#
# Why pre-baked model on top of an `runtime` layer: Stage A's .onnx
# (~67 MB FP16 / ~134 MB FP32) is pre-fetched in the provider-
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

# Minimal Python + image I/O. apt --no-install-recommends drops
# ~80 MB of optional Suggests/Recommends. Cleaning apt caches and
# tmp in the SAME RUN is what keeps the layer thin (a separate `rm`
# layer would still carry the cache bytes).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-numpy \
        python3-pil \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# onnxruntime-gpu pinned to a CUDA-12-compatible release. The wheel
# bundles cuDNN + the TensorRT EP; first-use compilation produces a
# cached engine under XDG_CACHE_HOME (mount a volume there in
# production to survive container churn).
#
# `--no-deps --no-cache-dir` is deliberate: numpy is already
# installed via apt, and pip's cache adds ~200 MB to the layer if
# left enabled.
RUN pip3 install --no-cache-dir --no-deps onnxruntime-gpu==1.18.1 \
    && rm -rf /root/.cache /tmp/*

# Late-arriving artefacts (small, change every build) ────────────────
COPY --from=gobuild /out/real-esrgan-serve /usr/local/bin/real-esrgan-serve
COPY runtime/upscaler.py /usr/share/real-esrgan-serve/runtime/upscaler.py
RUN chmod +x /usr/share/real-esrgan-serve/runtime/upscaler.py

# Cache dirs. /var/cache/real-esrgan-serve/ holds fetched .onnx +
# TRT engine cache. Mount a persistent volume here in providers
# (RunPod Network Volume, K8s PVC, etc.) so engine compilation is
# paid once per GPU class and survives container restarts.
ENV XDG_CACHE_HOME=/var/cache \
    REAL_ESRGAN_RUNTIME=/usr/share/real-esrgan-serve/runtime/upscaler.py

ENTRYPOINT ["real-esrgan-serve"]
CMD ["--help"]
