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
# Why :base + no cuDNN: enabling ORT's TensorRT EP would let us
# consume pre-compiled .engine artefacts (~30 s → ~0.5 s first
# request, ~5x warm), but TRT EP needs libcudnn on the linker path
# AND a matching TensorRT version, AND ORT compiled against both.
# Concretely, that means upgrading to ORT 1.20.x (cuDNN 9), and
# apt-installing libcudnn9-cuda-12 (~1 GB by itself — cuDNN 9 is
# split into libcudnn_engines_precompiled.so etc that bundle CUDA
# kernels for every supported SM arch). The resulting image lands
# at ~3.8 GB, more than doubling cold-start image-pull time. Until
# we can amortize that cost (high-throughput workload, or RunPod's
# warm-pool feature), we ship CUDA-EP-only — slower per-request
# but a leaner cold start. Engine artefacts in the provider
# Dockerfile are gated by handler.py's _trt_ep_loadable() probe.
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

# onnxruntime-gpu provides the CUDA EP we use here. The TRT EP is
# also bundled but won't load without libcudnn — see the header
# comment for why we don't ship cuDNN. Pillow + numpy are pulled in
# explicitly because we dropped python3-numpy/pil from apt above;
# --no-deps keeps the layer from accidentally picking up TensorRT's
# pip wheel (~2 GB) or other system-provided GPU libs.
RUN pip3 install --no-cache-dir --no-deps \
        onnxruntime-gpu==1.18.1 \
        numpy==1.26.4 \
        Pillow==10.4.0 \
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
