# providers/runpod/Dockerfile — RunPod serverless image for
# real-esrgan-serve. Layers on top of the base runtime image, adds
# RunPod's Python SDK, the platform-specific handler, and bakes in
# the .onnx model so cold starts don't pay network cost.
#
# Cold start is the bottleneck on RunPod serverless. The image-pull
# itself is the dominant cost (mitigated by the slim base — see the
# parent Dockerfile). First-request latency is small because
# onnxruntime's CUDA EP loads the .onnx in <1 s.
#
# Build (from repo root):
#   make docker docker-runpod
# Layout produced:
#   /workspace/realesrgan-x4plus_fp16.onnx
#
# Once an upstream release exists at
# https://github.com/ls-ads/real-esrgan-serve/releases/download/v*/...
# this Dockerfile can switch to `real-esrgan-serve fetch-model`
# instead of COPY. For the initial bring-up we use COPY from the
# locally-built build/dist/ — see build/README.md for how to
# produce those.
ARG BASE_TAG=dev
FROM real-esrgan-serve:${BASE_TAG}

# RunPod-specific Python deps. Pin the SDK so a runtime upgrade
# doesn't surprise an in-flight deployment.
RUN pip3 install --no-cache-dir \
        "runpod==1.7.4" \
        "pydantic==2.9.2" \
        "requests==2.32.3" \
    && rm -rf /root/.cache /tmp/*

# Bake the .onnx model into the image. Tiny relative to ORT's wheel
# (~33 MB FP16), and saves the worker a network fetch on first
# request. handler.py reads from $REAL_ESRGAN_MODEL or
# /workspace/realesrgan-x4plus_fp16.onnx, falling back to
# `real-esrgan-serve fetch-model` if neither resolves.
COPY build/dist/realesrgan-x4plus_fp16.onnx /workspace/realesrgan-x4plus_fp16.onnx

WORKDIR /app
COPY providers/runpod/handler.py /app/handler.py

# Override the base image's ENTRYPOINT (real-esrgan-serve CLI). On
# RunPod the runpod-python SDK is the entrypoint — it polls the
# RunPod queue and shells out to our Python helper directly via
# WarmHelper, bypassing the Go CLI. Leaving ENTRYPOINT inherited
# turns "python3 -u /app/handler.py" into a positional arg for
# the CLI, which Cobra rejects with
#   Error: unknown command "python3" for "real-esrgan-serve"
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
