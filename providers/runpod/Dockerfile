# providers/runpod/Dockerfile — RunPod serverless image for
# real-esrgan-serve. Layers on top of the base runtime image, adds
# RunPod's Python SDK, the platform-specific handler, and bakes in
# the model artefacts so cold starts don't pay network cost.
#
# Cold start is the bottleneck on RunPod serverless. The image-pull
# itself is one cost (mitigated by the slim base — see the parent
# Dockerfile). The OTHER cost is "first request latency" — by the
# time RunPod has spun up a worker, we still have to:
#   1) load the .onnx into onnxruntime
#   2) build the TensorRT engine (~30s on first request)
#   3) serve the request
# Pre-baking a per-GPU-class .engine into the image collapses (2)
# from 30s to ~0.5s. .onnx is the fallback for GPUs we don't have
# a pre-baked engine for.
#
# Build (from repo root):
#   make docker docker-runpod
# Layout produced:
#   /workspace/realesrgan-x4plus_fp16.onnx                    — fallback
#   /workspace/realesrgan-x4plus-rtx-4090-sm89-trt10.16_fp16.engine  — fast path
#
# Once an upstream release exists at
# https://github.com/ls-ads/real-esrgan-serve/releases/download/v*/...
# this Dockerfile can switch to `real-esrgan-serve fetch-model`
# instead of COPY. For the initial bring-up we use COPY from the
# locally-built build/dist/ — see build/README.md for how to
# produce those (Stage A locally + Stage B remote on RunPod).
ARG BASE_TAG=dev
FROM real-esrgan-serve:${BASE_TAG}

# RunPod-specific Python deps. Pin the SDK so a runtime upgrade
# doesn't surprise an in-flight deployment.
RUN pip3 install --no-cache-dir \
        "runpod==1.7.4" \
        "pydantic==2.9.2" \
        "requests==2.32.3" \
    && rm -rf /root/.cache /tmp/*

# Bake the model. Engine artefacts are intentionally NOT copied
# here — the base image ships without cuDNN to keep cold-start
# image-pull time down (see ../../Dockerfile header). Without
# cuDNN, ORT's TRT EP can't load and a baked .engine file is
# unparseable as ONNX. handler.py's _trt_ep_loadable() probe
# guards this and falls through to the .onnx via CUDA EP.
# To re-enable engines: install libcudnn9-cuda-12 in the base
# Dockerfile and bump onnxruntime-gpu to 1.20.x, then re-add
#   COPY build/dist/*.engine /workspace/
# Trade-off documented in ../../Dockerfile header.
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
