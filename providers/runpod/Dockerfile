# providers/runpod/Dockerfile — RunPod serverless image for
# real-esrgan-serve. Layers on top of the base runtime image, adds
# RunPod's Python SDK and the platform-specific handler.
#
# Model artefacts (.onnx, .engine) are NOT baked in. handler.py runs
# `real-esrgan-serve fetch-model` at module-import time (before the
# runpod SDK starts polling the job queue), downloads the right
# variant for the requested EP and the host GPU's SM arch, and
# caches under $XDG_CACHE_HOME/real-esrgan-serve/models for the
# rest of the worker's lifetime. Trade-off: +5–30 s on first cold
# start in exchange for an image that's portable across providers,
# decoupled from model version, and lets one image serve cpu/cuda/
# trt workloads via REAL_ESRGAN_PROVIDER env var.
#
# Build (from repo root):
#   make docker docker-runpod
ARG BASE_TAG=dev
FROM real-esrgan-serve:${BASE_TAG}

# RunPod-specific Python deps. Pin the SDK so a runtime upgrade
# doesn't surprise an in-flight deployment.
RUN pip3 install --no-cache-dir \
        "runpod==1.7.4" \
        "pydantic==2.9.2" \
        "requests==2.32.3" \
    && rm -rf /root/.cache /tmp/*

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
