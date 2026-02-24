#!/bin/bash
set -e

echo "Starting Real-ESRGAN Engine Builder..."

# 1. Detect TensorRT Version
# The NVIDIA TensorRT container usually has libnvinfer installed
TRT_FULL=$(dpkg-query -W -f='${Version}' libnvinfer-bin 2>/dev/null || dpkg -l | awk '/^ii  libnvinfer[0-9]+/ {print $3}' | head -n 1)
TRT_VER=$(echo "$TRT_FULL" | cut -d. -f1,2)

if [ -z "$TRT_VER" ]; then
    TRT_VER="unknown"
    echo "Warning: Could not detect TensorRT version."
else
    echo "Detected TensorRT Version: $TRT_VER"
fi

# 2. Detect Native GPU Architecture
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found or no GPU attached."
    echo "This builder container must be run on a machine with a physical NVIDIA GPU."
    exit 1
fi

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
ARCH="sm${COMPUTE_CAP//./}"
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1)
# Sanitize GPU name: lowercase, replace spaces/special chars with hyphens, remove redundant hyphens
GPU_NAME_SAN=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\{1,\}/-/g' | sed 's/^-//;s/-$//')

echo "Detected GPU: $GPU_NAME ($ARCH)"

# 3. Build the Engines

# 3.1 Build FP16 Engine
PRECISION="fp16"
SUFFIX="_fp16"
OUT_NAME="realesrgan-x4plus-${GPU_NAME_SAN}-${ARCH}-trt${TRT_VER}${SUFFIX}.engine"
OUT_PATH="/output/${OUT_NAME}"

echo "------------------------------------------------------"
echo "Compiling Engine ($PRECISION): $OUT_PATH"
echo "------------------------------------------------------"
/app/real-esrgan-serve build --onnx "/app/realesrgan-x4plus${SUFFIX}.onnx" --engine "$OUT_PATH" --fp16

# 3.2 Build FP32 Engine
PRECISION="fp32"
SUFFIX="_fp32"
OUT_NAME="realesrgan-x4plus-${GPU_NAME_SAN}-${ARCH}-trt${TRT_VER}${SUFFIX}.engine"
OUT_PATH="/output/${OUT_NAME}"

echo "------------------------------------------------------"
echo "Compiling Engine ($PRECISION): $OUT_PATH"
echo "------------------------------------------------------"
/app/real-esrgan-serve build --onnx "/app/realesrgan-x4plus${SUFFIX}.onnx" --engine "$OUT_PATH"

touch /output/DONE

echo "------------------------------------------------------"
echo "Engine compiled successfully!"
echo "Saved to: $OUT_PATH"
