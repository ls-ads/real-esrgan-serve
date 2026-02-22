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

echo "Detected GPU Architecture: $ARCH (Compute Capability: $COMPUTE_CAP)"

# 3. Build the Engine
OUT_NAME="realesrgan-x4plus-${ARCH}-trt${TRT_VER}.engine"
OUT_PATH="/output/${OUT_NAME}"

echo "------------------------------------------------------"
echo "Compiling Engine: $OUT_PATH"
echo "------------------------------------------------------"

/app/real-esrgan-serve build --onnx /workspace/realesrgan-x4plus.onnx --engine "$OUT_PATH"

echo "------------------------------------------------------"
echo "Engine compiled successfully!"
echo "Saved to: $OUT_PATH"
