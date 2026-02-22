FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# Install system dependencies required by OpenCV and Basicsr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the official Real-ESRGAN repo for the pytorch2onnx script
RUN git clone https://github.com/xinntao/Real-ESRGAN.git .

# Checkout the specific commit that works
RUN git checkout a4abfb2979a7bbff3f69f58f58ae324608821e27

# Use uv to install dependencies with time-travel resolution
RUN uv venv && \
    uv pip install --exclude-newer 2022-09-04T23:59:59Z -r requirements.txt

# Download the official RealESRGAN_x4plus weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights

# Create a volume mount point for the output
VOLUME /output

LABEL org.opencontainers.image.source="https://github.com/ls-ads/real-esrgan-serve"
LABEL org.opencontainers.image.description="Export RealESRGAN_x4plus.pth to ONNX format"
LABEL org.opencontainers.image.title="real-esrgan-serve/onnx-export"

# Copy custom dynamic exporter
COPY onnx_export.py .

# Set the entrypoint to run the conversion script and output to the mounted volume
ENTRYPOINT [".venv/bin/python", "onnx_export.py", "--input", "weights/RealESRGAN_x4plus.pth", "--output", "/output/realesrgan-x4plus.onnx", "--params"]
