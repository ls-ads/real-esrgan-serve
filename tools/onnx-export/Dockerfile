FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

# Install system dependencies required by OpenCV and Basicsr
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the official Real-ESRGAN repo for the pytorch2onnx script
RUN git clone https://github.com/xinntao/Real-ESRGAN.git .

# Use uv to install dependencies with time-travel resolution
RUN uv venv && \
    uv pip install --exclude-newer 2022-09-04T23:59:59Z -r requirements.txt

# Download the official RealESRGAN_x4plus weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights

# Create a volume mount point for the output
VOLUME /output

# Set the entrypoint to run the conversion script and output to the mounted volume
ENTRYPOINT [".venv/bin/python", "scripts/pytorch2onnx.py", "--input", "weights/RealESRGAN_x4plus.pth", "--output", "/output/realesrgan-x4plus.onnx", "--params"]
