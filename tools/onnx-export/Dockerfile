FROM python:3.10-slim

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

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the official RealESRGAN_x4plus weights
RUN mkdir -p experiments/pretrained_models && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O experiments/pretrained_models/RealESRGAN_x4plus.pth

# Create a volume mount point for the output
VOLUME /output

# Set the entrypoint to run the conversion script and output to the mounted volume
ENTRYPOINT ["python", "scripts/pytorch2onnx.py", "--input", "experiments/pretrained_models/RealESRGAN_x4plus.pth", "--output", "/output/realesrgan-x4plus.onnx"]
