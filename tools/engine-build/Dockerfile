FROM ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0

# Install wget
RUN apt-get update && apt-get install -y wget

WORKDIR /app

# Download the pre-exported compiled ONNX graphs from the official GitHub Release
# (Saves users from having to build the python extractor locally)
RUN wget https://github.com/ls-ads/real-esrgan-serve/releases/download/v0.1.0/realesrgan-x4plus_fp16.onnx -O /app/realesrgan-x4plus_fp16.onnx && \
    wget https://github.com/ls-ads/real-esrgan-serve/releases/download/v0.1.0/realesrgan-x4plus_fp32.onnx -O /app/realesrgan-x4plus_fp32.onnx

# Copy the customized build script
COPY build.sh /app/build.sh
RUN chmod +x /app/build.sh

# Create a volume mount point for the output
VOLUME /output

LABEL org.opencontainers.image.source="https://github.com/ls-ads/real-esrgan-serve"
LABEL org.opencontainers.image.description="Build TensorRT engines for Real-ESRGAN"
LABEL org.opencontainers.image.title="real-esrgan-serve/engine-build"

# Remove CLI entrypoint so we can run sequential scripts
ENTRYPOINT []

# When the container spins up on a remote cloud GPU architecture,
# it executes build.sh to dynamically detect the GPU architecture & TRT version,
# and generates a specific '.engine' file directly into the mounted /output directory.
CMD ["/app/build.sh"]
