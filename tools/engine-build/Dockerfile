FROM ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0

# Install wget
RUN apt-get update && apt-get install -y wget

# Remove CLI entrypoint so we can run sequential scripts
ENTRYPOINT []

WORKDIR /workspace

# Download the pre-exported compiled ONNX graph from the official GitHub Release
# (Saves users from having to build the python extractor locally)
RUN wget https://github.com/ls-ads/real-esrgan-serve/releases/download/v0.1.0/realesrgan-x4plus.onnx -O /workspace/realesrgan-x4plus.onnx

# Copy the customized build script
COPY build.sh /workspace/build.sh
RUN chmod +x /workspace/build.sh

# When the container spins up on a remote cloud GPU architecture,
# it executes build.sh to dynamically detect the GPU architecture & TRT version,
# and generates a specific '.engine' file directly into the mounted /output directory.
CMD ["/workspace/build.sh"]
