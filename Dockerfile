# --- Stage 1: Build the Go application ---
FROM nvcr.io/nvidia/tensorrt:25.01-py3 AS builder

# Install Go and Make
RUN apt-get update && apt-get install -y wget make && \
    wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

ENV PATH=$PATH:/usr/local/go/bin
ENV CGO_ENABLED=1

# Set up workspace
WORKDIR /app

# Pre-copy/cache go.mod so we don't redownload dependencies every build
COPY go.mod go.sum ./
RUN go mod download

# Copy application source code
COPY . .

# Build the standalone binary using the Makefile
RUN make build

# --- Stage 2: Create a minimal runtime environment ---
# We still need the TensorRT and CUDA shared libraries (.so files) at runtime,
# so we use a runtime TensorRT image rather than a completely empty scratch container
FROM nvcr.io/nvidia/tensorrt:25.01-py3

# Set up working directory for the application
WORKDIR /app

# Copy only the compiled Go binary from the builder stage
COPY --from=builder /app/real-esrgan-serve /app/real-esrgan-serve

# Create an io directory for users to mount their input/output files into
RUN mkdir -p /workspace

# Expose the HTTP server port
EXPOSE 8080

# Add labels for container description
LABEL org.opencontainers.image.source="https://github.com/ls-ads/real-esrgan-serve"
LABEL org.opencontainers.image.description="A standalone Go CLI tool that bridges to a TensorRT (C/C++) implementation of Real-ESRGAN. Optimized specifically for the `realesrgan-x4plus` model."
LABEL org.opencontainers.image.title="real-esrgan-serve/cli"

# The default behavior logic can be controlled via Docker CMD/ENTRYPOINT overrides
ENTRYPOINT ["/app/real-esrgan-serve"]
