# Real-ESRGAN Serve

A standalone Go CLI tool using Cobra that bridges to a TensorRT (C/C++) implementation of Real-ESRGAN. Optimized specifically for the `realesrgan-x4plus` model.

This tool is designed to support both "one-shot" local inference and a persistent HTTP server.

## Installation Requirements

Since this project links directly to the TensorRT C++ API, you must have the required system dependencies installed:
- Go 1.20+
- CUDA Toolkit (`/usr/local/cuda`)
- TensorRT (`/usr/lib/x86_64-linux-gnu`)

Build the CLI using the provided `Makefile`:

```bash
make build
```

This will output the `real-esrgan-serve` binary.

## Usage

### 1. Local Inference (File or Directory)

You can run local inference on a single image file or an entire directory of images. The CLI natively iterates and processes all images within the directory if one is provided. Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`.

```bash
# Process a single image
./real-esrgan-serve -i path/to/image.jpg -o path/to/output.png

# Process a directory of images
./real-esrgan-serve -i path/to/input_dir -o path/to/output_dir
```

- `-i`, `--input`: The input file path or directory path.
- `-o`, `--output`: The output file path or directory path. If omitted, the tool automatically appends `_out` to the input name.
- `-g`, `--gpu-id`: (Optional) The GPU device ID to use (default: 0).

*Note: The upscale ratio is fixed to 4x, as this is optimized for `realesrgan-x4plus`.*

### 2. HTTP Server

To avoid the overhead of loading the model into VRAM for every image, you can start a persistent HTTP server. 

```bash
# Start the server on port 8080 (default) mapping to a specific engine
./real-esrgan-serve server start -p 8080 --engine path/to/model.engine --gpu-id 0

# Stop the server via PID file
./real-esrgan-serve server stop
```

### 3. Builder

You can compile an ONNX model into a hardware-specific TensorRT `.engine` file via the build command:

```bash
./real-esrgan-serve build --onnx path/to/model.onnx --engine path/to/model.engine
```

## Generating the ONNX Model

The `.engine` builder built into this tool is configured to dynamically accept input dimensions scaling from `64x64` up to `1024x1024`. It strips out the fixed image constraints baked into standard ONNX trace graphs via TensorRT `IOptimizationProfile`.

You must generate the ONNX file (`realesrgan-x4.onnx`) from the official `Real-ESRGAN_x4plus.pth` PyTorch weights before building your engine.

Because the official extraction script relies on PyTorch and OpenCV (which requires specific `C` libraries like `libgl1` and `libxcb`), we have provided an isolated Dockerfile to generate it reproducibly without clashing with your host system.

1. Build the exporter image:
```bash
docker build -t realesrgan-onnx-exporter tools/onnx-export/
```

2. Run the container, mounting your current directory to extract the output `.onnx` file:
```bash
docker run --rm -v $(pwd):/output realesrgan-onnx-exporter
```

This will automatically download the official `.pth` model, execute the trace, and save `realesrgan-x4plus.onnx` into your current directory!

## Limitations & VRAM

Because this tool relies on the `realesrgan-x4plus` model processing via TensorRT, it holds the following constraints:

1. **Memory Ceiling**: All TensorRT context memory linearly correlates with the input image's dimensions. Since the `realesrgan-x4plus` model outputs are exactly 4x the input size in width and height (yielding a $16\times$ larger pixel map overall), large input files (e.g., $1920\times 1080$ and above) will cause drastic spikes in VRAM usage during the convolution and upscaling execution layers. Expect an Out Of Memory (OOM) exception on smaller GPUs for large input imagery.
2. **Dimension Tiling**: While other implementations fallback to patching or "tiling" to solve VRAM exhaustions, this pure C++ backend expects the entire activation map locally. Future updates may introduce chunking for massive geometries.
3. **Optimized Engine Size**: The compiled `.engine` profile must be specifically generated for the target shape you intend to run (with min/opt/max bounds defined). Images that far exceed your engine's `.max` profile dimensions will fail to enqueue.
