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

## Limitations & VRAM

Because this tool relies on the `realesrgan-x4plus` model processing via TensorRT, it holds the following constraints:

1. **Memory Ceiling**: All TensorRT context memory linearly correlates with the input image's dimensions. Since the `realesrgan-x4plus` model outputs are exactly 4x the input size in width and height (yielding a $16\times$ larger pixel map overall), large input files (e.g., $1920\times 1080$ and above) will cause drastic spikes in VRAM usage during the convolution and upscaling execution layers. Expect an Out Of Memory (OOM) exception on smaller GPUs for large input imagery.
2. **Dimension Tiling**: While other implementations fallback to patching or "tiling" to solve VRAM exhaustions, this pure C++ backend expects the entire activation map locally. Future updates may introduce chunking for massive geometries.
3. **Optimized Engine Size**: The compiled `.engine` profile must be specifically generated for the target shape you intend to run (with min/opt/max bounds defined). Images that far exceed your engine's `.max` profile dimensions will fail to enqueue.
