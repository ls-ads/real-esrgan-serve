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

You must generate the ONNX file (`realesrgan-x4.onnx`) from the official `Real-ESRGAN_x4plus.pth` PyTorch weights before building your engine. 

> [!TIP]
> **Pre-exported Model Available**: We have already exported the standard `realesrgan-x4plus` model to ONNX for you. You can download the verified [realesrgan-x4plus.onnx](https://github.com/ls-ads/real-esrgan-serve/releases/tag/v0.1.0) directly from the GitHub releases page.

Because the official extraction script relies on PyTorch and OpenCV (which requires specific `C` libraries like `libgl1` and `libglib2.0-0`), we have provided an isolated Dockerfile to generate it reproducibly without clashing with your host system.

1. Pull or build the exporter image:
```bash
docker pull ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0
# Alternatively, build locally: docker build -t ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0 tools/onnx-export/
```

2. Run the container, mounting your current directory to extract the output `.onnx` file:
```bash
docker run --rm -v $(pwd):/output ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0
```

This will automatically download the official `.pth` model, execute the trace, and save `realesrgan-x4plus.onnx` into your current directory!

### Verification
To ensure the mathematical graph sequence was exported flawlessly without any hardware translation discrepancies, verify the MD5 checksum of the generated `.onnx` file:
```bash
$ md5sum realesrgan-x4plus.onnx 
6216c6b3d76719b0732fc49bb0f6c879  realesrgan-x4plus.onnx
```

## Limitations & VRAM

Because this tool relies on the `realesrgan-x4plus` model processing via TensorRT, it holds the following constraints:

1. **Memory Ceiling**: All TensorRT context memory linearly correlates with the input image's dimensions. Since the `realesrgan-x4plus` model outputs are exactly 4x the input size in width and height (yielding a $16\times$ larger pixel map overall), large input files (e.g., $1920\times 1080$ and above) will cause drastic spikes in VRAM usage during the convolution and upscaling execution layers. Expect an Out Of Memory (OOM) exception on smaller GPUs for large input imagery.
2. **Dimension Tiling**: While other implementations fallback to patching or "tiling" to solve VRAM exhaustions, this pure C++ backend expects the entire activation map locally. Future updates may introduce chunking for massive geometries.
## Acknowledgements

This project is a bridge to the TensorRT implementation of Real-ESRGAN. We would like to give full credit to the original authors and the official project:

- **Original Project**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Author**: [xinntao](https://github.com/xinntao)

The `realesrgan-x4plus` model and the underlying architecture are products of the research and development by the original authors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
