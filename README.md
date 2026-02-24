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

You can run local inference on a single image file or an entire directory of images. The CLI natively iterates and processes all images within the directory if one is provided. 

**Supported Formats:**
- **Inputs:** `.png`, `.jpg`, `.jpeg` (Max resolution: $1280 \times 1280$ pixels). *Any image exceeding these dimensions on either axis will be cleanly rejected.*
- **Outputs:** `.png` (lossless, slower to encode 5K images), `.jpg` / `.jpeg` (fast, high-quality compressed). *Any unsupported or missing output extension will automatically fallback to being encoded as a PNG.*
  - **Determinism:** If you process the exact same input image using the exact same TensorRT `.engine` file on the exact same physical GPU, the mathematical matrix multiplication path chosen by the compiled execution context is 100% deterministic. The resulting upscaled output image will have an identical byte structure and MD5 checksum across multiple runs, server restarts, and system reboots.

> [!TIP]
> **Performance Tip**: Exporting to `.jpg` is significantly faster than `.png`. Because the 4x upscaling creates massive 5K+ images, lossless PNG CPU-encoding can take over 5 seconds on the server. Specifying `.jpg` for your output file drops that CPU bottleneck to under a second!

```bash
# Process a single image
./real-esrgan-serve -i path/to/image.jpg -o path/to/output.png

# Process a directory of images
./real-esrgan-serve -i path/to/input_dir -o path/to/output_dir
```

- `-i`, `--input`: The input file path or directory path.
- `-o`, `--output`: The output file path or directory path. 
  - For single input files, omitting this flag appends `_out` to the filename (e.g., `image.jpg` → `image_out.jpg`).
  - For input directories, the tool assumes the output is also a directory and will automatically create it if it does not exist. If omitted, it appends `_out` directly to the input directory name (e.g., `input_dir` → `input_dir_out`), keeping your original input directory completely untouched.
- `-c`, `--continue-on-error`: (Optional) Continue processing the rest of a directory if an individual file fails (e.g., corrupted image or unsupported format). *Note: This flag applies only to directory batches; single-file failures will always exit immediately.*
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

### Direct API Usage (cURL)

Once the server is running, you can bypass the CLI entirely and send images directly to the `/upscale` endpoint using standard HTTP multipart form requests. You can optionally specify the desired output format via the `?ext=` query parameter (`.jpg` or `.png`).

```bash
# Upscale and save as a high-speed JPEG
curl -k -X POST -F "image=@path/to/image.jpg" "http://localhost:8080/upscale?ext=.jpg" -o path/to/output.jpg

# Upscale and save as a lossless PNG (default)
curl -k -X POST -F "image=@path/to/image.jpg" "http://localhost:8080/upscale" -o path/to/output.png
```

### 3. Builder

> [!TIP]
> **Pre-compiled Engines Available**: We have published pre-compiled, Architecture-Strict TensorRT engines for NVIDIA Ampere (`sm86`), Ada Lovelace (`sm89`), and Hopper (`sm90`) GPUs. You can download them directly from the [GitHub releases page](https://github.com/ls-ads/real-esrgan-serve/releases/tag/v0.1.0) and skip the build or cloud builder steps entirely!

You can compile an ONNX model into a hardware-specific TensorRT `.engine` file via the build command:

```bash
./real-esrgan-serve build --onnx path/to/model.onnx --engine path/to/model.engine --fp16
```

> [!TIP]
> **FP16 Recommended**: Building with the `--fp16` flag (enabled by default) enables half-precision optimizations. This significantly reduces the VRAM footprint and improves performance on modern GPUs (Maxwell architecture and newer). 
> 
> **Note on ONNX precision**: You do **not** need a half-precision `.onnx` file to build an FP16 engine. TensorRT will automatically cast the internal weights to FP16 during compilation if the `--fp16` flag is provided. However, using a half-precision ONNX file (via the `--half` export flag) is also supported.
> 
> **Note on Interface Compatibility**: Even when `--fp16` is enabled, the model's external input and output buffers remain **32-bit float (FP32)**. This ensures standard compatibility with the Go frontend and existing image processing logic (`imageutil.go`) while still reaping the 50% VRAM savings and speed from internal half-precision math.

> [!CAUTION]
> **VRAM Requirement**: Building the TensorRT `.engine` file from the ONNX model requires significant GPU memory (VRAM). 
> - **Full Precision (FP32)**: Requires **at least 16GB of VRAM**.
> - **Half Precision (FP16)**: Requires **at least 8GB of VRAM** (e.g., RTX 3060/4060, A2000).

### 4. Cloud Engine Builder

TensorRT compiles **Architecture-Strict** engines. This means an engine built on an RTX 3090 (`sm86` compute capability) contains hardware-specific CUDA kernel optimizations that will crash if executed on an A100 (`sm80`) or RTX 4090 (`sm89`). Because engines are strictly locked to the silicon architecture that compiled them, most users rent temporary cloud GPUs that match their deployment targets to generate them.

To streamline this, we provide a detached workflow image `tools/engine-build/Dockerfile` that handles the entire download-and-compile process gracefully:

**Pull or Build the Automated Image:**
You can pull the pre-built image directly from GitHub Container Registry:
```bash
docker pull ghcr.io/ls-ads/real-esrgan-serve/engine-build:v0.1.0
```
*(Alternatively, build it locally: `docker build -t engine-build:v0.1.0 tools/engine-build/`)*

When you deploy this image onto any cloud GPU instance, simply mount a persistent volume (or local folder) onto the container's `/output` path. The container will automatically boot up, download the verified `realesrgan-x4plus_fp16.onnx` file from the remote Github releases, run the intense C++ engine building process targeting the specific hardware you rented, and safely drop the finished `.engine` file into your mounted folder before exiting.

**Run the Builder:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/models:/output \
  ghcr.io/ls-ads/real-esrgan-serve/engine-build:v0.1.0
```

> **Dynamic Naming (`smXX`)**: The container will automatically parse the host's exact compute capability directly from the NVIDIA driver and dynamically name your engine file with the `sm` prefix and TensorRT version. For example:
> - Building on an **RTX A4000** yields: `realesrgan-x4plus-sm86-trt10.14_fp16.engine`
> - Building on an **RTX 2000 Ada** yields: `realesrgan-x4plus-sm89-trt10.14_fp16.engine`
> - Building on an **H100 PCIe** yields: `realesrgan-x4plus-sm90-trt10.14_fp16.engine`

## Docker

If you don't want to install the heavy TensorRT and CUDA C++ dependencies natively on your host machine, you can run the entire CLI via Docker. The container acts exactly like the native binary using an `ENTRYPOINT`.

**1. Pull or Build the Image:**
You can pull the pre-built image directly from the GitHub Container Registry:
```bash
docker pull ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0
```
*(Alternatively, build it locally: `docker build -t real-esrgan-serve:v0.1.0 .`)*

**2. Local Inference:**
Mount your local images into the container's `/workspace` directory using the `-v` flag:
```bash
docker run --rm --gpus all \
  -v $(pwd)/images:/workspace \
  ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0 \
  -i /workspace/input.jpg -o /workspace/output.jpg
```

**3. HTTP Server:**
Start the background server, map a port, and mount a directory containing your models:
```bash
docker run -d --gpus all \
  -v $(pwd)/models:/workspace \
  -p 8080:8080 \
  ghcr.io/ls-ads/real-esrgan-serve/cli:v0.1.0 \
  server start -p 8080 -e /workspace/realesrgan-x4plus.engine
```

## Generating the ONNX Model

You must generate the ONNX file (`realesrgan-x4plus_fp16.onnx`) from the official `Real-ESRGAN_x4plus.pth` PyTorch weights before building your engine. 

> [!TIP]
> **Pre-exported Model Available**: We have already exported the standard `realesrgan-x4plus` model to ONNX for you. You can download the verified [realesrgan-x4plus_fp16.onnx](https://github.com/ls-ads/real-esrgan-serve/releases/tag/v0.1.0) directly from the GitHub releases page.

Because the official extraction script relies on PyTorch and OpenCV (which requires specific `C` libraries like `libgl1` and `libglib2.0-0`), we have provided an isolated Dockerfile to generate it reproducibly without clashing with your host system.

### FP16 Export (Default)

The export container now defaults to **half-precision (FP16)** and automatically names the output file with the appropriate precision suffix (`_fp16.onnx` or `_fp32.onnx`).

1. Pull or build the exporter image:
```bash
docker pull ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0
```

2. Run the container:
```bash
docker run --rm -v $(pwd):/output ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0
```

This will automatically download the official `.pth` model, execute the trace, and save **`realesrgan-x4plus_fp16.onnx`** into your current directory!

> [!TIP]
> **FP32 Export**: If you explicitly require a full-precision (FP32) ONNX model, you can override the default entrypoint flags by running:
> `docker run --rm -v $(pwd):/output ghcr.io/ls-ads/real-esrgan-serve/onnx-export:v0.1.0 --half=false`
> (This will generate `realesrgan-x4plus_fp32.onnx`).

### Verification
To ensure the mathematical graph sequence was exported flawlessly without any hardware translation discrepancies, verify the MD5 checksum of the generated `.onnx` file:
```bash
$ md5sum realesrgan-x4plus_fp16.onnx 
3cc144c87adc6650ba950321a64ff5d9  realesrgan-x4plus_fp16.onnx
```

## Limitations & VRAM

Because this tool relies on the `realesrgan-x4plus` model processing via TensorRT, it holds the following constraints:

1. **Memory Ceiling**: All TensorRT context memory linearly correlates with the input image's dimensions. Since the `realesrgan-x4plus` model outputs are exactly 4x the input size in width and height (yielding a $16\times$ larger pixel map overall), large input files will cause drastic spikes in VRAM usage.
   - **Maximum Supported Resolution**: The default TensorRT execution context enforces a strict dynamic bounding box limit of **$1280 \times 1280$**. Any image exceeding these dimensions on either axis will be cleanly rejected by the server prior to processing.
   - **Inference Footprint (720p Example)**: Because TensorRT statically reserves the required VRAM for its allocated execution profiles:
     - **FP32**: Results in a dedicated **~14GB VRAM footprint**.
     - **FP16**: Results in a dedicated **~7GB VRAM footprint**.
2. **Dimension Tiling**: While other implementations fallback to patching or "tiling" to solve VRAM exhaustions, this pure C++ backend expects the entire activation map locally. Future updates may introduce chunking for massive geometries.
## Acknowledgements

This project is a bridge to the TensorRT implementation of Real-ESRGAN. We would like to give full credit to the original authors and the official project:

- **Original Project**: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Author**: [xinntao](https://github.com/xinntao)

The `realesrgan-x4plus` model and the underlying architecture are products of the research and development by the original authors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
