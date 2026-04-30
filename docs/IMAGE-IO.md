# Image input/output contract

What this service accepts on the wire, what it returns, and what
gets silently transformed in between. Aimed at iosuite.io, third-
party callers, and anyone wiring a UI on top of a deployed
endpoint.

## 1. Quick reference

| Property | Value |
|---|---|
| **Image dimensions** | 64 × 64 minimum, **1280 × 1280 maximum** per axis without tiling, **4096 × 4096 maximum** with `tile: true` |
| **Aspect ratio** | Any — H and W are independently dynamic; rectangular OK |
| **Upscale factor** | Fixed 4× (output is exactly `(4·W) × (4·H)` pixels) |
| **Input formats accepted** | PNG, JPEG, WebP, BMP, TIFF, GIF, plus any other format PIL recognises |
| **Output formats produced** | JPEG (default) or PNG — set via `output_format: "jpg" \| "png"` |
| **Channels** | Always RGB internally; alpha is dropped, grayscale is promoted |
| **Bit depth** | 8-bit per channel in / out (any higher input is quantised) |
| **Color space** | sRGB assumed; ICC profiles + non-sRGB metadata are stripped |
| **Per-batch size cap** | 4 same-shape images (batched-engine profile) |
| **Per-request payload** | ~15 MB of binary image data (RunPod's ~20 MB body cap minus base64 overhead) |

## 2. How to send an image

The handler's `BatchPayload` accepts three forms — pick whichever
fits the caller. They're all equivalent once the image is on disk.

### 2.1 As a URL (`image_url`)

```json
{"input": {"image_url": "https://example.com/in.jpg",
           "output_format": "jpg"}}
```

The worker fetches via `requests.get(url, timeout=30)`. URL must
be HTTPS-reachable from the RunPod pod (not behind a firewall, not
on `localhost`, etc.).

### 2.2 As base64 (`image_base64`)

```json
{"input": {"image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
           "output_format": "jpg"}}
```

Either the raw base64 bytes (no prefix) or a data URL with the
`data:image/png;base64,` prefix — handler strips the prefix
automatically. The base64 expansion (~33 %) eats into the
~20 MB RunPod request-body cap.

### 2.3 As a path (`image_path`)

```json
{"input": {"image_path": "frames/0042.jpg"}}
```

Resolved against `WORKSPACE` (= `/workspace`) if relative.
Useful when the caller has already staged a network volume into
the pod, and avoids the base64 round-trip cost.

### 2.4 Batched

```json
{"input": {"images": [
   {"image_base64": "..."},
   {"image_url": "https://..."},
   {"image_path": "frame_b.jpg"}
 ],
 "output_format": "jpg",
 "discard_output": false,
 "telemetry": false}}
```

Batched requests can mix transport modes (some base64, some path,
etc.); the handler resolves each independently. The
`output_format` at the top level is the default; per-item
`output_format` overrides if set.

Practical batch size: 4 same-shape images go through the batched
TRT engine in a single forward pass (best perf). Mixed-shape
batches are grouped by shape and processed per-group.

## 3. What happens to the bytes

Every input image, regardless of how it was sent or what format it
was in, runs through this pipeline:

```
bytes → PIL.Image.open(BytesIO(bytes))
      → img.convert("RGB")        ← always 3-channel, 8-bit
      → np.asarray(img) / 255.0   ← float32 [0..1]
      → CHW transpose, batch-dim  ← (1, 3, H, W)
      → cast to engine input dtype (typically FP16)
      → TRT execute_async_v3
```

Things that get silently transformed:

| Input quirk | Handling |
|---|---|
| RGBA (alpha) | Alpha channel **discarded**. No transparency in output. |
| Grayscale (single channel) | **Promoted to RGB** by replicating the gray channel. |
| CMYK | PIL converts to RGB. |
| Indexed-palette (P) | PIL converts to RGB. |
| 16-bit-per-channel (e.g. 16-bit PNG) | Quantised to 8-bit on PIL load. |
| Non-sRGB color profile (Adobe RGB, P3, …) | Profile **ignored**. Pixels treated as sRGB. |
| EXIF orientation flag | **Not applied.** A "rotate 90° CW" EXIF tag is preserved literally; pixels stay in their on-disk orientation. |
| Animated GIF / multi-page TIFF | Only the **first frame / page** processed. |
| ICC profile | **Stripped** from output. |

If callers care about any of these (e.g. they have wide-gamut or
HDR images), they need to convert to plain 8-bit sRGB RGB before
sending. The service won't preserve them.

## 4. Size constraints, in detail

### 4.1 Maximum: 1280 × 1280 per axis

**Hard-rejected by the handler** (`MAX_INPUT_DIM = 1280` in
`providers/runpod/handler.py`) — anything over returns
`input WxH exceeds max 1280x1280` before any GPU work. A 1281 × 1
input fails just as fast as 4096 × 4096.

The TRT engine profile also caps at 1280 — `set_input_shape`
would reject larger inputs at the kernel level even if the
handler didn't.

Why 1280 and not larger:

  - Output is 4× input. At 1280 → 5120 output. Real-ESRGAN's
    intermediate feature maps after the 2× pixel-shuffles are
    `(1, 64, 4H, 4W)` elements. At 1280² that's 1.68 B elements —
    just under TRT's 2³¹ INT32 element-count cap. At 1500² it
    overflows.
  - VRAM scales the same way. 1280² FP16 input → ~3.3 GB
    intermediate; 2048² → ~8.6 GB. 24 GB consumer cards run out
    of headroom once workspace + I/O are accounted.

### 4.2 Minimum: 64 × 64 per axis

The TRT engine profile starts at `min=(1, 3, 64, 64)`. Inputs
below 64 in either dim get a clean `set_input_shape rejected`
error. Real-ESRGAN was trained on 128 × 128 patches; even valid
< 64 inputs would produce visually poor output, so the cap is
acceptable.

### 4.3 Aspect ratio

H and W are **independently dynamic**. Anything in
`64 × 64 ≤ (h, w) ≤ 1280 × 1280` works. Tested aspect ratios:

| Shape | Use case | Status |
|---|---|---|
| 720 × 720 | Square (engine `opt` point) | ✓ optimal |
| 1280 × 720 | 16:9 video frame | ✓ |
| 720 × 1280 | Vertical / portrait | ✓ |
| 480 × 1280 | Letterbox / panorama | ✓ |
| 1280 × 64 | Extreme aspect | ✓ technically; quality unverified |

The TRT engine compiles tactics across this whole range; per-image
exec scales with pixel count, not aspect ratio.

### 4.4 Batched-engine size cap (separate from handler cap)

The **batched** TRT engine has a tighter profile: `max=(4, 3, 720, 720)`.
Requests that fit go through the batched path (single forward pass
for N images, ~10-26 % per-image win). Anything above 720 × 720
in any dim falls back to per-image iteration on the primary
engine — same wall-clock as if no batched engine were available.

Routing decision happens at request time in
`runtime/upscaler.py:_serve_one_batch` based on the actual input
shape; callers don't need to think about it.

## 5. What you get back

### 5.1 Single-image legacy shape

Backwards compat with the pre-batching API:

```json
{"image_base64": "/9j/4AAQ...",
 "model": "realesrgan-x4plus",
 "input_resolution": "720x720",
 "output_resolution": "2880x2880",
 "output_format": "jpg",
 "outputs": [{"image_base64": "...", "exec_ms": 531, ...}],
 "_diagnostics": {...}}
```

### 5.2 Batched response

```json
{"outputs": [
    {"image_base64": "...", "input_resolution": "720x720",
     "output_resolution": "2880x2880", "output_format": "jpg",
     "exec_ms": 530},
    {"image_base64": "...", "input_resolution": "512x512",
     "output_resolution": "2048x2048", "output_format": "jpg",
     "exec_ms": 290},
    ...
 ],
 "model": "realesrgan-x4plus",
 "_diagnostics": {
    "providers": ["TensorrtDirect"],
    "model": "realesrgan-x4plus-l40s-sm89-trt10.8_fp16.engine",
    "batch_size": 2,
    "batch_total_ms": 820,
    "telemetry": {"samples": [...], "interval_ms": 200}  // if requested
 }}
```

### 5.3 Output specs

  - Pixel dimensions: **always exactly 4× input** in each axis. A
    720 × 480 input becomes a 2880 × 1920 output. No rounding,
    no aspect-ratio adjustment.
  - Channels: 3-channel RGB. No alpha even if input had it.
  - Bit depth: 8-bit per channel.
  - Encoding: PIL's default settings — JPEG quality ~75 (lossy,
    smaller files), PNG (lossless, larger).
  - File size rough estimate: PNG ~12-16× input PNG size;
    JPEG ~6-10× input JPEG size (quality 75 default).

### 5.4 Output destinations

Three modes, mutually exclusive per item:

| Mode | Set via | Response shape |
|---|---|---|
| Inline base64 (default) | nothing — default behaviour | `output.image_base64` |
| Saved to a path | `image_path: "out/0042.jpg"` per item | `output.output_path` (no bytes in response) |
| Discarded (benchmarking) | `discard_output: true` at batch level | `output.output_size_bytes` only |

The discarded mode is what the benchmark harness uses — saves
50-200 KB per image on the response payload, which matters when
running a batch of 64 frames.

## 6. Beyond 1280: tiling

For inputs over 1280 in any axis (1080p frames, 4K stills, etc.),
set **`tile: true`** at the top level of the request payload. The
handler accepts inputs up to 4096 × 4096 in tile mode and slices
them server-side; callers see one input → one output, same I/O
shape as a non-tiled request.

```json
{"input": {"image_url": "https://...", "tile": true,
           "output_format": "jpg"}}
```

Algorithm (`runtime/tiling.py`):

  1. Slice into 1024 × 1024 tiles with ≥32 px overlap on shared
     edges. Tile count is the smallest that covers the full image
     with at least min-overlap; positions are evenly distributed
     so adjacent tiles share a wide blend region.
  2. Run inference per tile through the warm session (no extra
     warmup cost — `serve` mode keeps the engine resident).
  3. Stitch into the output canvas with linear ramps across the
     full overlap region. Two complementary ramps sum to 1.0
     everywhere → no visible seams.
  4. Output is exactly 4× the input on each axis, same as the
     untiled path.

Trade-offs:

  - Memory: float32 stitch canvas. ~200 MB at 2K input → 8K output;
    ~3 GB at 4K input → 16K output. The 4096² cap is set by this
    canvas allocation; larger inputs need streaming-strip
    processing (a future enhancement).
  - Latency: per-tile time × tile count. For a 4 K (3840 × 2160)
    input with 1024² tiles and ~32 px overlap → 12 tiles. At
    ~600 ms per tile on rtx-4090 trt → ~7 s GPU time. Smaller
    inputs (≤ 2K) tile into 4 tiles → ~2.5 s.
  - The batched engine doesn't help — its profile maxes at 720²,
    smaller than the tile size.

Inputs at or below 1280 × 1280 with `tile: true` short-circuit to
the single-shot path inside the helper at zero extra cost. Setting
`tile: true` unconditionally is safe and is the recommended pattern
for callers that don't pre-check dimensions.

## 7. Failure modes (what callers see)

| Trigger | Response |
|---|---|
| Input > 1280 × 1280 (tile not set) | `{"output": {"error": "input WxH exceeds max 1280x1280 (set tile=true to accept up to 4096x4096)"}}` |
| Input > 4096 × 4096 (tile=true) | `{"output": {"error": "input WxH exceeds max 4096x4096"}}` |
| Input < 64 in either dim | `{"output": {"error": "set_input_shape(...) rejected — outside the engine's optimisation profile"}}` |
| All input fields empty | Validation error mentioning `image_url, image_base64, image_path` |
| `output_format` not in `{"jpg", "png"}` | Pydantic validation error |
| `image_url` returns non-200 / times out | `{"output": {"error": "http <code>: <body>"}}` or timeout |
| `image_path` doesn't exist | `{"output": {"error": "image_path not found: ..."}}` |
| Image decode fails (corrupt bytes) | `{"output": {"error": "<PIL error>"}}` |
| Worker boot failed (GPU / engine issue) | Every job returns `{"output": {"error": "boot_failed", "_diagnostics": {"boot_error": {...}}}}` until the worker recycles |

For batched requests, per-item errors don't sink the whole batch —
successful items return their outputs, failed ones land in
`_diagnostics.per_item_errors` with their original index.

## 8. Quality notes (model-level)

The realesrgan-x4plus model was trained on natural photographic
images. Performance varies by content type:

  - **Photos:** the design target. Excellent on most natural
    images.
  - **Anime / illustrations:** sub-optimal; consider
    `realesr-anime-6b` upstream (not in our manifest yet).
  - **Text:** model can blur text or invent characters that
    weren't there. Avoid for documents / screenshots.
  - **Heavy compression artefacts:** model amplifies what it sees —
    a heavily-JPEG'd input may have its blockiness "enhanced"
    into the output.
  - **Synthetic / AI-generated images:** untested; output
    quality is content-dependent.
  - **Very small text in photos:** likely degraded.

FP16 numerics introduce sub-pixel noise vs FP32 inference but
it's imperceptible on natural images. For pixel-perfect
reproducibility at the cost of ~2× exec time, the FP32 ONNX is
also available via `--model-path realesrgan-x4plus_fp32.onnx`
through the local CLI (not exposed on RunPod by default since
it doubles inference cost with no visible quality gain on
photos).

## 9. Migration notes

The single-image legacy shape (`image_url` / `image_base64` /
`image_path` at top level, no `images: [...]` array) is preserved
for backwards compatibility but **callers should migrate to the
batched `images: [...]` shape**:

  - Future API additions (per-item flags, telemetry per item) will
    only be exposed in the batched shape.
  - Single-image callers get exactly the same handler path
    internally — just wrapped as a 1-element batch.
  - The legacy single-image fields at the response top level
    (`image_base64`, `output_resolution`, etc.) will keep working
    for as long as iosuite.io is on the legacy shape, but they
    duplicate `outputs[0]` and aren't worth depending on going
    forward.
