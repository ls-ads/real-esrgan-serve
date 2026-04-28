# Provider guide

How to add a new GPU provider (vast.ai, Modal, Replicate, …) to
real-esrgan-serve. Each provider lives under `providers/<name>/` and
ships its own deploy artefacts.

## Required files per provider

```
providers/<name>/
├── Dockerfile            # builds on top of real-esrgan-serve:<base>
├── handler.py (or eq.)   # provider-specific entrypoint
├── README.md             # deploy walkthrough + GPU-class recommendations
└── requirements.txt      # provider SDK pins
```

## Contract

Every provider handler must:

1. **Reuse `runtime/upscaler.py`** — don't reimplement inference.
   The Python helper handles ONNX session loading, CUDA EP / CPU EP
   selection, pre/post-processing, and emits standard
   `{event,id,...}` JSONL. Provider handlers translate the
   platform's job shape to/from helper frames.

2. **Warm the helper once per container / pod / worker.** Spawn it
   in `--serve` mode, wait for `{"event":"ready"}`, then accept jobs.
   First-request latency is the user's biggest visible cost; pre-paying
   it amortises across the pod's lifetime.

3. **Match the input contract.** Accept any of:
   - `image_url` — fetch via HTTPS
   - `image_base64` — decode in-process
   - `image_path` — read from a mounted volume
   The output is either base64 in the response, or written to
   `output_path` if the caller specified one. Keep this stable —
   iosuite.io and the iosuite CLI both pass jobs in this shape.

4. **Cap input dimensions at 1280×1280.** The model produces 4× output
   (5120×5120). Larger inputs exhaust GPU memory on consumer cards
   and produce diminishing visual quality past that resolution.
   Handler should reject earlier rather than burn GPU time.

5. **Surface helper errors as user-readable messages.** The Python
   helper distinguishes user errors (bad input image) from runtime
   errors (CUDA OOM) from environment errors (no model file). Don't
   collapse them all to "internal error" — the iosuite CLI uses the
   distinction to format error messages for end-users.

## Adding a provider — checklist

- [ ] `providers/<name>/Dockerfile` building on `real-esrgan-serve:<base>`
- [ ] Handler that opens a `WarmHelper`-style wrapper around the helper subprocess
- [ ] `requirements.txt` with pinned provider SDK
- [ ] `README.md` with build steps, env vars, GPU class table
- [ ] Update root `README.md`'s provider list section
- [ ] Smoke test: deploy, send a 1280×1280 input, confirm result + log shape

## GPU class recommendations format

The provider README should include a table like the one in
`providers/runpod/README.md`:

```
| Class       | Throughput (jobs/min) | Cold start | Notes |
```

Numbers come from the iosuite.io load tests against that provider.
If you're adding a new provider, run a single full benchmark cycle
on each GPU class you list and cite the run report.
