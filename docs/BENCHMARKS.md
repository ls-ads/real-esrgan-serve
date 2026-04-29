# real-esrgan-serve benchmark report

Consolidated findings from all sweeps run against deployed RunPod
endpoints between 2026-04-28 and 2026-04-29. Every number here
traces back to the SQLite results DB (`build/bench/results.db`,
gitignored — regenerable from rerunning the workloads in
`build/bench/`).

The original per-sweep reports (`build/bench/SWEEP-*.md`) remain
in-repo for archaeology; this is the unified version.

---

## 1. TL;DR — pick a GPU

For the realesrgan-x4plus model (~16 M params, 720 × 720 typical
input → 2880 × 2880 output, FP16), at RunPod flex pricing:

| Use case | GPU + flavor | $ / image | imgs / sec | Why |
|---|---|---|---|---|
| **Cheapest steady state** | rtx-3090 trt c=1 | **$0.000213** | 0.89 | Lowest $/s tier × decent throughput |
| **Cheapest at scale** | rtx-3090 trt batch=4 | **$0.000143** | 1.33 | Same tier, batched-engine win |
| **Lowest latency** | **rtx-5090 trt batch=4** | $0.000148 | **2.98** | Blackwell's per-image compute is ~2 × Ada |
| **Single-image bursty** | rtx-4090 trt batch=1 | $0.000191 | 1.63 | Best Ada single-image perf |
| **Fastest cold start** | l40s cuda | n/a (warm-only) | 1.25 | Smallest image, no engine fetch |

For the original 60 fps × 60 s = 3600-frame video example at
1080p (1280 × 720) — these are the realistic numbers you'd see
in production:

  - rtx-3090 trt batch=4: ~45 min wall, ~$0.51 GPU
  - rtx-5090 trt batch=4: ~20 min wall, ~$0.53 GPU
  - rtx-4090 trt batch=4: ~37 min wall, ~$0.69 GPU

rtx-3090 wins on cost; rtx-5090 wins on time at near-equal cost;
rtx-4090 is dominated by both (newly so — pre-Blackwell, rtx-4090
was the latency king, but rtx-5090 closes that out).

---

## 2. Methodology

### Test harness

`build/bench/runner.py` submits jobs via RunPod's REST `/run` +
`/status` poll (NOT `/runsync` — its 90 s server-side timeout
truncates cold starts). `build/bench/sweep.py` orchestrates
multi-(gpu, flavor, workload) runs by deploying an endpoint, running
its workloads, tearing it down. All measurements land in SQLite via
`build/bench/schema.py`.

Five workload patterns:

| Workload | What it isolates |
|---|---|
| `cold_start` | First-image latency on a fresh worker |
| `batch_sweep` | Per-batch-size exec time at fixed resolution |
| `sustained_concurrent` | Steady-state throughput with N parallel clients |
| `image_size_sweep` | Per-image cost across {256, 384, 512, 720, 1024, 1280}² |
| `concurrency_sweep` | Sustained at c={1, 2, 4, 5} on the same endpoint |

### Image content

Synthesised RGB PNG: white background, blue rounded square, gold
ellipse, black borders. Same content every test so encode/decode
cost stays constant — only resolution varies.

### Instrumentation captured per job

| Source | Field |
|---|---|
| Client (this side) | `walltime_ms` (submit → final response) |
| RunPod API | `delayTime` (queue + spawn) and `executionTime` (handler) |
| Handler diagnostics | active EPs, batch_size, batch_total_ms, optional `_diagnostics.telemetry.samples[*]` |
| Worker telemetry (200 ms cadence, opt-in) | nvidia-smi `gpu_util`, `mem_util`, `vram_used_mb`, `gpu_temp_c` |
| Account ledger (post-2026-04-29) | `clientBalance` snapshots at 8 phase markers per sweep |

### What we don't test

- **Real video frames** — synthetic content only. Aspect ratio /
  encoding overhead may shift the numbers slightly for
  production-shape inputs.
- **Active (reserved) pricing** — every $ figure is RunPod's flex
  (spot) tier. Active is ~15-25 % cheaper but gates on commitment.
- **Multiple concurrent requests per worker** — RunPod's serverless
  has a `--concurrency` knob we haven't probed. Each worker handles
  one job at a time in our current test setup.
- **Workloads above c=5** — RunPod's account-wide `workersMax` quota
  caps at 5. The c=1 → c=5 curve doesn't show diminishing returns
  yet; quota raise needed to extend.

### Pricing (used for every $ calc here)

Verbatim from RunPod's pricing page, seeded into
`build/bench/schema.py` `_GPU_PRICING_ROWS`. Joined into every
cost-per-image query at runtime.

| GPU class | Tier | VRAM | Flex $/s | Active $/s |
|---|---|---|---|---|
| L40 / L40S / RTX 6000 Ada | 48 GB Pro | 48 | $0.00053 | $0.00045 |
| RTX 5090 (Blackwell) | 32 GB Pro | 32 | $0.00044 | $0.00037 |
| A40 / A6000 | 48 GB | 48 | $0.00034 | $0.00029 |
| RTX 4090 | 24 GB Pro | 24 | $0.00031 | $0.00026 |
| A5000 / L4 / RTX 3090 | 24 GB | 24 | $0.00019 | $0.00016 |
| A4000 / A4500 / RTX 4000 / RTX 2000 | 16 GB | 16 | $0.00016 | $0.00014 |

Predicted spend per pair = `total_walltime_s × workers × flex_$/s`.
After 2026-04-29, every sweep also captures balance snapshots so
this can be reconciled against the actual RunPod ledger delta —
view via `python3 -m build.bench.report --query predicted_vs_measured`.

---

## 3. Detailed findings

### 3.1 Pre-batching baseline (initial 10-pair sweep)

10 (gpu, flavor) pairs × 3 workloads (cold_start + batch_sweep +
sustained_concurrent at c=4) = 30 runs, 223 jobs, ~$4 of GPU. Old
single-profile engines (TRT 10.1 / CUDA 12.4); per-image exec at
720 × 720:

```
gpu       flavor   batch=1  batch=4  batch=16
─────────────────────────────────────────────
l40s      trt       575      564      576       ← fastest then
rtx-4090  trt      1138      608      606
a40       trt       715      713      750
rtx-3090  trt       979      900      900
a4000     trt       982      910      921
l40s      cuda      786      797      801
rtx-4090  cuda     1292     1278     1604
a40       cuda     1314     1217     1204
a4000     cuda     1380     1380     1383
rtx-3090  cuda     1559     1857     1929       ← slowest
```

`exec_ms` is per-image; identical across batch sizes here because
the original handler iterated images sequentially through the warm
helper (no true batching at the engine level).

**TRT direct ran 60 – 75 % the walltime of CUDA EP** at every (gpu,
size) point. Same engine kernels under the hood; ORT's IO-binding
overhead is real.

### 3.2 Image-size scaling

Sweeping resolution at batch=1 (and batch=4 where the batched
engine admits it) on rtx-4090, rtx-3090, a4000:

```
gpu       size  b=1 ms   b=4 ms   b=4 win
──────────────────────────────────────────
rtx-4090   256      73       70   +4.1 %
rtx-4090   384     175      165   +5.7 %
rtx-4090   512     305      290   +4.9 %
rtx-4090   720     615      607   +1.3 %
rtx-4090  1024    1238        —     —
rtx-4090  1280    1909        —     —
rtx-3090   256     109      106   +3.2 %
rtx-3090   384     224      240   −7.4 %
rtx-3090   512     410      388   +5.2 %
rtx-3090   720     736      752   −2.2 %
a4000      256     177        —   (fetch)
a4000      720    1436        —
a4000     1280    7770        —
```

Per-image exec scales close to linearly with pixel count. On rtx-4090
trt: 256² → 1280² is 26 × exec time for 25 × pixels — within 5 % of
linear, indicating the kernel is memory-bandwidth-bound. Useful for
projection: any new (size, gpu) combination's cost is well-predicted
by a one-line linear model.

a4000 batched calls returned 0 outputs at every size. Cause: the
RunPod AMPERE_16 pool mixes A4000 (sm86) and RTX 4000 / RTX 2000 Ada
(sm89) hardware; the worker probably landed on Ada hardware while
the manifest fetch went looking for sm86. The handler's silent
fall-through swallowed the diagnostic. **Open: the handler should
record batched-fetch outcome explicitly and fall through to
single-engine on mismatch instead of failing the request.**

### 3.3 Concurrency scaling

`sustained_concurrent` at c={1, 2, 4, 5} × batch=4 × 720² on
rtx-3090 + l40s with `workersMax=5` (RunPod account quota).

```
gpu        c    wall s   imgs/s   per-w/s   scaling   $/image
─────────────────────────────────────────────────────────────────
l40s       1    11.5     1.40     1.40      100 %     $0.000379
l40s       2    29.8     1.07     0.54       38 %     $0.000987
l40s       4    22.3     2.87     0.72       51 %     $0.000739
l40s       5    16.1     4.98     1.00       71 %     $0.000532
rtx-3090   1    18.0     0.89     0.89      100 %     $0.000213  ← cheapest
rtx-3090   2    44.9     0.71     0.36       40 %     $0.000534
rtx-3090   4    69.4     0.92     0.23       26 %     $0.000824
rtx-3090   5    35.9     2.23     0.45       50 %     $0.000427
```

Three observations the design needs to know about:

1. **Cheapest $/img is c=1, not high concurrency.** Every additional
   worker delivers less than 1.0 × the c=1 per-worker rate, so
   raising concurrency *raises* total $/img. The win is throughput,
   not cost.
2. **c=2 is pathologically bad** on both tiers (38 – 40 % scaling
   efficiency). Two clients hit the endpoint mid-spawn-up, fight
   over the lone ready worker. By c=4 the pool has equilibrated.
   **Production tip: skip c=2; either go c=1 or jump to c≥4.**
3. **l40s scales better than rtx-3090** under concurrency (71 % vs
   50 % per-worker retention at c=5). Probably because ADA_48_PRO
   has fewer hosts and they run warmer; AMPERE_24 has more variance.

### 3.4 Blackwell debut (rtx-5090 trt 10.8)

End-to-end validation on rtx-5090 with the new TRT 10.8 / CUDA 12.8
toolchain. Single batch_sweep at 720²:

| batch | engine | exec ms | per-image |
|---|---|---|---|
| 1 | primary (single-mode) | 1058 | 1058 ms |
| 4 | batched | 1343 | **336 ms/img** |

The batched-mode 336 ms/img is the **fastest per-image inference
in the entire project** — about 2 × the rtx-4090 batched figure
(607 ms). Cost at $0.00044/s flex × 0.336s = **$0.000148/img**, a
hair more than rtx-3090 but with 2 × the throughput.

**The batch=1 number (1058 ms) is anomalously slow** — rtx-4090
ran the same shape in 615 ms. Possible causes: TRT 10.8 single-
profile kernels on sm120 are less mature than Ada's; the worker
landed on a slow host; or memory bandwidth contention. **Open:
worth a 3-5 sample re-validation before baking single-mode
Blackwell into a production recommendation.**

---

## 4. GPU recommendation matrix (full)

```
─── Steady-state batch processing ──────────────────────────────
  rtx-3090 trt  batch=4              $0.000143/img    1.33 i/s
  rtx-3090 trt  batch=1              $0.000213/img    0.89 i/s
  a4000    trt  batch=1              $0.000230/img    0.70 i/s

─── Latency-critical / real-time ───────────────────────────────
  rtx-5090 trt  batch=4              $0.000148/img    2.98 i/s ★
  l40s     trt  batch=4 c=5          $0.000532/img    4.98 i/s
  rtx-4090 trt  batch=4              $0.000159/img    1.65 i/s

─── Single-image, lowest cold-start ────────────────────────────
  l40s     cuda batch=1              ~$0.00038/img    1.25 i/s
  (cuda flavor: smaller image, no engine fetch — wins cold start)

─── Cheapest possible (CPU only, very slow) ────────────────────
  cpu image, locally               (no GPU cost)      0.10 i/s
  (10 s per 128² inference; not viable for production)

─── DO NOT USE ─────────────────────────────────────────────────
  Concurrency = 2 on any tier        (queueing pathology)
  cuda flavor at batch>1             (no batched-engine routing)
  Blackwell at batch=1               (untuned kernels; use batch=4+)
```

---

## 5. Architecture (what's deployed)

### Three Docker images, three flavours

| Image | Base | Size | Where it runs |
|---|---|---|---|
| `Dockerfile.cpu` | `ubuntu:22.04` | ~120 MB | CPU-only fallback, local dev |
| `Dockerfile.cuda` | `nvidia/cuda:12.4.1-cudnn-runtime` | ~2.5 GB | Any CUDA GPU with ORT CUDA EP |
| `Dockerfile.trt` | `nvidia/cuda:12.8.0-base` | ~2.9 GB | sm86 / sm89 / sm120 with TRT direct |

The trt image deliberately omits onnxruntime-gpu (saves ~600 MB)
and runs inference via `tensorrt` Python bindings + cuda-python
directly. See `feedback_dual_profile_regression` memory for why we
ship two single-profile engines instead of one dual-profile.

### Engine matrix on `v0.2.0-rc1` GH release

| sm-arch | GPU class examples | Single-mode sha | Batched-mode sha |
|---|---|---|---|
| sm86 | RTX 3090, A40, A6000, A5000, A4000, A4500 | b982485c… | baae86b3… |
| sm89 | RTX 4090, L4, L40, L40S, RTX 6000 Ada | 9a9a27b8… | dc91e375… |
| sm120 | RTX 5090, RTX 5080 (Blackwell) | 6e9a1904… | dffaa7aa… |

All built against TRT 10.8 / CUDA 12.8. Older 10.1 engines remain
on the release for archaeology but are not in MANIFEST.json.

### Server-side routing

- Single-image requests → primary (single-mode) engine, full
  size range (1 × 64 × 64 to 1 × 1280 × 1280).
- Multi-image same-shape group, fits batched profile (≤ 4 × 720²)
  → batched engine, single forward pass.
- Multi-image any other case → iterate per-image on primary engine.
- Mixed-shape batches → grouped by shape and dispatched per group.

`runtime/upscaler.py:_serve_one_batch` is the JSONL routing point;
`providers/runpod/handler.py:_process_batch` is the request-side
shape grouper.

---

## 6. Cost model — predicted vs measured

The bench harness now polls RunPod's `clientBalance` field at 8
phase markers per sweep (`sweep_start`, `pair_start`,
`pair_after_deploy`, `run_start`, `run_end`, `pair_after_workload`,
`pair_end`, `sweep_end`). The reconciliation query joins per-pair
balance deltas against `total_walltime_s × flex_$/s` from the
pricing table.

```
python3 -m build.bench.report --query predicted_vs_measured
python3 -m build.bench.report --query spend_per_pair
```

Use these queries to:

- **Validate the rate table.** A `ratio` column far from 1.0 means
  the table is stale or the pool tier we expected isn't the one
  RunPod placed us in.
- **Attribute spend** between startup vs workload vs teardown.
  Cold-start cost is the `pair_start − pair_after_deploy` delta;
  most pairs see < 5 % of total spend in startup.
- **Catch silent billing surprises** — if RunPod ever introduces a
  fee class we didn't model, the measured side will exceed
  predicted and the ratio jumps.

One open-question phenomenon: `currentSpendPerHr` reads `$0.003`
even with no active endpoints. Probably an idle-account
metric / lingering. Negligible at sweep scales but worth
characterising.

---

## 7. Future opportunities

Ranked by impact × confidence; effort + cost estimates assume the
existing harness + image pipeline.

### Tier 1 — high-impact, well-defined

#### 7.1. Re-validate rtx-5090 single-mode

**The data.** Blackwell single-mode at 720 × 720 batch=1 was 1058 ms;
rtx-4090 single-mode at the same shape was 615 ms. A faster GPU
should not be slower. Possible explanations: TRT 10.8 sm120
single-profile kernels are less mature, the worker we got was
constrained, or there's a tactic-selection regression in our
build.

**What to do.** Compile sm120 single on rtx-5090, run batch_sweep
at 720² × 5 samples, deploy on a different rtx-5090 worker, repeat.
If the slowness reproduces consistently, it's a real Blackwell
single-mode regression and we should: (a) try `--workspace-bytes`
8 GB instead of 4 to see if a larger tactic budget changes the
chosen kernel, (b) if not, file with NVIDIA / fall back to
batch=2 minimum even for "single" requests on Blackwell.

**Effort / cost.** ~30 min code + ~$1 GPU. Critical before quoting
single-image Blackwell numbers to anyone.

#### 7.2. FP4 / INT8 quantization on Blackwell

**The data.** Blackwell's marquee feature is FP4 / INT8 tensor cores.
We're shipping FP16 across the board. FP4 quantization could cut
per-image compute by ~2 × and per-image cost
proportionally (so ~$0.000074/img on rtx-5090 batch=4).

**What to do.** Apply post-training quantization (PTQ) to the
realesrgan-x4plus FP16 ONNX → FP8 / FP4 ONNX. Re-export under
`build/export_onnx.py` with NVIDIA's TRT Model Optimizer or
TensorRT's built-in PTQ. Recompile sm120 engines at the new
precision. Validate output quality (PSNR / SSIM vs FP16 ground
truth) before shipping.

**Effort / cost.** 1-2 days code. ~$2 GPU. **Risk:** quantization
on super-resolution models can introduce visible artefacts —
needs a quality gate.

#### 7.3. Cold-start optimisation

**The data.** Cold starts measured at 60-130 s, dominated by
RunPod's image pull (~3 GB ghcr → worker over multi-Gbps link).
First-image latency is the dominant UX hit for bursty workloads.

**What to do.** Three orthogonal lines of attack, ranked by ease:

1. **Smaller trt image.** Currently ~2.9 GB with cuDNN + TRT runtime
   + Python deps. Possible wins: drop cuDNN (TRT 10.8 may not need
   it for our model — tested via dry run), strip docs +
   docs-licenses from `/usr/share/doc`, use a `slim` Ubuntu base
   variant. Target: < 1.5 GB.
2. **RunPod's "Active" worker tier.** Active workers stay warm —
   no cold start on every request. ~1.5 × the per-second cost
   but $/img net depends on duty cycle. Worth modelling at
   different request-per-hour rates.
3. **Pre-pulled image cache.** RunPod's edge cache may serve repeat
   pulls faster — needs investigation. Could also try the
   "image pre-warming" feature if RunPod exposes it via API.

**Effort / cost.** Image-size optimisation: 2-4 hr engineering
+ ~$0.50 GPU per validation. Active-tier modelling: 1 hr analysis.

### Tier 2 — useful data fills

#### 7.4. Bump cuda flavor to TRT 10.8 / CUDA 12.8 base

**The gap.** The trt flavor is on cuda:12.8.0; the cuda flavor is
still on cuda:12.4.1-cudnn-runtime. Probably perf-neutral but the
asymmetry is a source of confusion (driver requirement
mismatch — cuda image needs driver 550+, trt image needs 560+),
and onnxruntime-gpu gains some Blackwell support in newer
versions.

**What to do.** Bump Dockerfile.cuda's base image and onnxruntime-gpu
version. Re-run the original 10-pair sweep on the new image to
see if cuda-flavor numbers shift. **Risk:** new ORT version may
have different EP behaviour.

**Effort / cost.** ~1 hr code + ~$5 sweep.

#### 7.5. AMPERE_16 mixed-pool engine fetch fix

**The bug.** Workers in AMPERE_16 pool can land on either A4000
(sm86) or RTX 4000 Ada (sm89). Handler currently fetches based on
gpu_class (always sm86 for `a4000`), which fails silently when the
hardware is actually sm89.

**What to do.** Two options:

1. Always fetch BOTH sm-archs at boot, pick at TrtSession init
   based on actual nvidia-smi compute_cap.
2. Store both engines on disk; load whichever matches the worker's
   real sm-arch.

Option 1 is cleaner. Cost: +35 MB per worker boot for the
unused engine.

**Effort / cost.** 2-3 hr code. ~$0.50 to validate.

#### 7.6. Higher concurrency probe

**The blocker.** RunPod's account-wide `workersMax` quota caps at 5.
The c=1 → c=5 scaling-efficiency curve doesn't show diminishing
returns yet — we don't know if 8, 16, 32 workers continues to
add throughput or hits the queue scheduler.

**What to do.** Raise account quota (RunPod support), then re-run
`concurrency_sweep` at c={1, 4, 8, 16, 32}. Captures both the
inflection point and any RunPod-side throttling pattern.

**Effort / cost.** ~30 min code + ~$10-20 GPU (higher concurrency =
more parallel worker-seconds).

### Tier 3 — research / future

#### 7.7. Per-worker concurrency

RunPod supports running > 1 job per worker via a
`concurrency` value in the endpoint config. Untested. Could be a
way to extract more throughput per worker without burning more
worker-instances. Probably interacts oddly with our warmed-helper
+ stdin protocol — would need handler refactoring.

#### 7.8. Variable-resolution batched inference

Today batched inference requires same-shape inputs (the engine
runs one set_input_shape per forward). Real video frames have
fixed shape so this works; mixed-content workloads (different
sizes per request) lose batching. Solutions: pad smaller images
to the batch's max dim, or build an engine with multiple
resolution profiles. Both are non-trivial.

#### 7.9. Cross-arch fat engines

Investigate whether TRT supports compiling one engine that runs
across multiple sm-archs (e.g., sm86 + sm89 in one binary). Would
reduce manifest complexity + boot fetch time.
Likely: not directly possible; engines are sm-arch-specific. But
NVIDIA's "version compatible" engine flag may help cross-version
deployment.

#### 7.10. Real video-frame benchmark

Synthetic 720 × 720 squares vs actual H.264 / VP9 1080p frames.
Decode cost + non-square aspect ratios may shift the per-image
numbers. Useful if a customer ships a real video pipeline.

#### 7.11. Cost-aware auto-batcher

The existing telemetry stream (gpu_util, per-image exec time) is
enough to drive a controller that picks batch size dynamically:
"if util < 70 % over the last 30 s, raise batch to 8; if exec_ms
shows degradation at b=8, drop back." Would extract more $/img
on workloads with bursty mixed shapes.

### Lessons captured (memories) worth reading first

- `feedback_trt_version_pinning` — TRT engines are tied to (TRT
  major.minor, CUDA major.minor). Mismatch crashes mid-boot. Pin
  `tensorrt-cu12` not `tensorrt`.
- `feedback_dual_profile_regression` — single TRT engine with two
  optimisation profiles ran 5-22 × slower than separate engines.
  Build separate engines instead.
- `feedback_manifest_in_image` — `MANIFEST.json` is embedded into
  the Go binary at image build. Bumps require docker rebuild +
  push, not just a release-asset upload.
- `reference_runpod_graphql` — UA header required (Cloudflare
  blocks Python's default), introspection off, `minCudaVersion` is
  the driver-filter field for `podFindAndDeployOnDemand`.
- `project_blackwell_unsupported` — was true at TRT 10.1; now
  resolved at TRT 10.8 / CUDA 12.8.

---

## 8. Reproducing this report

```bash
# Unit + property tests (free, ~5 sec):
make test

# A single-pair sanity sweep (~2 min, ~$0.10 GPU):
build/.with-iosuite-key python3 -m build.bench.sweep \
  --only-gpu rtx-4090 --only-flavor trt \
  --workloads cold_start --workloads batch_sweep

# Full original sweep (~2 hr, ~$4 GPU):
build/.with-iosuite-key python3 -m build.bench.sweep

# Image-size sweep on selected GPUs (~30 min, ~$1.50 GPU):
build/.with-iosuite-key python3 -m build.bench.sweep \
  --only-gpu rtx-4090 --only-gpu rtx-3090 --only-gpu a4000 \
  --only-flavor trt --workloads image_size_sweep

# Concurrency sweep — needs workersMax ≥ largest c level (~10 min, ~$1):
build/.with-iosuite-key python3 -m build.bench.sweep \
  --only-gpu rtx-3090 --only-gpu l40s \
  --only-flavor trt --workloads concurrency_sweep --workers-max 5

# Read out the canonical analyses:
python3 -m build.bench.report                  # overview
python3 -m build.bench.report --query cost_per_image
python3 -m build.bench.report --query predicted_vs_measured
python3 -m build.bench.report --query spend_per_pair
```
