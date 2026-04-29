# Sweep follow-ups

The `SWEEP-2026-04-29.md` analysis flagged four pieces of work that
the data argues for but weren't in scope of the original benchmark.
Below: what each one is, what it costs to do, what it costs *not* to
do, and the order I'd tackle them in.

## Priority order (recommended)

| # | Item | Effort | Cost | Throughput payoff |
|---|---|---|---|---|
| 1 | True batched inference | 1 day code + ~$1 compute | ~$1 (engine recompile × 2 sm-archs) | 2–4× per-image |
| 2 | Image-size sweep | 2 hr code + 2 hr sweep | ~$2 GPU | 0× (data only) |
| 3 | Higher-concurrency runs | 30 min code + 3 hr sweep | ~$3 GPU | 0× (data only) |
| 4 | Blackwell support (sm120) | 1 day code + ~$1 compute | ~$1 (full engine matrix rebuild) | enables a tier we currently skip |

#1 ships visible perf gains and is meaningfully cheaper than the
others on a $/win basis. #2 and #3 fill in the data we need to
actually choose batch sizes and concurrency knobs in production;
they're cheap and unblock decisions but don't ship perf themselves.
#4 unblocks one GPU tier (the 32 GB Blackwell one) that current
users mostly aren't on — defer until there's a real ask.

---

## 1. True batched inference

**The problem the sweep surfaced.** Per-image exec time is flat
across batch sizes (575 ms at b=1 == 565 ms at b=64 on l40s/trt).
That's because `_process_one_image` in `handler.py` iterates the
batch sequentially — each image gets its own TRT forward pass with
batch-dim 1. Server-side batching as it stands amortises only HTTP /
queue overhead, not GPU work. Average GPU utilisation sits at
33–72% — the rest is per-image setup overhead (`set_input_shape`,
buffer alloc/copy, exec launch) for a forward pass too short to hide
those costs behind.

**Why it's a small change.** The .onnx already exports with
`{0: "batch_size"}` declared as a dynamic axis (see
`build/export_onnx.py` `DYNAMIC_AXES`) — no re-export. The TRT
engines DON'T use that flexibility because `build/compile_engine.py`
fixes the profile at `min=(1,3,64,64)`, `opt=(1,3,720,720)`,
`max=(1,3,1280,1280)`. Three concrete steps:

1. **Update `compile_engine.py`** to widen the profile to span batch
   1..MAX_BATCH, e.g.:
   ```python
   profile.set_shape(
       "input",
       min=(1, 3, 64, 64),
       opt=(8, 3, 720, 720),       # the new opt: median request shape
       max=(16, 3, 1280, 1280),    # ceiling chosen by VRAM
   )
   ```
   Engine size grows ~30–50% from the wider tactic search; per-arch
   compile time goes from ~4 min to ~10 min.

2. **Update `runtime/upscaler.py` `TrtSession.run`** to pass batched
   NCHW with N > 1 directly. The current `set_input_shape((n, c, h, w))`
   call already accepts n > 1 — it'd just fail today because the
   engine's profile rejects it. Update I/O buffer sizing to use the
   actual N. Output shape becomes `(N, 3, 4H, 4W)` — same as today
   but the iteration loop in `handler._process_one_image` collapses
   into a single call.

3. **Refactor `handler.py`'s batch loop** to call TrtSession once
   per request with the stacked batch tensor, then split the output
   array back into per-image entries. Currently each image is its
   own `_HELPER.upscale()` call; we'd add a `_HELPER.upscale_batch()`
   and route batched requests through it. Mixed-resolution batches
   stay on the per-image path (the engine profile assumes a single
   shape per forward).

**Recompile cost**: 2 engines × ~10 min × ~$0.0005/s ≈ $0.60.

**Expected payoff**: GPU utilisation should jump from ~50% avg to
>90% on the high-batch end. Per-image exec drops 2–4× depending on
how much of the current 600 ms is launch overhead vs raw compute.
That'd push l40s/trt from 1.77 imgs/s to 5–7 imgs/s on b=16, and
shift the cost-per-image leaderboard meaningfully (cheaper tiers
benefit relatively less since their per-image launch overhead is a
smaller fraction of total).

**Risk**: a wider TRT profile produces engines that pick worse
tactics for the b=1 case. Worth measuring after recompile —
if b=1 latency regresses, ship two engines (one b=1-tuned, one
b≥4-tuned) and pick at boot based on expected workload.

---

## 2. Per-image-size sweep

**Why it matters.** The sweep used 720×720 throughout (the engine's
`opt` shape). Real-ESRGAN exec scales roughly with pixel count, so
1280×1280 should be ~3.2× slower per image and tighten the
cost-per-image leaderboard differently. Some workloads (4K source
material) hit 1280×1280, others (thumbnails) sit at 256×256 — without
size sweep data we can't tell users the right tier for *their*
resolution.

**Implementation**: trivial — add a `workload_image_size_sweep` in
`workloads.py` that emits jobs at `(w, h) ∈ {256, 512, 720, 1024,
1280}²` with `batch_size=1`, then a corresponding `--workload
image_size_sweep` flag. `sweep.py` already runs all workloads per
endpoint deploy.

**Cost**: ~2 hr code + 1–2 hr sweep run on a subset of GPUs
(rtx-3090 + l40s + a4000 cover the cost/throughput extremes). Each
extra workload per pair is ~5 min of GPU time, so 3 GPUs × 2
flavors × 1 workload ≈ 30 min ≈ $1–2.

**No code changes outside the bench package** — handler already
handles arbitrary sizes via the existing engine profile.

**Output**: a new section in the next sweep report — "cost per image
by resolution" — so users can pick their tier by the size of their
typical input.

---

## 3. Higher-concurrency runs

**Why it matters.** The sweep used `workersMax=2` and
`concurrency=4` — 4 client threads queued behind 2 workers, so we
were measuring queue throughput, not parallel-worker throughput.
Real production deployments scale workers up to 4–16 depending on
budget. We don't know whether RunPod's queue scheduler is the
bottleneck (probably not — it's fast) or whether GPU pool
availability throttles us at the larger workersMax counts (probably,
based on the deploy retries we already saw).

**Implementation**:

1. Plumb `workers_max` through `sweep.py` (currently hardcoded at 2
   in the `Namespace` it builds for `deploy_endpoint`). Default
   stays at 2 for cost-conscious runs.

2. Add a new workload `concurrency_sweep` that runs `concurrency=
   {1, 4, 16}` × `jobs_per_worker={4}` × `batch_size={4}` against
   the *same* endpoint, with `workersMax` matching the largest
   concurrency setting. Time the steady-state interval per
   concurrency level and record. The harness already supports
   per-spec concurrency.

3. Pick a small representative subset of GPUs (rtx-3090 + l40s)
   and sweep at workersMax=8 and 16. Watch RunPod's
   `queueDelayTime` to detect throttling.

**Cost**: ~30 min code + ~3 hr sweep at higher workersMax (each
worker = additional $/s during the run). Estimated $3–5 of GPU
time depending on how aggressively we provision.

**Risk**: RunPod's free quota / soft limits might cap concurrent
workers at some account-level. If we hit that, the sustained
workload's queueing artefacts dominate and we measure RunPod, not
the workload. The fix is then to talk to RunPod about a quota bump
rather than to keep iterating tests.

---

## 4. Blackwell (sm120) support

**Why it matters.** RTX 5090 / 5080 are excluded from the current
build. Today that's mostly fine — the 4090 covers consumer workloads
at lower cost — but if a customer hits cost/perf needs that only
Blackwell hits (memory bandwidth, FP4 quantisation, etc.), we're
blocked behind a base-image bump.

**Implementation** (the order matters — TRT versions cascade through
the toolchain):

1. **Bump runtime base image** in `Dockerfile.trt` from
   `nvidia/cuda:12.4.1-base-ubuntu22.04` to
   `nvidia/cuda:12.6.x-base-ubuntu22.04` (or 13.x — pick whichever
   has the longest support window).

2. **Update TRT pin** to a version with sm120 codegen. As of the
   last sweep, `libnvinfer10=10.7.0.23-1+cuda12.6` is the lowest
   confirmed-good pin. `apt-cache madison libnvinfer10` against the
   new base image confirms availability.

3. **Update build pod** in `build/remote_build.py` to install the
   matching `tensorrt-cu12==10.7.0`. Same pinning rules as today
   (use `tensorrt-cu12`, not the metapackage).

4. **Recompile every existing engine** with the new TRT version.
   ABI break — sm86 + sm89 engines built against TRT 10.1 won't
   deserialise on libnvinfer 10.7. Three compiles needed: sm86 +
   sm89 + sm120. `make remote-build-engine GPU_CLASS=…` for each.

5. **Update MANIFEST.json** with new engine entries (new sha256s,
   new TRT version), upload to GH Releases, **rebuild + push the
   trt image** so the embedded manifest is current. (See the
   `feedback_manifest_in_image` memory — the gotcha that bit us
   mid-sweep.)

6. **Add the rtx-5090 row to `bench/sweep.py`'s matrix** and
   `BLACKWELL_96` mapping in `runpod_deploy.py` (already present,
   just needs the schema price entry — already in
   `bench/schema.py`).

7. **Re-run the sweep** for the new engine set. The sm86/sm89
   results will shift slightly on the new TRT version (different
   tactic selection); old results stay in the DB for comparison.

**Compute cost**: 3 engine compiles × ~$0.30 each ≈ $1. Re-sweep
costs ~$5 (new full sweep including the new tier).

**Total effort**: ~1 day of code + waiting for compiles + the
re-sweep run. Lower priority unless there's a customer ask.

---

## Recommended sequencing if all four happen

1. **Week 1**: True batched inference (#1). Big perf win, small
   blast radius. Validates that the existing engine compile flow
   handles a wider profile cleanly before we ask it to handle a
   TRT-version bump too.

2. **Week 1 (later)**: Image-size sweep (#2). Cheap and informs
   whether the next round of optimisations should focus on small
   inputs (where launch overhead dominates) or large inputs (where
   memory bandwidth matters).

3. **Week 2**: Higher-concurrency sweep (#3). Once we've fixed the
   per-image launch overhead via batching, concurrency becomes the
   next bottleneck to measure.

4. **Deferred until customer ask**: Blackwell (#4). One day of work
   that doesn't move existing customers.

If only one of these gets done, do **#1**.
