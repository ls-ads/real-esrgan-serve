# Image-size sweep — 2026-04-29 (post-batching)

After shipping two-engine batched inference (commit `7253243`),
re-ran a focused sweep across image sizes on three trt endpoints to
validate the batched gains across resolution and find the optimal
GPU per-resolution. Three GPU classes × 11 jobs each = 33 jobs +
3 cold starts. Total: ~25 min wall, ~$1.50 in compute.

## Per-image exec time by (gpu, size, batch)

```
gpu        size   b=1 ms     b=4 ms     b4 win
--------------------------------------------------
rtx-4090   256       73         70    +4.1%
rtx-4090   384      175        165    +5.7%
rtx-4090   512      305        290    +4.9%
rtx-4090   720      615        607    +1.3%
rtx-4090   1024    1238          —      —
rtx-4090   1280    1909          —      —

rtx-3090   256      109        106    +3.2%
rtx-3090   384      224        240    -7.4%
rtx-3090   512      410        388    +5.2%
rtx-3090   720      736        752    -2.2%
rtx-3090   1024    1516          —      —
rtx-3090   1280    2365          —      —

a4000      256      177          —      —      (batched fetch missed)
a4000      384      405          —      —
a4000      512      735          —      —
a4000      720     1436          —      —
a4000      1024    2881          —      —
a4000      1280    7770          —      —
```

`—` for batch=4 above 720 = engine doesn't admit (batched profile
caps at 720×720; handler routes those through the single-mode
engine, which is what batch=1 already measures).

`—` for a4000 = batched calls returned with 0 outputs. The
AMPERE_16 RunPod pool mixes sm86 (A4000 Ampere) and sm89
(RTX 4000 / RTX 2000 Ada) hardware; the worker likely landed on
hardware whose actual sm-arch our manifest didn't have a batched
entry for at fetch time, and the fall-through path didn't surface
a clean error. **Worth fixing**: handler should log + record the
batched-fetch outcome in diagnostics so this is observable.

## Cost-per-image, ranked at each size

```
size  best gpu       batch  $/img (flex)
-------------------------------------------
256   rtx-3090       4      $0.000020   ← cheapest
384   rtx-3090       1      $0.000043
512   rtx-3090       4      $0.000074
720   rtx-3090       1      $0.000140
1024  rtx-3090       1      $0.000288
1280  rtx-3090       1      $0.000449
```

**rtx-3090 wins the $/img leaderboard at every size we tested.**
$0.00019/s flex × 1.0–2.4 s per image = $0.00002–$0.00045 per
image. rtx-4090 is faster per image but ~1.6× more expensive per
second; the speed advantage doesn't pay back the price gap on
this model.

## Findings

1. **Batched gains are smaller than the rtx-4090 validation
   suggested.** On rtx-4090, batch=4 saves 1–6% per image across
   sizes. On rtx-3090, the batched gain is essentially noise (-7%
   to +5%). The 26% win we measured at rtx-4090 720×720 right
   after shipping was an outlier — most sizes show single-digit
   improvement. Real-ESRGAN's compute scales linearly with pixel
   count, so the launch-overhead amortisation is a small fraction
   of total per-image work.

2. **rtx-3090 batched perf doesn't track rtx-4090's.** Same
   batched engine artifact (sm86), same TRT version, but Ampere's
   tactic selection produces different kernels. Per-image times
   are roughly flat across batch sizes. Either the rtx-3090's
   compute is already saturated at batch=1, or the sm86 batched
   engine's tactics weren't as well-tuned as sm89's. Worth
   recompiling sm86 on a 3090 (rather than A40) to see if the
   match improves — same arch, different memory bandwidth.

3. **Per-image cost scales close-to-linearly with pixel count.**
   On rtx-4090 trt: 73 ms → 1909 ms is a 26× exec-time increase
   for a 25× pixel-count increase (256² → 1280²). Within 5%, this
   is exactly what a memory-bandwidth-bound kernel looks like.
   That's useful for cost projection: an iosuite consumer with
   median input dim X can estimate $/img with a single linear
   model.

4. **A4000 lags badly at high res.** 7.77 s for a 1280×1280
   image — 4× slower than rtx-3090 for the same work, on a card
   that's only ~16% cheaper per second. At sizes ≥ 720, the
   16 GB tier loses its cost advantage entirely.

5. **For the original 60 fps × 60 s = 3600 video frames example
   at typical 1080p (1280×720)**, the new numbers say:
   - rtx-3090 trt batch=4: ~745 ms × 3600 / 4 = ~11 min per minute
     of video, $0.51 in flex GPU.
   - rtx-4090 trt batch=4: ~611 ms × 3600 / 4 = ~9 min, $0.71.
   rtx-3090 still wins on cost at video res; the speedup from
   rtx-4090 isn't worth the price premium for batch jobs.

## What's missing

- **Higher concurrency** (FOLLOWUPS #3): only tested with one
  thread per workload. Multiple parallel clients × multiple
  workers per endpoint would reveal whether the per-image numbers
  hold under contention.
- **a4000 batched re-test** with a fix for the silent
  fetch-failure case. Probably one logged-warning + a `pull
  whichever engine the actual sm reports` retry on the handler
  side.
- **sm86 batched recompile on a 3090.** A40 is 48 GB and may have
  picked compute-fat tactics that don't transfer well; a 3090 build
  would tune for the same memory bandwidth as the typical
  customer card.
