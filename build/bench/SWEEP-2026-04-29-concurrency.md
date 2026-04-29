# Concurrency sweep — 2026-04-29

After the image-size sweep, ran a focused concurrency sweep on the
two trt tiers most likely to see real production traffic
(rtx-3090 = cost king, l40s = throughput king). Per (gpu, c) measures
end-to-end imgs/s + per-worker scaling efficiency + true USD/image.

Test: 4 client threads × 4 jobs each × batch_size=4, repeated at
c={1, 2, 4, 5}. Each pass against the same long-lived endpoint
deployed with `workersMax=5` (the standard RunPod account-wide
quota — anything higher returns
`Max workers across all endpoints must not exceed your workers
quota (5)`).

## Headline data

```
gpu        c    wall s   imgs/s   per-w/s   scaling  $/image
─────────────────────────────────────────────────────────────────
l40s       1    11.5     1.40     1.40      100%     $0.000379
l40s       2    29.8     1.07     0.54       38%     $0.000987
l40s       4    22.3     2.87     0.72       51%     $0.000739
l40s       5    16.1     4.98     1.00       71%     $0.000532
rtx-3090   1    18.0     0.89     0.89      100%     $0.000213  ← cheapest
rtx-3090   2    44.9     0.71     0.36       40%     $0.000534
rtx-3090   4    69.4     0.92     0.23       26%     $0.000824
rtx-3090   5    35.9     2.23     0.45       50%     $0.000427
```

`per-w/s` = imgs/s ÷ c (per-worker steady-state throughput). `scaling`
= per-w/s as % of c=1 baseline (100% = perfect linear scaling).
`$/image` is `c × wall × flex_$/s ÷ images_delivered` — the real
amount of GPU-second cost paid for each image.

## Findings

1. **Cheapest $/image is at c=1, not at high concurrency.** rtx-3090
   c=1 hits **$0.000213/img** — the lowest number anywhere in this
   project's data. Concurrency scaling losses (each new worker
   delivers less than 1.0× the per-worker throughput of c=1) mean
   that adding workers raises total $/image even though throughput
   goes up. For batch-mode workloads where wall time doesn't
   matter, c=1 is the right call.

2. **For real-time/latency workloads, l40s c=5 is the throughput
   king** at 4.98 imgs/s, ~2.2× the rtx-3090 c=5 rate. The
   $/image penalty is ~25% (`$0.000532` vs c=1's `$0.000379`),
   which buys you a 3.6× throughput jump. Net: cheaper-per-image
   on rtx-3090, faster-per-image on l40s, and l40s scales better.

3. **The c=2 case is anomalously bad** on both tiers (38-40%
   scaling efficiency). Two clients hitting an endpoint that
   hasn't fully scaled up its second worker yet seems to cause
   pathological queueing — both clients fight for the lone warm
   worker while RunPod's scheduler is still spinning up the
   second one. By c=4 the scheduler has equilibrated. A
   production tip: skip c=2 entirely; either go single-threaded
   or jump straight to c≥4.

4. **l40s scales meaningfully better than rtx-3090** under
   concurrency. l40s c=5 retains 71% of c=1 per-worker
   throughput; rtx-3090 retains only 50%. l40s actually has its
   per-worker rate INCREASE from c=4 (0.72) to c=5 (1.00),
   suggesting RunPod's pool has more reliably-fast l40s workers
   than rtx-3090 ones. Possible explanation: ADA_48_PRO (l40s
   pool) has fewer GPUs, so the workers we get tend to be on
   warmer hosts; AMPERE_24 (rtx-3090 pool) has more variation.

5. **Account quota limits how far this can scale.** workersMax=5
   account-wide is the cap; a customer needing >5 parallel
   workers would either need RunPod to raise their quota, or
   split traffic across multiple endpoints. The c=1 → c=5 curve
   suggests the diminishing-returns inflection isn't reached
   yet — we'd want to test c=8/16 to see whether the gains keep
   coming.

## Recommendations updated for concurrency

| Goal | Pick | $/img | imgs/s |
|---|---|---|---|
| **Batch-mode, cheapest** | rtx-3090 trt **c=1** | **$0.000213** | 0.89 |
| **Real-time, fastest** | l40s trt **c=5** | $0.000532 | **4.98** |
| **Balanced** | rtx-3090 trt **c=5** | $0.000427 | 2.23 |

For the original 60 fps × 60 s = 3600-frame video example at 720×720:

  - **rtx-3090 c=1**: 3600 ÷ 0.89 = ~67 min wall, $0.77 cost
  - **rtx-3090 c=5**: 3600 ÷ 2.23 = ~27 min wall, $1.54 cost
  - **l40s c=5**: 3600 ÷ 4.98 = ~12 min wall, $1.92 cost

l40s c=5 wins on time, rtx-3090 c=1 wins on cost. Choose by which
matters more for the workload.

## Caveats

- All numbers from a single run per (gpu, c). Variance from
  RunPod's host-roulette / pool warmth could shift the c=2-4
  middle ground noticeably. The c=1 and c=5 endpoints are
  steadier (one worker is naturally consistent; five workers
  averaged makes per-call jitter smaller).
- Only trt flavor tested. cuda flavor would scale similarly per
  worker but each worker is ~30-40% slower per image — the
  ratios should hold but absolute numbers shift.
- sm89 (l40s) used the batched engine; sm86 (rtx-3090) also.
  Both have the dual-engine routing live as of `7253243`.
