# Phase 6 fixed-window NanoServe L40S evidence

Source: the owner-supplied `phase6-vm-scored-20260929T173830Z.zip`, preserved
as [source.zip](source.zip), SHA-256
`d349aaba41a30373d658502fd209592b2382ce0b4277e17bd5a492f92e84f825`.
The VM ran engine commit `e22f9ea9b9d915db3a2a01c308474b3f4f66d02a` on
an NVIDIA L40S (driver 580.173.02, CUDA 13.0), torch 2.13.0+cu130, and
FlashInfer 0.6.18.post1. It used the pinned Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16, 16-token pages
for the backend comparison, a 256 MiB KV pool, and at most 16 active
sequences. FlashInfer adds its separate 128 MiB planning workspace.

The archive is complete: 75 fixed-window replays (short/long paired backend
runs plus three ablation configurations), three independently seeded
repetitions per rate, and no run-error file. All 75 saved scores were
recomputed from the raw checksummed traces and replay records. Every p99
client send lag was below the predeclared 50 ms gate; all runs had a
server-side emitted-token window count, usage, and true-token TPOT where
needed. The sampler recorded no metrics/GPU-memory errors. Across all
configurations, 8,543 requests were offered and 2,459 failed, concentrated
at deliberately overloaded points. Failed and drain-completed work remains
in the full-cohort accounting.

The offered interval is 60 seconds with a 30-second bounded drain. Goodput
counts only requests completed before the offered-window cutoff that meet
both predeclared project SLOs: client TTFT <= 1 s and server-reported TPOT
<= 100 ms. Throughput is tokens actually emitted by the worker within the
offered interval, not HTTP chunks or full-run completed tokens. Values below
are **medians of three per-run measurements**, not a pooled percentile.

| Short synthetic workload | Reference | FlashInfer |
| --- | ---: | ---: |
| 2 offered requests/s: emitted tokens/s | 75.4 | 136.6 |
| 2 offered requests/s: SLO goodput (requests/s) | 0.00 | 2.12 |
| 2 offered requests/s: median run p99 client TTFT | 19.20 s | 0.145 s |
| 2 offered requests/s: failed requests, all repetitions | 92/387 | 0/387 |
| 3 offered requests/s: emitted tokens/s | 73.5 | 203.3 |
| 3 offered requests/s: SLO goodput (requests/s) | 0.00 | 2.60 |

At 4 offered requests/s, FlashInfer emitted about 211 tokens/s but its
median run p99 TTFT rose to 11.62 s, only 566/735 requests finished within
the offered windows, and the sampled queue grew from 39 near mid-window to
56 near cutoff. At 6, it recorded 210 failures/1,095 offered requests.
Those are overload observations, not sustainable capacity claims. The
reference path already had substantial backlog at 1 request/s and 92
failures/387 by 2 requests/s. On the long-prefill synthetic workload,
FlashInfer improved the 1.6 requests/s SLO goodput median from 0.083 to
0.917 requests/s, but the corresponding p99 TTFTs (17.40 and 3.55 s)
show that this was also beyond the 1-second tail target.

The 2x2 ablation at 2 offered requests/s varied **supported block size**
(16/32) and maximum active sequences (1/16), holding the 256 MiB KV
budget fixed. This is not paging-on/paging-off. With batch cap 16,
16-token blocks yielded 75.4 emitted tokens/s and 4.41% time-weighted
internal fragmentation; 32-token blocks yielded 76.5 tokens/s and 8.69%
fragmentation. With batch cap 1, throughput was only about 32 tokens/s
at either block size, with 258–260 failures out of 387. All four ablation
points were overloaded, so these describe behavior under pressure, not a
low-latency operating point. Sampled preemptions and recomputed-token
counter deltas were zero in this run; queued work, not recomputation, was
the observed pressure mechanism.

[summary.json](report/summary.json) contains every by-rate aggregate,
sample-count list, failure fraction, queue midpoint/cutoff, fragmentation,
and sampled peak GPU memory. [runs.csv](report/runs.csv) and
[by-rate.csv](report/by-rate.csv) retain per-repetition and grouped rows.
Figures: [throughput vs TTFT](report/throughput-vs-ttft.svg),
[goodput vs rate](report/goodput-vs-rate.svg),
[failure fraction](report/failure-vs-rate.svg),
[2x2 ablation](report/ablation-2x2.svg),
[block size](report/block-size.svg), and
[long-prefill goodput](report/long-goodput-vs-rate.svg).
Regenerate in a **new** output directory with:

```bash
python scripts/phase6_scored_summary.py \
  --archive environment/phase6-vm-scored/source.zip \
  --output-dir phase6-report-recheck
```

Limits: The prompts are deliberately repetitive synthetic lengths (40/131/
248 input tokens and 64 output tokens; long 1,028/2,055/3,082 input tokens
and 32 output tokens), not a production trace. Low-rate repetitions have
too few completions for strong p99 inference; their counts are published.
Sampled resource peaks are not continuous extrema. Reference and FlashInfer
remain numerically non-identical at full-model BF16 logits, and their
longer greedy outputs differ; the optimized path remains opt-in. An exact
same-session vLLM baseline was requested separately because prior Phase 5
vLLM data used a different workload and VM software stack.
