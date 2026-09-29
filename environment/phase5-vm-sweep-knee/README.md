# Phase 5 L40S knee pilot (5/6/7 requests/s)

The complete user-provided VM export is preserved in `source.zip` (SHA-256
`54b223a311641c061e6091f54f1a7943720a73d1eae7b85324c222191ec1651f`).
The generated report is unpacked here for browsing. The VM ran engine commit
`fe2f405617124ae98f787338b8957221ccdc1817` on a single NVIDIA L40S
(driver 580.173.02, `nvidia-smi` CUDA Version 13.0), with vLLM 0.30.0 and pinned
`Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Each engine replayed the same nine 30-second Poisson traces (three repetitions
per rate), with 16 generated tokens per request, a 20-second bounded drain,
and 256 non-polling client workers. Both used BF16, a 64-token context limit,
16 active sequences, and 4,672 KV-token slots. vLLM had prefix caching and
chunked prefill disabled but retained compiled/CUDA-graph and FlashAttention 2
paths. nanoserve used its slower reference paged-gather attention backend.
The source archive contains the exact server logs, package list, traces,
replays, warmups, and output records.
Regenerating `report-sweep` locally produced the same report content (the
archive and Windows-generated files differed only in newline encoding).

| Offered rate | nanoserve median run p99 client TTFT | vLLM median run p99 client TTFT | nanoserve completions after 30 s |
| ---: | ---: | ---: | ---: |
| 5 requests/s | 1.90 s | 17.9 ms | 73/465 |
| 6 requests/s | 7.36 s | 18.1 ms | 155/559 |
| 7 requests/s | 12.43 s | 17.9 ms | 245/649 |

Both engines completed all 1,673 requests in all nine runs, with no failures,
drain timeouts, unsent requests, or missing usage. The maximum per-run p99
client send lag was 0.83 ms, passing the predeclared 50 ms gate. The 6 and 7
requests/s nanoserve rows show server-side backlog rather than client send
delay: median TTFT in the second half of each run exceeded that in its first
half, and the final completion occurred 37.6–48.1 seconds after run start.
At 5 requests/s, latency was variable across repetitions (run p99 TTFT
1.83–3.73 s), so even that point is not established as stable capacity.
vLLM's latency stayed near 18 ms p99 at these rates; this pilot did not reach
its capacity limit.

These are diagnostic pilot observations, not SLO goodput or a scored
sustainable-throughput frontier. The report's p99 summary is the median of
three **per-run** p99 values, with only 149–232 requests per repetition. Its
full-run output-token throughput includes drain time and should not be read as
steady-state capacity. The pilot identifies a nanoserve knee around the
previous clean 4 requests/s point and the degraded 5 requests/s point;
further exploratory high-rate sweeps are not needed before profiling the
reference backend and designing scored measurements.
