# L40S reference-paged runner profile

Source: `phase6-paged-profile-20260928T220052Z-1-001.zip` supplied by the
project owner, SHA-256
`06796522c357fe877703b58f2c28ce3e8a734ab05c18ebd4770f2e9a5c3fb00e`.
Only the compact `summary.json` is checked in. The 464 MB uncompressed Chrome
trace stays outside Git; retain the source archive to inspect individual kernels.

The fixed eight-request BF16 Qwen2.5-1.5B batch on an L40S used the
`reference_paged_gather` backend and 16 output tokens per request. Instrumented
prefill took 221.5 ms; the median of 15 instrumented decode steps was 215.3 ms.
These are single-process diagnostic timings, **not** HTTP or throughput scores.
The `nanoserve.*_step` profiler ranges are not individual GPU kernels and must
not be interpreted as device self-time for one operator.

Across the 16 measured steps, `aten::_softmax` occurred 3,584 times, matching
8 requests x 16 steps x 28 model layers. The profile also recorded 14,336
`aten::index`, 28,928 `aten::arange`, and 39,864 `aten::copy_` calls. This is
consistent with the reference path gathering pages and executing small
attention operations independently per request and layer. It motivates a
batched paged-kernel compatibility check; it does **not** establish the speedup
of a replacement backend.
