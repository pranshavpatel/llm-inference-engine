# Phase 6 same-session vLLM baseline

The [original VM archive](source.zip) is preserved byte-for-byte (SHA-256
`b16cdb14604eef378cc79a8eca407f8a00a611c54d0dcce5f00e1d5a8347d265`).
It contains 33 scored vLLM 0.30.0 replays on the L40S: three independent
repetitions at each of six short and five long-prefill offered rates. The
model and tokenizer use Qwen2.5-1.5B-Instruct revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The vLLM run used Torch
2.13.0+cu130 and BF16. Its code checkout was
`23c955bb20056ea34b1e768ef37eb3d671fffee1` with no tracked changes;
the earlier NanoServe archive used commit
`e22f9ea9b9d915db3a2a01c308474b3f4f66d02a` on the same VM. The
runner copied each NanoServe arrival trace byte-for-byte. Both plans and all 33 traces
and saved SLO scores were independently checked by
[`scripts/phase6_vllm_summary.py`](../../scripts/phase6_vllm_summary.py).

The vLLM launch fixed a 256 MiB KV pool, max 16 sequences, BF16, the same
context/batch-token caps (512 short, 4096 long), disabled prefix caching and
chunked prefill, and enabled per-request token metrics. FlashAttention 2,
compilation, and CUDA graphs remained enabled. These are comparable *workload
and KV-pool* settings, not equal total device footprint or equivalent
implementations. vLLM warned that disabling chunked prefill is not officially
supported for this model; the observed runs still completed cleanly.

The predeclared SLO was client TTFT <= 1 s and server-reported TPOT <= 100 ms.
The 60-second offered window and 30-second drain used 256 client workers.
All 3,691 vLLM requests completed, none failed, every run passed the p99
send-lag <= 50 ms gate, and no usage or TPOT metrics were missing. Twenty-one
requests completed during drain and earned no in-window goodput. All original
replays, scores, server logs, launch arguments, and environment manifests are
in the source zip.

| Workload / offered rate | Reference goodput | FlashInfer goodput | vLLM goodput | Median-of-run p99 client TTFT, reference / FlashInfer / vLLM |
| --- | ---: | ---: | ---: | --- |
| Short, 2 req/s | 0.00 | 2.12 | 2.15 | 19.20 / 0.145 / 0.023 s |
| Short, 3 req/s | 0.00 | 2.60 | 3.30 | 28.95 / 1.77 / 0.023 s |
| Short, 6 req/s | 0.00 | 0.27 | 6.10 | 35.32 / 24.82 / 0.027 s |
| Long, 0.8 req/s | 0.65 | 0.75 | 0.78 | 2.10 / 0.80 / 0.068 s |
| Long, 1.6 req/s | 0.083 | 0.917 | 1.85 | 17.40 / 3.55 / 0.074 s |

Goodput is the median of three run-level SLO-qualified completion rates;
small differences from offered rate reflect Poisson counts. Reference and
FlashInfer failed many requests at overload; vLLM failed none through the
highest tested rate. Thus this sweep establishes an observed gap on these
synthetic workloads, but **does not locate vLLM's capacity knee** or justify
a maximum-throughput ratio. The p99 values are descriptive median-of-run
p99s, not a statistically strong pooled tail guarantee. The long low-rate
runs have very small completion counts. vLLM has no NanoServe-compatible
server token-window counter, so its exact in-window emitted-token throughput
is not available; completed-token yield must not be substituted for it.

The [cross-engine summary](report/summary.json), [per-rate CSV](report/by-rate.csv),
[per-run vLLM CSV](report/vllm-runs.csv), [short goodput plot](report/short-goodput-vs-rate.svg),
and [long goodput plot](report/long-goodput-vs-rate.svg) are regenerable:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\phase6_vllm_summary.py `
  --vllm-archive environment\phase6-vm-vllm\source.zip `
  --nano-archive environment\phase6-vm-scored\source.zip `
  --output-dir environment\phase6-vm-vllm\report
```

The separately supplied [fresh-VM FlashInfer probe](fresh-probe-managed.json)
(SHA-256 `6e136703b3591132fcfa4ee7897023fc2b50532f14687a81b0792371bbba6415`)
passed BF16 paged decode on the L40S (0/6,144 elements outside its stated
0.03 absolute/relative tolerance). That is a kernel gate, not full-model
numerical parity.
