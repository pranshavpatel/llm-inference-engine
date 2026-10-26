# Phase 6 opt-in FlashInfer serving smoke

The project owner ran the pinned two-request Phase 4 debug trace on the Ubuntu
L40S VM against nanoserve's opt-in FlashInfer paged-decode backend. `server.log`
records a ready server for `Qwen/Qwen2.5-1.5B-Instruct` on port 8001, with
`attention_backend` set to `flashinfer-paged-decode-reference-prefill`, 292 KV
blocks, and a 128 MiB KV pool. The saved log contains only the readiness line;
the replay supplies the request-level evidence.

`replay.json` reports 2/2 completed, zero failed, and zero missing-usage
requests. `r000000` returned ` four,` with `length` finish and 4 prompt / 2
completion tokens; `r000001` returned ` Paris.` with `length` finish and 5
prompt / 2 completion tokens. Both text and usage records match the saved
reference debug smoke. The trace checksum is
`b283b40c9a433735f3a6e5134011c34192cae14708c60e6ec1746875f06e59d1`.

The source files were copied without modification. SHA-256:

| File | SHA-256 |
| --- | --- |
| `replay.json` | `39d086d4754dd6684ece6496b05ecb5c569588611004d64bcccfae624a3c12f4` |
| `server.log` | `0040d9c1dcab964c2256242ba9311e6fd36e40c90864e44855ffdac2ef762bae` |

This passes the short functional HTTP gate for the opt-in path. Two tiny
requests cannot establish numerical parity for arbitrary prompts, stable
capacity, or a performance improvement. The BF16 full-model comparison to
the gather path still fails, while a targeted float32 attention oracle check
supports the optimized kernel on the worst-differing layers. The reference
backend remains the default; a controlled same-host measurement is required
before making any speed claim.
