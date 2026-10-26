# L40S FlashInfer decode gate

The owner ran `scripts/phase6_flashinfer_probe.py` on the Ubuntu L40S VM after
installing matching prebuilt FlashInfer packages. `probe.json` passed the
declared BF16 tolerance (`atol=rtol=0.03`) for page size 16, GQA 12/2,
head dimension 128, and sequence lengths 1/16/17/33. All 6,144 elements
matched; maximum absolute error was 0.0078125.

Source attachment SHA-256 values: `phase6-flashinfer-probe-aot.json`
`0120b6445e142b6e32b8113504cc34a9c72fff89bbacea621784586bac62eeb0`;
`phase6-flashinfer-config.txt`
`84b7ca326d86215bada560e70c62df4153020b5bc2344af174dae9745c970dd4`.

The runtime used PyTorch 2.13.0+cu132, FlashInfer 0.6.18.post1,
flashinfer-cubin 0.6.18.post1, and flashinfer-jit-cache
0.6.18.post1+cu130. `show-config.txt` reports no `nvcc` and a module
registration warning, but the specific paged-decode kernel ran successfully.
That warning is not, by itself, a failure of this checked kernel. This test
establishes compatibility for the listed geometry, not full model parity or
serving performance.
