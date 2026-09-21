"""Opt-in real-model tests; excluded unless NANOSERVE_RUN_MODEL_TESTS=1."""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Pin updated only after an explicit parity run records the exact Hub revision.
MODEL_REVISION = os.environ.get(
    "NANOSERVE_QWEN_REVISION", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


@pytest.mark.skipif(
    os.environ.get("NANOSERVE_RUN_MODEL_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set NANOSERVE_RUN_MODEL_TESTS=1 on a CUDA host",
)
def test_real_model_teacher_forcing_and_greedy_parity():
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoserve.model import Qwen2Config, Qwen2ForCausalLM, load_safetensors

    local = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt"],
    )
    config = Qwen2Config.from_json(os.path.join(local, "config.json"))
    # FP32 is the architecture-correctness gate. Reduced-precision behavior is
    # recorded separately by scripts/model_parity.py with explicit diagnostics.
    dtype = torch.float32
    custom = Qwen2ForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    load_safetensors(custom, local)
    reference = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=dtype, attn_implementation="eager"
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(local)
    prompts = ["The capital of France is", "Write one word for the color of grass:"]

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            expected = reference(tokens).logits.float()
            actual = custom(tokens).logits.float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

        reference_tokens = tokens
        with torch.inference_mode():
            for _ in range(8):
                next_token = reference(reference_tokens).logits[:, -1].argmax(-1, keepdim=True)
                reference_tokens = torch.cat((reference_tokens, next_token), dim=1)
            custom_tokens = custom.generate(tokens, max_new_tokens=8)
        if not torch.equal(custom_tokens, reference_tokens):
            divergent = int((custom_tokens != reference_tokens).nonzero()[0, 1])
            prefix = custom_tokens[:, :divergent]
            logits = custom(prefix).logits[0, -1].float()
            top = torch.topk(logits, 2).values
            pytest.fail(
                f"first greedy divergence at absolute token {divergent}; "
                f"custom top-two margin={(top[0] - top[1]).item():.8f}"
            )
