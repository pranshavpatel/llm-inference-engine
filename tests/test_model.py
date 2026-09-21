import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config as HFQwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from nanoserve.model import Qwen2Config, Qwen2ForCausalLM, Qwen2RMSNorm, RotaryEmbedding
from nanoserve.model.loader import WeightCoverageError, load_safetensors
from nanoserve.model.qwen2 import build_attention_mask


def tiny_config(tied: bool = True) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
    )


def hf_tiny_config() -> HFQwen2Config:
    values = {name: value for name, value in tiny_config().__dict__.items() if value is not None}
    return HFQwen2Config(**values)


class PrimitiveTests(unittest.TestCase):
    def test_rms_norm_matches_definition(self):
        norm = Qwen2RMSNorm(4, eps=1e-6)
        norm.weight.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        values = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6) * norm.weight
        torch.testing.assert_close(norm(values), expected, rtol=0, atol=1e-7)

    def test_rope_offset_matches_full_sequence_slice(self):
        rope = RotaryEmbedding(8, theta=10000.0)
        query = torch.randn(1, 2, 3, 8)
        key = torch.randn(1, 1, 3, 8)
        positions = torch.tensor([[5, 6, 7]])
        offset_query, offset_key = rope(query, key, positions)
        cos, sin = rope.cos_sin(torch.arange(8).unsqueeze(0), query.dtype)
        expected_query = query * cos[:, None, 5:8] + self._rotate(query) * sin[:, None, 5:8]
        expected_key = key * cos[:, None, 5:8] + self._rotate(key) * sin[:, None, 5:8]
        torch.testing.assert_close(offset_query, expected_query)
        torch.testing.assert_close(offset_key, expected_key)

    @staticmethod
    def _rotate(values):
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def test_offset_causal_and_padding_mask(self):
        padding = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)
        mask = build_attention_mask(2, 4, 2, torch.device("cpu"), padding)
        expected = torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 1]]]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_gqa_shapes_and_cache_growth(self):
        model = Qwen2ForCausalLM(tiny_config()).eval()
        first = model(torch.tensor([[1, 3, 5]]), use_cache=True)
        self.assertEqual(first.logits.shape, (1, 3, 67))
        self.assertEqual(len(first.past_key_values), 2)
        self.assertEqual(first.past_key_values[0][0].shape, (1, 2, 3, 8))
        second = model(torch.tensor([[7]]), past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(second.past_key_values[0][0].shape, (1, 2, 4, 8))


class ForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = Qwen2ForCausalLM(tiny_config()).eval()
        self.tokens = torch.tensor([[1, 5, 8, 13, 21, 34]])

    def test_full_forward_matches_incremental_cache(self):
        full = self.model(self.tokens).logits
        cache = None
        pieces = []
        for index in range(self.tokens.shape[1]):
            output = self.model(
                self.tokens[:, index : index + 1], past_key_values=cache, use_cache=True
            )
            cache = output.past_key_values
            pieces.append(output.logits)
        incremental = torch.cat(pieces, dim=1)
        torch.testing.assert_close(incremental, full, rtol=1e-5, atol=2e-6)

    def test_teacher_forced_logits_match_hugging_face(self):
        hf_config = hf_tiny_config()
        reference = HFQwen2ForCausalLM(hf_config).eval()
        self.model.load_state_dict(reference.state_dict(), strict=True)
        with torch.inference_mode():
            expected = reference(self.tokens).logits
            actual = self.model(self.tokens).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_greedy_tokens_match_hugging_face_forward_loop(self):
        hf_config = hf_tiny_config()
        reference = HFQwen2ForCausalLM(hf_config).eval()
        self.model.load_state_dict(reference.state_dict(), strict=True)
        prompt = self.tokens[:, :3]
        expected = prompt
        with torch.inference_mode():
            for _ in range(8):
                token = reference(expected).logits[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), dim=1)
            actual = self.model.generate(prompt, max_new_tokens=8)
        self.assertTrue(torch.equal(actual, expected))

    def test_tied_and_untied_embeddings(self):
        tied = Qwen2ForCausalLM(tiny_config(tied=True))
        untied = Qwen2ForCausalLM(tiny_config(tied=False))
        self.assertIs(tied.lm_head.weight, tied.model.embed_tokens.weight)
        self.assertIsNot(untied.lm_head.weight, untied.model.embed_tokens.weight)


class LoaderTests(unittest.TestCase):
    def test_tied_checkpoint_loads_with_complete_coverage(self):
        torch.manual_seed(9)
        source = Qwen2ForCausalLM(tiny_config(tied=True))
        state = {name: value.detach().clone() for name, value in source.state_dict().items()}
        del state["lm_head.weight"]
        with tempfile.TemporaryDirectory() as directory:
            save_file(state, str(Path(directory) / "model.safetensors"))
            target = Qwen2ForCausalLM(tiny_config(tied=True))
            report = load_safetensors(target, directory)
        self.assertTrue(report.tied_lm_head_from_embeddings)
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)

    def test_loader_rejects_incomplete_checkpoint(self):
        model = Qwen2ForCausalLM(tiny_config(tied=False))
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        del state["model.layers.0.self_attn.q_proj.bias"]
        with tempfile.TemporaryDirectory() as directory:
            save_file(state, str(Path(directory) / "model.safetensors"))
            with self.assertRaisesRegex(WeightCoverageError, "q_proj.bias"):
                load_safetensors(model, directory)


if __name__ == "__main__":
    unittest.main()
