import unittest

import torch

from nanoserve.attention import ReferencePagedAttention
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager
from nanoserve.model import PagedQwen2Runner, Qwen2Config, Qwen2ForCausalLM


def config():
    return Qwen2Config(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def setup_runner(backend=None):
    torch.manual_seed(123)
    model = Qwen2ForCausalLM(config()).eval()
    blocks = BlockManager(32, block_size=4, watermark=0)
    cache = PagedKVCache(KVCacheSpec(2, 32, 4, 2, 8), dtype=torch.float32)
    manager = PagedKVCacheManager(blocks, cache)
    runner = PagedQwen2Runner(model, manager, backend or ReferencePagedAttention())
    return model, blocks, cache, manager, runner


class PagedModelTests(unittest.TestCase):
    def test_single_request_prefill_matches_contiguous_model_at_boundaries(self):
        for length in (3, 4, 5):
            model, blocks, _, manager, runner = setup_runner()
            tokens = torch.arange(1, length + 1, dtype=torch.long)
            manager.allocate("a", length)
            expected = model(tokens.unsqueeze(0)).logits[0]
            actual = runner.forward(["a"], [tokens])[0]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            self.assertEqual(manager.written_tokens("a"), length)
            manager.free("a")
            blocks.check_invariants()

    def test_variable_length_static_batch_matches_individual_forwards(self):
        model, _, _, manager, runner = setup_runner()
        first = torch.tensor([1, 5, 8, 13, 21])
        second = torch.tensor([2, 3, 5, 7, 11, 13, 17])
        manager.allocate("first", len(first))
        manager.allocate("second", len(second))
        expected_first = model(first.unsqueeze(0)).logits[0]
        expected_second = model(second.unsqueeze(0)).logits[0]
        actual_first, actual_second = runner.forward(
            ["first", "second"], [first, second]
        )
        torch.testing.assert_close(actual_first, expected_first, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(actual_second, expected_second, rtol=1e-5, atol=2e-6)

    def test_paged_decode_matches_contiguous_cache(self):
        model, _, _, manager, runner = setup_runner()
        prompt = torch.tensor([1, 4, 9, 16, 25])
        next_token = torch.tensor([36])
        # Reserve more pages than the current sequence needs; metadata must use
        # the committed sequence length without rejecting future capacity.
        manager.allocate("a", 12)
        expected_prompt = model(prompt.unsqueeze(0), use_cache=True)
        actual_prompt = runner.forward(["a"], [prompt])[0]
        torch.testing.assert_close(actual_prompt, expected_prompt.logits[0], rtol=1e-5, atol=2e-6)

        expected_decode = model(
            next_token.unsqueeze(0),
            past_key_values=expected_prompt.past_key_values,
            use_cache=True,
        ).logits[0]
        actual_decode = runner.forward(["a"], [next_token])[0]
        torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-5, atol=2e-6)
        self.assertEqual(manager.written_tokens("a"), 6)

    def test_failed_backend_does_not_commit_token_positions(self):
        class FailingBackend(ReferencePagedAttention):
            def prefill(self, query, cache, layer, metadata):
                raise RuntimeError("injected attention failure")

        _, _, _, manager, runner = setup_runner(FailingBackend())
        tokens = torch.tensor([1, 2, 3])
        manager.allocate("a", 4)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            runner.forward(["a"], [tokens])
        self.assertEqual(manager.written_tokens("a"), 0)
        self.assertEqual(manager.stats()["pending_appends"], 0)

    def test_runner_rejects_mismatched_pool_geometry(self):
        model = Qwen2ForCausalLM(config()).eval()
        blocks = BlockManager(8, block_size=4, watermark=0)
        cache = PagedKVCache(KVCacheSpec(1, 8, 4, 2, 8))
        manager = PagedKVCacheManager(blocks, cache)
        with self.assertRaisesRegex(ValueError, "layer count"):
            PagedQwen2Runner(model, manager, ReferencePagedAttention())


if __name__ == "__main__":
    unittest.main()
