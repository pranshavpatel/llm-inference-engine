import unittest

import torch

from nanoserve.attention import PagedBatchMetadata, ReferencePagedAttention, contiguous_attention
from nanoserve.attention.flashinfer import probe_flashinfer
from nanoserve.memory import BlockManager, KVCacheSpec, PagedKVCache, PagedKVCacheManager


def make_cache(*, layers=2, blocks=8, block_size=4, heads=2, dim=8, device="cpu"):
    return PagedKVCache(
        KVCacheSpec(layers, blocks, block_size, heads, dim),
        dtype=torch.float32,
        device=device,
    )


def kv_values(layers, heads, tokens, dim, offset=0, device="cpu", dtype=torch.float32):
    keys = []
    values = []
    for layer in range(layers):
        base = torch.arange(heads * tokens * dim, device=device, dtype=dtype).reshape(
            heads, tokens, dim
        )
        keys.append(base + offset + layer * 1000)
        values.append(base * 0.25 + offset + layer * 2000)
    return keys, values


class PhysicalPoolTests(unittest.TestCase):
    def test_noncontiguous_scatter_and_gather_across_boundary(self):
        cache = make_cache()
        keys, values = kv_values(2, 2, 6, 8)
        table = (5, 1)
        for layer in range(2):
            cache.write(layer, table, 0, keys[layer], values[layer])
            actual_key, actual_value = cache.gather(layer, table, 6)
            torch.testing.assert_close(actual_key, keys[layer])
            torch.testing.assert_close(actual_value, values[layer])
        self.assertTrue(torch.count_nonzero(cache.keys[:, 0]) == 0)

    def test_append_within_and_across_pages(self):
        cache = make_cache(layers=1)
        first_key, first_value = kv_values(1, 2, 3, 8)
        second_key, second_value = kv_values(1, 2, 4, 8, offset=100)
        cache.write(0, (7, 2), 0, first_key[0], first_value[0])
        cache.write(0, (7, 2), 3, second_key[0], second_value[0])
        key, value = cache.gather(0, (7, 2), 7)
        torch.testing.assert_close(key, torch.cat((first_key[0], second_key[0]), dim=1))
        torch.testing.assert_close(value, torch.cat((first_value[0], second_value[0]), dim=1))

    def test_failed_write_does_not_modify_pool(self):
        cache = make_cache(layers=1)
        before_key = cache.keys.clone()
        before_value = cache.values.clone()
        key, value = kv_values(1, 2, 5, 8)
        with self.assertRaises(IndexError):
            cache.write(0, (1,), 0, key[0], value[0])
        self.assertTrue(torch.equal(cache.keys, before_key))
        self.assertTrue(torch.equal(cache.values, before_value))

    def test_pool_memory_accounting(self):
        cache = make_cache(layers=3, blocks=5, block_size=4, heads=2, dim=8)
        stats = cache.stats()
        expected_per_block = 2 * 3 * 4 * 2 * 8 * 4
        self.assertEqual(stats["bytes_per_block"], expected_per_block)
        self.assertEqual(stats["physical_pool_bytes"], expected_per_block * 5)

    def test_validation_rejects_aliasing_and_wrong_dtype(self):
        cache = make_cache(layers=1)
        key, value = kv_values(1, 2, 1, 8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cache.write(0, (1, 1), 0, key[0], value[0])
        with self.assertRaisesRegex(ValueError, "dtype"):
            cache.write(0, (1,), 0, key[0].double(), value[0].double())


class CacheManagerTests(unittest.TestCase):
    def test_fragmented_allocator_table_addresses_physical_pages(self):
        blocks = BlockManager(4, block_size=2, watermark=0)
        for index in range(4):
            blocks.allocate(f"holder-{index}", 2)
        blocks.free("holder-3")
        blocks.free("holder-1")
        cache = make_cache(layers=1, blocks=4, block_size=2)
        manager = PagedKVCacheManager(blocks, cache)
        self.assertEqual(manager.allocate("target", 3), (3, 1))
        key, value = kv_values(1, 2, 3, 8)
        manager.append("target", key, value)
        actual, _ = manager.gather("target", 0)
        torch.testing.assert_close(actual, key[0])

    def test_allocator_storage_lifecycle_and_counters(self):
        blocks = BlockManager(8, block_size=4, watermark=0)
        cache = make_cache(blocks=8)
        manager = PagedKVCacheManager(blocks, cache)
        table = manager.allocate("a", 6)
        keys, values = kv_values(2, 2, 3, 8)
        manager.append("a", keys, values)
        manager.reserve("a", 10)
        more_keys, more_values = kv_values(2, 2, 7, 8, offset=100)
        manager.append("a", more_keys, more_values)

        self.assertEqual(manager.written_tokens("a"), 10)
        self.assertEqual(manager.stats()["completed_kv_tokens"], 10)
        self.assertEqual(manager.stats()["reserved_tokens"], 10)
        expected_key = torch.cat((keys[1], more_keys[1]), dim=1)
        actual_key, _ = manager.gather("a", 1)
        torch.testing.assert_close(actual_key, expected_key)

        final_table = blocks.block_table("a")
        self.assertEqual(final_table[: len(table)], table)
        manager.free("a")
        self.assertEqual(blocks.stats()["free_blocks"], 8)
        self.assertEqual(manager.stats()["completed_kv_tokens"], 0)
        self.assertTrue(torch.count_nonzero(cache.keys[:, list(final_table)]) == 0)
        blocks.check_invariants()

    def test_append_beyond_reservation_is_transactional(self):
        blocks = BlockManager(4, block_size=4, watermark=0)
        cache = make_cache(layers=1, blocks=4)
        manager = PagedKVCacheManager(blocks, cache)
        manager.allocate("a", 3)
        key, value = kv_values(1, 2, 4, 8)
        before = cache.keys.clone()
        with self.assertRaisesRegex(IndexError, "reserved"):
            manager.append("a", key, value)
        self.assertEqual(manager.written_tokens("a"), 0)
        self.assertTrue(torch.equal(cache.keys, before))

    def test_incomplete_layer_transaction_cannot_commit(self):
        blocks = BlockManager(4, block_size=4, watermark=0)
        cache = make_cache(layers=2, blocks=4)
        manager = PagedKVCacheManager(blocks, cache)
        manager.allocate("a", 3)
        key, value = kv_values(2, 2, 3, 8)
        plan = manager.begin_append("a", 3)
        manager.write_layer(plan, 0, key[0], value[0])
        with self.assertRaisesRegex(RuntimeError, "every layer"):
            manager.commit_append(plan)
        manager.abort_append(plan)
        self.assertEqual(manager.written_tokens("a"), 0)

    def test_repeated_mixed_length_cycles_do_not_leak(self):
        blocks = BlockManager(16, block_size=4, watermark=0)
        cache = make_cache(layers=1, blocks=16)
        manager = PagedKVCacheManager(blocks, cache)
        for cycle in range(40):
            length = cycle % 11 + 1
            manager.allocate(f"r{cycle}", length)
            key, value = kv_values(1, 2, length, 8, offset=cycle * 10)
            manager.append(f"r{cycle}", key, value)
            manager.free(f"r{cycle}")
            blocks.check_invariants()
            self.assertEqual(manager.stats()["completed_kv_tokens"], 0)
            self.assertEqual(manager.stats()["free_blocks"], 16)
        self.assertTrue(torch.count_nonzero(cache.keys) == 0)
        self.assertTrue(torch.count_nonzero(cache.values) == 0)


class ReferenceAttentionTests(unittest.TestCase):
    def test_full_prefill_matches_contiguous_gqa(self):
        torch.manual_seed(4)
        cache = make_cache(layers=1)
        key = torch.randn(2, 7, 8)
        value = torch.randn(2, 7, 8)
        query = torch.randn(1, 4, 7, 8)
        table = (6, 2)
        cache.write(0, table, 0, key, value)
        metadata = PagedBatchMetadata.from_lists([table], [7], [7])
        actual = ReferencePagedAttention().prefill(query, cache, 0, metadata)[0]
        expected = contiguous_attention(query[0], key, value, query_start=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decode_matches_contiguous_at_page_boundary(self):
        torch.manual_seed(5)
        cache = make_cache(layers=1)
        key = torch.randn(2, 5, 8)
        value = torch.randn(2, 5, 8)
        query = torch.randn(1, 4, 1, 8)
        table = (3, 0)
        cache.write(0, table, 0, key, value)
        metadata = PagedBatchMetadata.from_lists([table], [5], [1])
        actual = ReferencePagedAttention().decode(query, cache, 0, metadata)[0]
        expected = contiguous_attention(query[0], key, value, query_start=4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_static_batch_with_variable_sequence_and_query_lengths(self):
        torch.manual_seed(6)
        cache = make_cache(layers=1)
        key_a, value_a = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        key_b, value_b = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
        cache.write(0, (7,), 0, key_a, value_a)
        cache.write(0, (1, 5), 0, key_b, value_b)
        query = torch.randn(2, 4, 3, 8)
        metadata = PagedBatchMetadata.from_lists([(7,), (1, 5)], [3, 6], [3, 2])
        actual = ReferencePagedAttention().prefill(query, cache, 0, metadata)
        expected_a = contiguous_attention(query[0], key_a, value_a, query_start=0)
        expected_b = contiguous_attention(query[1, :, :2], key_b, value_b, query_start=4)
        torch.testing.assert_close(actual[0], expected_a, rtol=0, atol=0)
        torch.testing.assert_close(actual[1, :, :2], expected_b, rtol=0, atol=0)
        self.assertTrue(torch.count_nonzero(actual[1, :, 2:]) == 0)

    def test_metadata_and_decode_validation(self):
        cache = make_cache(layers=1)
        query = torch.zeros(1, 4, 2, 8)
        metadata = PagedBatchMetadata.from_lists([(0,)], [5], [2])
        with self.assertRaisesRegex(ValueError, "does not cover"):
            ReferencePagedAttention().prefill(query, cache, 0, metadata)
        valid = PagedBatchMetadata.from_lists([(0,)], [3], [2])
        cache = make_cache(layers=1, block_size=4)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ReferencePagedAttention().decode(query, cache, 0, valid)

    def test_flashinfer_probe_is_explicit_on_windows(self):
        probe = probe_flashinfer()
        if __import__("platform").system() != "Linux":
            self.assertFalse(probe.available)
            self.assertIn("Linux", probe.reason)


if __name__ == "__main__":
    unittest.main()
