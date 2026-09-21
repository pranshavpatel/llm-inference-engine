import contextlib
import hashlib
import io
import json
import random
import unittest
from pathlib import Path

from nanoserve.bench import make_trace
from nanoserve.cli import main
from nanoserve.config import ModelGeometry
from nanoserve.memory import BlockManager, OutOfBlocks


class MemoryTests(unittest.TestCase):
    def test_geometry_and_capacity(self):
        geometry = ModelGeometry.from_config(json.loads(Path("configs/qwen2.5-7b.geometry.json").read_text()))
        self.assertEqual(geometry.kv_bytes_per_token(), 57344)
        capacity = geometry.capacity(1024**3, 16)
        self.assertEqual(capacity["bytes_per_block"], 917504)
        self.assertEqual(capacity["pool_bytes"] + capacity["unusable_remainder_bytes"], 1024**3)
        self.assertLess(capacity["unusable_remainder_bytes"], capacity["bytes_per_block"])

    def test_invalid_geometry(self):
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                ModelGeometry(value, 8, 2, 512, 64)
        with self.assertRaises(ValueError):
            ModelGeometry(2, 8, 3, 512, 64)
        with self.assertRaises(ValueError):
            ModelGeometry.from_config(dict(num_hidden_layers=2, num_attention_heads=3, hidden_size=8))

    def test_explicit_head_dim(self):
        geometry = ModelGeometry.from_config(dict(num_hidden_layers=2, num_attention_heads=4,
                                                 hidden_size=16, head_dim=8))
        self.assertEqual(geometry.kv_bytes_per_token(), 256)

    def test_small_pool(self):
        geometry = ModelGeometry(2, 4, 2, 32, 8)
        self.assertEqual(geometry.capacity(1, 16)["token_capacity"], 0)


class AllocatorTests(unittest.TestCase):
    def test_boundaries_and_addressing(self):
        for tokens, count in ((15, 1), (16, 1), (17, 2)):
            pool = BlockManager(4, 16, 0)
            table = pool.allocate("a", tokens)
            self.assertEqual(len(table), count)
            self.assertEqual(pool.slot("a", tokens - 1), (table[-1], (tokens - 1) % 16))
            with self.assertRaises(IndexError):
                pool.slot("a", tokens)
            pool.check_invariants()

    def test_watermark_is_available_for_growth(self):
        pool = BlockManager(4, 16, 0.25)
        pool.allocate("a", 48)
        with self.assertRaises(OutOfBlocks):
            pool.allocate("b", 1)
        pool.reserve("a", 64)
        self.assertEqual(pool.stats()["free_blocks"], 0)
        pool.check_invariants()

    def test_failed_growth_is_transactional(self):
        pool = BlockManager(2, 16, 0)
        pool.allocate("a", 16)
        before = pool.stats(), pool.block_table("a")
        with self.assertRaises(OutOfBlocks):
            pool.reserve("a", 48)
        self.assertEqual(before, (pool.stats(), pool.block_table("a")))
        pool.allocate("b", 16)
        pool.check_invariants()

    def test_failed_admission_is_transactional(self):
        pool = BlockManager(2, 16, 0)
        before = pool.stats()
        with self.assertRaises(OutOfBlocks):
            pool.allocate("a", 33)
        self.assertEqual(before, pool.stats())
        pool.allocate("a", 1)

    def test_free_and_duplicate_ownership(self):
        pool = BlockManager(2, 16, 0)
        pool.allocate("a", 1)
        with self.assertRaises(ValueError):
            pool.allocate("a", 1)
        pool.free("a")
        with self.assertRaises(KeyError):
            pool.free("a")
        self.assertEqual(pool.stats()["free_blocks"], 2)
        self.assertEqual(pool.stats()["internal_fragmentation"], 0)

    def test_fragmentation_and_no_shrink(self):
        pool = BlockManager(4, 16, 0)
        pool.allocate("a", 17)
        self.assertEqual(pool.stats()["unused_tail_slots"], 15)
        with self.assertRaises(ValueError):
            pool.reserve("a", 16)
        pool.reserve("a", 18)
        self.assertEqual(pool.stats()["unused_tail_slots"], 14)

    def test_invalid_inputs(self):
        for watermark in (-0.1, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                BlockManager(4, watermark=watermark)
        pool = BlockManager(4)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                pool.allocate("a", count)
        with self.assertRaises(ValueError):
            pool.allocate("", 1)

    def test_randomized_lifecycle_against_slot_model(self):
        for seed in range(10):
            rng = random.Random(seed)
            pool = BlockManager(32, 8, 0.05)
            expected = {}
            for step in range(1000):
                op = rng.choice(("add", "grow", "free")) if expected else "add"
                if op == "add":
                    key, tokens = str(step), rng.randint(1, 40)
                    can_fit = (tokens + 7) // 8 <= pool.stats()["free_blocks"] - 2
                    self.assertEqual(pool.can_allocate(tokens), can_fit)
                    try:
                        pool.allocate(key, tokens)
                        self.assertTrue(can_fit)
                        expected[key] = tokens
                    except OutOfBlocks:
                        self.assertFalse(can_fit)
                else:
                    key = rng.choice(list(expected))
                    if op == "free":
                        pool.free(key)
                        del expected[key]
                    else:
                        tokens = expected[key] + rng.randint(0, 20)
                        needed = (tokens + 7) // 8 - (expected[key] + 7) // 8
                        free = pool.stats()["free_blocks"]
                        try:
                            pool.reserve(key, tokens)
                            self.assertLessEqual(needed, free)
                            expected[key] = tokens
                        except OutOfBlocks:
                            self.assertGreater(needed, free)
                pool.check_invariants()
                self.assertEqual(pool.stats()["reserved_tokens"], sum(expected.values()))
                self.assertEqual(pool.stats()["allocated_blocks"], sum((n + 7) // 8 for n in expected.values()))
            for key in expected:
                pool.free(key)
            self.assertEqual(pool.stats()["free_blocks"], 32)


class TraceTests(unittest.TestCase):
    def test_reproducibility_and_checksum(self):
        trace = make_trace(100, 2, 42)
        self.assertEqual(trace, make_trace(100, 2, 42))
        self.assertNotEqual(trace["sha256"], make_trace(100, 2, 43)["sha256"])
        checksum = trace.pop("sha256")
        canonical = json.dumps(trace, sort_keys=True, separators=(",", ":"))
        self.assertEqual(checksum, hashlib.sha256(canonical.encode()).hexdigest())
        arrivals = [row["arrival_offset_s"] for row in trace["requests"]]
        self.assertTrue(all(b > a for a, b in zip([0] + arrivals, arrivals)))

    def test_invalid_trace(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_trace(10, rate, 0)

    def test_poisson_intervals_have_expected_mean(self):
        trace = make_trace(10000, 5, 0)
        mean_interval = trace["requests"][-1]["arrival_offset_s"] / 10000
        self.assertAlmostEqual(mean_interval, 0.2, delta=0.01)

    def test_cli_memory(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["memory", "--config", "configs/qwen2.5-7b.geometry.json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["bytes_per_token"], 57344)


if __name__ == "__main__":
    unittest.main()
