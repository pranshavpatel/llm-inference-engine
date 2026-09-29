import hashlib
import json
import unittest
from pathlib import Path

from scripts.phase5_logit_probe import inspect_request
from scripts.phase5_vllm_logprob_probe import prepare_probe


class Phase5LogitProbeTests(unittest.TestCase):
    @staticmethod
    def pilot_root():
        return Path(__file__).resolve().parents[1] / "environment" / "phase5-vm-pilot"

    def test_selects_completed_hf_record_from_matching_pilot_trace(self):
        root = self.pilot_root()
        trace = json.loads((root / "fixed-trace.json").read_text(encoding="utf-8"))
        replay = json.loads((root / "hf-fixed.json").read_text(encoding="utf-8"))
        request, record = inspect_request(trace, replay, "r000000")
        self.assertEqual(request["prompt"], "Two plus two equals")
        self.assertEqual(record["output_text"], " four, but but what if the numbers")
        with self.assertRaisesRegex(ValueError, "request ID"):
            inspect_request(trace, replay, "missing")
        replay["trace_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checksummed trace"):
            inspect_request(trace, replay, "r000000")

    def test_saved_hf_probe_matches_replay_and_excludes_vllm_choice_at_divergence(self):
        root = self.pilot_root()
        trace = json.loads((root / "fixed-trace.json").read_text(encoding="utf-8"))
        replay_bytes = (root / "hf-fixed.json").read_bytes()
        replay = json.loads(replay_bytes)
        probe = json.loads((root / "hf-logit-probe-r000000.json").read_text(encoding="utf-8"))
        self.assertEqual(probe["trace_sha256"], trace["sha256"])
        self.assertEqual(probe["hf_replay_file_sha256"], hashlib.sha256(replay_bytes).hexdigest())
        self.assertEqual(probe["generated_text"], replay["records"][0]["output_text"])
        self.assertTrue(probe["matches_hf_replay"])
        fourth = probe["steps"][3]
        self.assertEqual(fourth["chosen_text"], " but")
        self.assertEqual(fourth["chosen_token_id"], 714)
        self.assertEqual(fourth["top_two_logit_margin"], 0.75)
        self.assertNotIn(" what", [candidate["text"] for candidate in fourth["top_five"]])

    def test_vllm_probe_uses_same_saved_request_policy(self):
        root = self.pilot_root()
        trace = json.loads((root / "fixed-trace.json").read_text(encoding="utf-8"))
        replay = json.loads((root / "vllm-fixed.json").read_text(encoding="utf-8"))
        payload, record = prepare_probe(trace, replay, "r000000")
        self.assertEqual(payload, {
            "model": trace["model"],
            "prompt": "Two plus two equals",
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "logprobs": 10,
            "ignore_eos": True,
        })
        self.assertEqual(record["output_text"], " four, but what if you have three")
        with self.assertRaisesRegex(ValueError, "request ID"):
            prepare_probe(trace, replay, "missing")
        replay["trace_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checksummed trace"):
            prepare_probe(trace, replay, "r000000")

    def test_saved_vllm_probe_reproduces_replay_and_excludes_hf_choice(self):
        root = self.pilot_root()
        trace = json.loads((root / "fixed-trace.json").read_text(encoding="utf-8"))
        replay_bytes = (root / "vllm-fixed.json").read_bytes()
        replay = json.loads(replay_bytes)
        probe = json.loads((root / "vllm-logprob-probe-r000000.json").read_text(encoding="utf-8"))
        self.assertEqual(probe["trace_sha256"], trace["sha256"])
        self.assertEqual(probe["vllm_replay_file_sha256"], hashlib.sha256(replay_bytes).hexdigest())
        self.assertEqual(probe["http_status"], 200)
        self.assertTrue(probe["matches_vllm_replay"])
        response = probe["response"]
        self.assertEqual(response["model"], trace["model"])
        self.assertEqual(response["usage"]["completion_tokens"], 8)
        choice = response["choices"][0]
        self.assertEqual(choice["text"], replay["records"][0]["output_text"])
        self.assertEqual(choice["finish_reason"], "length")
        logprobs = choice["logprobs"]
        self.assertEqual(logprobs["tokens"][:4], [" four", ",", " but", " what"])
        fourth = logprobs["top_logprobs"][3]
        self.assertEqual(max(fourth, key=fourth.get), " what")
        self.assertAlmostEqual(fourth[" what"] - fourth[" if"], 0.875, places=6)
        self.assertNotIn(" but", fourth)
        self.assertGreaterEqual(fourth[" what"] - min(fourth.values()), 2.25)


if __name__ == "__main__":
    unittest.main()
