import json
import unittest
from pathlib import Path

from scripts.phase5_logit_probe import inspect_request


class Phase5LogitProbeTests(unittest.TestCase):
    def test_selects_completed_hf_record_from_matching_pilot_trace(self):
        root = Path(__file__).resolve().parents[1] / "environment" / "phase5-vm-pilot"
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


if __name__ == "__main__":
    unittest.main()
