"""Seeded open-loop arrival plans, independent of any server's completion rate."""
import hashlib
import json
import math
import random
from nanoserve.config import positive_int


def make_trace(count: int, rate: float, seed: int, prompt_tokens: int = 128,
               output_tokens: int = 64) -> dict:
    positive_int("count", count)
    positive_int("prompt_tokens", prompt_tokens)
    positive_int("output_tokens", output_tokens)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    rng = random.Random(seed)
    arrival = 0.0
    requests = []
    for index in range(count):
        arrival += rng.expovariate(rate)
        requests.append({"request_id": f"r{index:06d}", "arrival_offset_s": arrival,
                         "prompt_tokens": prompt_tokens, "output_tokens": output_tokens})
    payload = {"schema_version": 1, "kind": "synthetic-length-plan",
               "seed": seed, "rate_rps": rate, "requests": requests}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}
