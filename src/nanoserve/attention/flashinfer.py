"""Availability probe for the deferred optimized FlashInfer backend."""

from __future__ import annotations

import importlib.util
import platform

from nanoserve.attention.backend import BackendProbe


def probe_flashinfer() -> BackendProbe:
    if platform.system() != "Linux":
        return BackendProbe(
            name="flashinfer",
            available=False,
            version=None,
            reason="FlashInfer publishes Linux-only runtime wheels",
        )
    if importlib.util.find_spec("flashinfer") is None:
        return BackendProbe(
            name="flashinfer",
            available=False,
            version=None,
            reason="flashinfer-python is not installed",
        )
    import flashinfer

    return BackendProbe(
        name="flashinfer",
        available=True,
        version=getattr(flashinfer, "__version__", "unknown"),
        reason="package import succeeded; kernel smoke test still required",
    )
