"""Opt-in CUDA checks for the physical page pool and reference backend."""

import os

import pytest
import torch

from nanoserve.attention import PagedBatchMetadata, ReferencePagedAttention, contiguous_attention
from nanoserve.memory import KVCacheSpec, PagedKVCache

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("NANOSERVE_RUN_GPU_PAGING") != "1" or not torch.cuda.is_available(),
    reason="set NANOSERVE_RUN_GPU_PAGING=1 on a CUDA host",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
def test_cuda_physical_pages_match_contiguous_attention(dtype, page_size):
    torch.manual_seed(11 + page_size)
    length = 33
    pages = (length + page_size - 1) // page_size
    cache = PagedKVCache(
        KVCacheSpec(1, pages + 3, page_size, 2, 128), dtype=dtype, device="cuda"
    )
    table = tuple(reversed(range(pages)))
    key = torch.randn(2, length, 128, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(1, 12, length, 128, device="cuda", dtype=dtype)
    cache.write(0, table, 0, key, value)
    metadata = PagedBatchMetadata.from_lists([table], [length], [length])
    actual = ReferencePagedAttention().prefill(query, cache, 0, metadata)[0]
    expected = contiguous_attention(query[0], key, value, query_start=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
