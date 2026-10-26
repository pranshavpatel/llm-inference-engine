"""CPU checks for the FlashInfer page-table adapter; no kernel import needed."""

import pytest
import torch

from nanoserve.attention import PagedBatchMetadata
from nanoserve.attention.flashinfer import flashinfer_page_tensors
from nanoserve.memory import KVCacheSpec, PagedKVCache


def test_flashinfer_metadata_omits_reserved_pages_and_preserves_order():
    cache = PagedKVCache(KVCacheSpec(1, 8, 16, 2, 8))
    metadata = PagedBatchMetadata.from_lists(
        [(5, 3), (7, 2, 6), (4, 0, 1, 3)], [1, 16, 33], [1, 1, 1]
    )
    indptr, indices, last_page_len = flashinfer_page_tensors(metadata, cache)
    assert all(t.dtype == torch.int32 for t in (indptr, indices, last_page_len))
    assert indptr.tolist() == [0, 1, 2, 5]
    assert indices.tolist() == [5, 7, 4, 0, 1]
    assert last_page_len.tolist() == [1, 16, 1]


def test_flashinfer_metadata_rejects_short_page_table():
    cache = PagedKVCache(KVCacheSpec(1, 4, 16, 2, 8))
    metadata = PagedBatchMetadata.from_lists([(0,)], [17], [1])
    with pytest.raises(ValueError, match="does not cover"):
        flashinfer_page_tensors(metadata, cache)
