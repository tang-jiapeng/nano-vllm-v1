from __future__ import annotations

import torch

from nanovllm.kernels.triton.kv_cache import store_kvcache_kernel


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    N, num_heads, head_dim = key.shape
    hidden_dim = num_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden_dim and v_cache.stride(1) == hidden_dim
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        hidden_dim,
    )
