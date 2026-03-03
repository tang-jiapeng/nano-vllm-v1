"""
基于Flash Attention的多头注意力层，集成KV-cache管理、prefix caching和Triton优化的cache写入。
当 flash-attn 未安装时，自动回退到 Triton 自实现 kernel。
"""

import logging

import torch
import triton
import triton.language as tl
from torch import nn

from nanovllm.kernels.attention import flash_atten_prefill, paged_attention_decode
from nanovllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton内核：按slot_mapping将K/V并行写入KV-cache。"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1:
        return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """调用Triton内核将K/V写入KV-cache，写入前验证tensor内存布局。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )


class Attention(nn.Module):
    """
    多头注意力层，自动切换 prefill/decode 模式，集成 KV-cache 存储。

    后端选择策略:
        - flash-attn 可用时: 使用 flash_attn_varlen_func / flash_attn_with_kvcache
          （支持 prefix caching）
        - flash-attn 不可用时: 回退到 Triton 自实现 flash_atten_prefill /
          paged_attention_decode（不支持 prefix caching）
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """初始化注意力层，KV-cache在运行时由外部注入。"""
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        self.k_cache = self.v_cache = torch.tensor([])

    def _forward_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """使用 flash-attn 第三方库执行 attention（支持 prefix caching）。"""
        k_cache, v_cache = self.k_cache, self.v_cache

        if context.is_prefill:
            # Prefix caching：从cache读取完整K/V
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            return flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )

    def _forward_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context,
    ) -> torch.Tensor:
        """使用 Triton 自实现 kernel 执行 attention。"""
        k_cache, v_cache = self.k_cache, self.v_cache

        if context.is_prefill:
            return flash_atten_prefill(
                q,
                k,
                v,
                cu_seqlens=context.cu_seqlens_q,
                scale=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            return paged_attention_decode(
                q,
                k_cache,
                v_cache,
                block_tables=context.block_tables,
                context_lens=context.context_lens,
                scale=self.scale,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """将K/V写入cache后，根据后端和模式调用对应的attention kernel。"""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if HAS_FLASH_ATTN:
            o = self._forward_flash_attn(q, k, v, context)
        else:
            o = self._forward_triton(q, k, v, context)

        return o
