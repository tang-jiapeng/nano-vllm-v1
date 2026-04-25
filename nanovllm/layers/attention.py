"""
基于Flash Attention的多头注意力层，集成KV-cache管理和prefix caching。
"""

import logging

import torch
from torch import nn

from nanovllm.kernels.kv_cache import store_kvcache
from nanovllm.utils.context import get_context

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    多头注意力层，自动切换 prefill/decode 模式，集成 KV-cache 存储。
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

        return self._forward_flash_attn(q, k, v, context)
