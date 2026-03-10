"""Fused MoE 层的量化策略。

与 ``linear.py`` 中 dense 线性层策略模式一致：
  1. ``create_weights`` – 在 MoE module 上注册堆叠 expert 权重 Parameter，
     并附加 ``weight_loader`` 回调供 loader 使用。
  2. ``forward`` – 通过 ``fused_moe()`` 执行完整的 gate+up → SwiGLU → down 计算。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from nanovllm.kernels.fused_moe import fused_moe

if TYPE_CHECKING:
    from nanovllm.layers.quantization.linear import AWQConfig


class FusedMoEMethodBase(ABC):
    """MoE 层量化策略基类。"""

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None: ...

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        norm_topk_prob: bool,
    ) -> torch.Tensor: ...


# ── FP16 / BF16 ─────────────────────────────────────────────────────


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """FP16/BF16 fused MoE：堆叠 expert 权重 + Triton kernel。"""

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size):
        layer.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        layer.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )

        def _w1_loader(param, loaded_weight, expert_id, loaded_shard_id):
            offset = loaded_shard_id * intermediate_size
            param.data[expert_id, offset : offset + intermediate_size].copy_(
                loaded_weight
            )

        def _w2_loader(param, loaded_weight, expert_id):
            param.data[expert_id].copy_(loaded_weight)

        layer.w1.weight_loader = _w1_loader
        layer.w2.weight_loader = _w2_loader

    def apply(self, layer, hidden_states, router_logits, top_k, norm_topk_prob):
        return fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            w1=layer.w1,
            w2=layer.w2,
        )


# ── AWQ int4 ─────────────────────────────────────────────────────────


class AWQFusedMoEMethod(FusedMoEMethodBase):
    """AWQ int4 fused MoE：堆叠 expert 量化权重 + Triton kernel。"""

    def __init__(self, config: AWQConfig) -> None:
        self.config = config

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size):
        g = self.config.group_size
        p = self.config.pack_factor
        E, D, FFN = num_experts, hidden_size, intermediate_size

        # gate + up（fused）
        layer.w1_qweight = nn.Parameter(
            torch.empty(E, D, (2 * FFN) // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.w1_qzeros = nn.Parameter(
            torch.empty(E, D // g, (2 * FFN) // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.w1_scales = nn.Parameter(
            torch.empty(E, D // g, 2 * FFN, dtype=torch.float16),
            requires_grad=False,
        )

        # down
        layer.w2_qweight = nn.Parameter(
            torch.empty(E, FFN, D // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.w2_qzeros = nn.Parameter(
            torch.empty(E, FFN // g, D // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.w2_scales = nn.Parameter(
            torch.empty(E, FFN // g, D, dtype=torch.float16),
            requires_grad=False,
        )

        # 绑定 weight_loader
        self._attach_loaders(layer, intermediate_size, p)

    @staticmethod
    def _attach_loaders(layer, intermediate_size, pack_factor):
        FFN, p = intermediate_size, pack_factor

        # w1 的 qweight / qzeros 按 packed 维切分
        def _w1_packed_loader(param, loaded_weight, expert_id, loaded_shard_id):
            shard = FFN // p
            offset = loaded_shard_id * shard
            param.data[expert_id, :, offset : offset + shard].copy_(loaded_weight)

        # w1 的 scales 按原始维切分
        def _w1_scales_loader(param, loaded_weight, expert_id, loaded_shard_id):
            offset = loaded_shard_id * FFN
            param.data[expert_id, :, offset : offset + FFN].copy_(loaded_weight)

        # w2 各参数直接按 expert 复制
        def _w2_loader(param, loaded_weight, expert_id):
            param.data[expert_id].copy_(loaded_weight)

        layer.w1_qweight.weight_loader = _w1_packed_loader
        layer.w1_qzeros.weight_loader = _w1_packed_loader
        layer.w1_scales.weight_loader = _w1_scales_loader
        layer.w2_qweight.weight_loader = _w2_loader
        layer.w2_qzeros.weight_loader = _w2_loader
        layer.w2_scales.weight_loader = _w2_loader

    def apply(self, layer, hidden_states, router_logits, top_k, norm_topk_prob):
        return fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            w1_qweight=layer.w1_qweight,
            w1_qzeros=layer.w1_qzeros,
            w1_scales=layer.w1_scales,
            w2_qweight=layer.w2_qweight,
            w2_qzeros=layer.w2_qzeros,
            w2_scales=layer.w2_scales,
        )
