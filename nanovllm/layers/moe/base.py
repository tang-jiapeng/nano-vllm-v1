from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseMoeBackend(ABC):
    """MoE 后端接口。

    这层抽象的目标和 mini-sglang 类似：
    - 模型层只负责准备 hidden_states / router_logits / expert 权重
    - 具体的 topk、token 重排、专家计算、reduce 都交给 backend

    这样后续如果你想替换成别的 MoE 实现，只需要换 backend，
    模型层不需要跟着一起改。
    """

    @abstractmethod
    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        """在 layer 上注册专家权重参数，并绑定 checkpoint loader。"""
        ...

    @abstractmethod
    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """执行完整的 MoE 前向。"""
        ...
