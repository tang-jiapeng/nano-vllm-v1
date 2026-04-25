"""基于Gumbel-Max技巧的token采样器，通过温度缩放控制分布锐度后从logits中采样token。"""

import torch
from torch import nn


class Sampler(nn.Module):
    """Token采样器，支持per-sequence温度控制，使用Gumbel-Max技巧实现高效随机采样。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """对logits进行采样。temperature=0时走greedy，其余走Gumbel-Max。"""
        logits = logits.float()

        greedy_mask = temperatures <= 1e-10
        greedy_tokens = logits.argmax(dim=-1)

        safe_temperatures = torch.where(
            greedy_mask,
            torch.ones_like(temperatures),
            temperatures,
        )
        scaled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)

        sample_tokens = probs.div(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
