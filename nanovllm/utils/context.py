"""
全局推理上下文管理。

vLLM v1 风格：移除显式 is_prefill 标志，改为由 cu_seqlens_q 是否存在自动推断。
新增 seq_need_compute_logits 支持统一批次中选择性计算 logits。
"""

from dataclasses import dataclass

import torch


@dataclass
class Context:
    """
    全局推理上下文，在模型组件间传递 attention 所需的元信息。

    is_prefill 由 cu_seqlens_q 是否存在自动推断：
      - cu_seqlens_q is not None → 使用 varlen attention（prefill/混合批次）
      - cu_seqlens_q is None     → 使用 decode attention（CUDA Graph 路径）
    """

    is_speculative: bool = False
    num_speculative_tokens: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_need_compute_logits: torch.Tensor | None = None

    @property
    def is_prefill(self):
        """是否使用 varlen attention（eagor 模式下的 prefill 或混合批次）。"""
        return self.cu_seqlens_q is not None


_CONTEXT = Context()


def get_context():
    """获取当前全局 Context。"""
    return _CONTEXT


def set_context(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    seq_need_compute_logits=None,
    is_speculative=False,
    num_speculative_tokens=0,
):
    """设置全局 Context，由 ModelRunner 在每次推理前调用。"""
    global _CONTEXT
    _CONTEXT = Context(
        is_speculative,
        num_speculative_tokens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        seq_need_compute_logits,
    )


def reset_context():
    """重置全局 Context 为默认空值。"""
    global _CONTEXT
    _CONTEXT = Context()
