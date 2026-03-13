"""
词嵌入层和语言模型头实现

本模块实现了支持张量并行的词嵌入层和输出层：
1. VocabParallelEmbedding - 词嵌入层（token ID -> 向量）
2. ParallelLMHead - 语言模型头（向量 -> logits）

张量并行策略：
- 在词表维度上分片（vocab_size维度）
- 每个GPU持有部分词表
- 通过掩码和all-reduce处理跨GPU的词嵌入

使用场景：
- Token嵌入（将token ID转换为向量表示）
- 输出层（将隐藏状态转换为logits）
- 词表并行的分布式训练和推理

参考资料：
- Megatron-LM的词嵌入实现
- 张量并行的词汇分布策略
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行嵌入层，在vocab维度分片，通过mask和all-reduce聚合各GPU结果。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size

        # 当前GPU负责的词表范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中切分当前GPU的分片并加载。"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """token ID -> embedding向量，tensor parallel时通过mask + all-reduce聚合。"""
        if self.tp_size > 1:
            # mask标记属于当前GPU的token，将全局ID转为本地ID
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """并行语言模型头，将hidden state映射为logits，支持vocab parallel和prefill优化。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """hidden state -> logits，通过 seq_need_compute_logits 选择需要 logits 的序列。"""
        context = get_context()

        # 仅在 prefill 且非 speculative verify 阶段取每个序列最后位置。
        # decode / speculative verify 路径需要保留全部位置的 hidden states。
        if context.is_prefill and (not context.is_speculative):
            last_indices = context.cu_seqlens_q[1:] - 1
            if (
                context.seq_need_compute_logits is not None
                and context.seq_need_compute_logits.numel()
            ):
                last_indices = last_indices[context.seq_need_compute_logits]
            x = x[last_indices].contiguous()

        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
