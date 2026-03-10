"""支持 tensor parallelism 的线性层实现，包括列并行、行并行及 QKV 并行等策略。"""

from abc import abstractmethod

import torch
import torch.distributed as dist
from torch import nn

from nanovllm.layers.quantization.awq_config import AWQConfig
from nanovllm.layers.quantization.linear import (
    AWQLinearMethod,
    LinearMethodBase,
    UnquantizedLinearMethod,
)


def divide(numerator, denominator):
    """整除，不可整除时抛出异常。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """线性层基类，提供 tensor parallelism 配置和权重加载接口。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        awq_config: AWQConfig | None = None,
    ):
        super().__init__()

        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 根据是否量化选择策略，基类对量化细节无感知
        if awq_config is None:
            self.linear_method: LinearMethodBase = UnquantizedLinearMethod()
        else:
            self.linear_method = AWQLinearMethod(awq_config)

        self.linear_method.create_weights(self, input_size, output_size)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.bias_weight_loader
        else:
            self.register_parameter("bias", None)

    def bias_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """bias 加载：ColumnParallel 按输出维切分，RowParallel / Replicated 直接复制。"""
        if self.tp_dim != 0 or self.tp_size == 1:
            # Replicated / RowParallel / 单卡：直接复制完整 bias
            param.data.copy_(loaded_weight)
            return
        # ColumnParallel (tp_dim=0)：bias 按输出维切分
        shard_size = param.data.shape[0]
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    @abstractmethod
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """子类实现：将外部权重加载到参数（切分 / 复制）。"""
        ...

    def _forward_base(
        self, x: torch.Tensor, row_parallel_bias: bool = False
    ) -> torch.Tensor:
        bias = self.bias
        if row_parallel_bias and self.tp_rank != 0:
            bias = None
        return self.linear_method.apply(self, x, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """无并行的线性层，所有 GPU 持有完整权重副本。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        awq_config: AWQConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, awq_config=awq_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制完整权重。"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_base(x)


class ColumnParallelLinear(LinearBase):
    """列并行线性层，按输出维度分片权重到各 GPU。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        awq_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            tp_dim=0,
            awq_config=awq_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中切分当前 GPU 对应的列分片。

        普通权重 shape=[N, K]，输出维在 dim=0。
        AWQ 权重 shape=[K, N//p] / [K//g, N]，输出维在 dim=1。
        """
        if isinstance(self.linear_method, AWQLinearMethod):
            # AWQ: 输出维在 dim=1
            shard_size = param.data.shape[1]
            start_idx = self.tp_rank * shard_size
            param.data.copy_(loaded_weight.narrow(1, start_idx, shard_size))
        else:
            # 普通权重: 输出维在 tp_dim=0
            shard_size = param.data.shape[self.tp_dim]
            start_idx = self.tp_rank * shard_size
            param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_base(x)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并多个列并行投影为一个层（如 SwiGLU 的 gate 和 up 投影）。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        awq_config: AWQConfig | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, awq_config=awq_config)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        """按 loaded_shard_id 加载对应投影的列分片。"""
        if isinstance(self.linear_method, AWQLinearMethod):
            p = self.linear_method.config.pack_factor
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            if param.data.dtype == torch.int32:
                # qweight/qzeros: 需要 pack_factor 折算
                shard_offset //= p
                shard_size //= p
            # AWQ 输出维在 dim=1
            local_loaded = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, shard_offset, shard_size).copy_(local_loaded)
            return

        # 普通权重：输出维在 tp_dim
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        local_loaded = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.data.narrow(self.tp_dim, shard_offset, shard_size).copy_(local_loaded)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 合并列并行层，支持 GQA（Q 与 KV 头数可不同）。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        awq_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, awq_config=awq_config)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        """按 loaded_shard_id ("q"/"k"/"v") 加载对应投影的分片。"""
        assert loaded_shard_id in ("q", "k", "v")

        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size

        if isinstance(self.linear_method, AWQLinearMethod):
            p = self.linear_method.config.pack_factor
            awq_offset = shard_offset
            awq_size = shard_size
            if param.data.dtype == torch.int32:
                awq_offset //= p
                awq_size //= p
            # AWQ 输出维在 dim=1
            local_loaded = loaded_weight.chunk(self.tp_size, 1)[self.tp_rank]
            param.data.narrow(1, awq_offset, awq_size).copy_(local_loaded)
            return

        # 普通权重：输出维在 tp_dim
        local_loaded = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param.data.narrow(self.tp_dim, shard_offset, shard_size).copy_(local_loaded)


class RowParallelLinear(LinearBase):
    """行并行线性层，按输入维度分片权重，通过 all-reduce 聚合输出。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        awq_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            tp_dim=1,
            awq_config=awq_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从完整权重中切分当前 GPU 对应的行分片。

        普通权重 shape=[N, K]，输入维在 dim=1（tp_dim）。
        AWQ 权重 shape=[K, N//p] / [K//g, N]，输入维在 dim=0。
        """
        if isinstance(self.linear_method, AWQLinearMethod):
            # AWQ: 输入维在 dim=0
            shard_size = param.data.shape[0]
            start_idx = self.tp_rank * shard_size
            param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))
        else:
            # 普通权重: 输入维在 tp_dim=1
            shard_size = param.data.size(self.tp_dim)
            start_idx = self.tp_rank * shard_size
            param.data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """线性变换后通过 all-reduce 聚合各 GPU 结果。"""
        y = self._forward_base(x, row_parallel_bias=True)

        if self.tp_size > 1:
            dist.all_reduce(y)

        return y
