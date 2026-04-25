from __future__ import annotations

"""Mini-SGLang 风格的 MoE 高层逻辑。

这一层尽量保持“读起来像算法流程”：
1. router_logits -> topk_weights / topk_ids
2. 根据 topk_ids 做按 expert 的重排与 padding
3. 第一次 grouped GEMM 计算 gate/up
4. 激活函数
5. 第二次 grouped GEMM 计算 down
6. 对 top-k 结果做 sum-reduce

真正的 Triton kernel 定义不放在这里，而是下沉到 kernels/triton/。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.kernels.moe_impl import (
    fused_moe_kernel_triton,
    moe_align_block_size_triton,
    moe_sum_reduce_triton,
)
from nanovllm.kernels.triton.fused_moe import fused_topk_softmax_kernel

from .base import BaseMoeBackend


def _next_power_of_two(n: int) -> int:
    """返回 >= n 的最小 2 次幂。

    Triton kernel 的 tile 大小通常偏好 2 次幂，这里用它给 experts 维做 padding。
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pick_block_e(num_experts: int) -> int:
    """为 topk kernel 选择 experts 维的 tile 大小。"""
    return min(max(_next_power_of_two(num_experts), 16), 1024)


def _pick_num_warps(block_e: int) -> int:
    """根据 tile 大小选择 Triton 的 num_warps。"""
    if block_e <= 32:
        return 1
    if block_e <= 128:
        return 2
    if block_e <= 256:
        return 4
    return 8


def fused_topk(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行 MoE router 的 `softmax + topk`。

    这里和 mini-sglang 的 fused_topk 对应：
    - 输入是 router logits
    - 输出是每个 token 的 top-k expert id 与对应权重

    为什么还传 hidden_states：
    - 主要是保持接口和 mini-sglang 一致
    - 同时顺手检查 token 维是否对齐
    """
    assert hidden_states.shape[0] == router_logits.shape[0], "Number of tokens mismatch"
    token_num, _ = hidden_states.shape
    num_experts = router_logits.shape[1]

    block_e = _pick_block_e(num_experts)
    if num_experts > block_e:
        raise ValueError(f"Unsupported num_experts={num_experts}, max supported is {block_e}")

    topk_weights = torch.empty(
        token_num, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        token_num, topk, dtype=torch.int32, device=hidden_states.device
    )

    # 一个 program 处理一整行 gating logits。
    # 行内做 softmax，并重复 top-k 次取最大值。
    fused_topk_softmax_kernel[(token_num,)](
        router_logits,
        topk_weights,
        topk_ids,
        router_logits.stride(0),
        router_logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_experts=num_experts,
        top_k=topk,
        renormalize=renormalize,
        BLOCK_E=block_e,
        num_warps=_pick_num_warps(block_e),
    )
    return topk_weights, topk_ids


def _get_default_config(m_tokens: int, num_experts: int) -> dict[str, int]:
    """给 Triton grouped GEMM 选一组保守配置。

    这里没有做复杂 autotune，只做了一个简单 heuristic：
    - token 很少、expert 很多时，M 方向 tile 取小一点
    - 否则用更通用的大 tile
    """
    if m_tokens <= num_experts:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """应用专家 MLP 的激活。

    w1 的输出布局固定为 `[gate, up]` 两半拼接，
    所以这里先 split，再做 `silu(gate) * up`。
    """
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * up
    if activation == "gelu":
        return F.gelu(gate) * up
    raise ValueError(f"Unsupported MoE activation: {activation}")


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """执行整个 experts 计算主干。

    注意这里假设：
    - hidden_states: [T, D]
    - w1: [E, 2 * FFN, D]
    - w2: [E, D, FFN]
    - topk_ids / topk_weights: [T, K]

    这是一个非常典型的“先重排 token，再做 grouped GEMM”的 MoE 流程。
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_tokens, _ = hidden_states.shape
    num_experts, intermediate_twice, _ = w1.shape
    config = _get_default_config(num_tokens, num_experts)

    # 这一步把 [T, K] 的 top-k expert 选择，变成：
    # - sorted_token_ids: token-topk 对按 expert 分桶后的顺序
    # - expert_ids: 每个 M-tile 属于哪个 expert
    # - num_tokens_post_padded: 对齐到 block_size 后的总 token 数
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_triton(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    # 第一次 grouped GEMM：计算 gate/up。
    # 输出形状是 [T, K, 2*FFN]，因为每个 token 会被复制到 top-k 个 expert 上。
    intermediate_cache1 = torch.empty(
        (num_tokens, topk_ids.shape[1], intermediate_twice),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fused_moe_kernel_triton(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=apply_router_weight_on_input,
        top_k=topk_ids.shape[1],
        config=config,
        compute_type=hidden_states.dtype,
    )

    # 激活函数按 `[T*K, 2*FFN]` 来看最自然，所以这里先 view 成二维。
    intermediate_cache2 = _apply_activation(
        intermediate_cache1.view(num_tokens * topk_ids.shape[1], intermediate_twice),
        activation,
    )

    # 第二次 grouped GEMM：计算 down。
    intermediate_cache3 = torch.empty(
        (num_tokens, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,
        config=config,
        compute_type=hidden_states.dtype,
    )

    # 最后沿 top-k 维做 reduce，恢复成普通 MLP 输出形状 [T, D]。
    output = torch.empty(
        (num_tokens, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    moe_sum_reduce_triton(intermediate_cache3.contiguous(), output)
    return output


class FusedMoe(BaseMoeBackend):
    """当前 nano-vllm 使用的默认 MoE backend。"""

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        """创建 TP-aware 的专家权重。

        对于 Qwen3 MoE 的 MLP：
        - gate_proj / up_proj 是列并行，对应 w1 的第 0/1 半段
        - down_proj 是行并行，对应 w2

        所以这里直接把堆叠后的专家权重做成 TP shard 形状。
        """
        tp_size = dist.get_world_size()
        assert intermediate_size % tp_size == 0, (
            f"moe_intermediate_size={intermediate_size} must be divisible by tp_size={tp_size}"
        )

        intermediate_size_per_partition = intermediate_size // tp_size
        layer.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size)
        )
        layer.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition)
        )

        def _w1_loader(param, loaded_weight, expert_id, loaded_shard_id):
            # gate_proj / up_proj 原始权重是 [FFN, D]，列并行后按 dim=0 切。
            local_loaded = loaded_weight.chunk(tp_size, dim=0)[dist.get_rank()].contiguous()
            offset = loaded_shard_id * intermediate_size_per_partition
            param.data[expert_id, offset : offset + intermediate_size_per_partition].copy_(
                local_loaded
            )

        def _w2_loader(param, loaded_weight, expert_id):
            # down_proj 原始权重是 [D, FFN]，行并行后按 dim=1 切。
            local_loaded = loaded_weight.chunk(tp_size, dim=1)[dist.get_rank()].contiguous()
            param.data[expert_id].copy_(local_loaded)

        layer.w1.weight_loader = _w1_loader
        layer.w2.weight_loader = _w2_loader

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
        """执行 MoE 前向。

        注意这里拆成两步：
        1. router：算出 top-k 路由结果
        2. experts：真正按 expert 计算 MLP
        """
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        return fused_experts(
            hidden_states,
            layer.w1,
            layer.w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
