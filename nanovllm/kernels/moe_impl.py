from __future__ import annotations

"""MoE Triton kernel 的 Python 包装层。

这个文件的职责很像 mini-sglang 的 `kernel/moe_impl.py`：
它不定义真正的 Triton kernel，而是负责：
- 组织 launch 参数
- 选择 grid / dtype
- 分配必要的中间 buffer

这样做的好处是：
- `triton/fused_moe.py` 只保留 kernel 定义，读起来更纯粹
- 这里则专门处理 PyTorch tensor、stride、meta 参数这些工程细节
"""

import math

import torch
import triton
import triton.language as tl

from nanovllm.kernels.triton.fused_moe import (
    fused_moe_kernel,
    moe_align_block_size_stage1,
    moe_align_block_size_stage2,
    moe_align_block_size_stage3,
    moe_align_block_size_stage4,
    moe_sum_reduce_kernel,
)


def fused_moe_kernel_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, int],
    compute_type: torch.dtype,
) -> None:
    """调用 experts grouped GEMM Triton kernel。

    参数里最容易混淆的几项是：
    - `topk_ids`: 原始 [T, K] 的路由结果，只用于告诉 kernel `top_k` 和总元素数
    - `sorted_token_ids`: 已经按 expert 分桶后的 token-topk 顺序
    - `expert_ids`: 每个 M-tile 对应哪个 expert

    这三个量共同定义了“如何把一批 token 发给多个 expert 去算”。
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # grid 的两个维度本质上对应输出矩阵 C 的：
    # - M 方向 tile 数
    # - N 方向 tile 数
    grid = lambda meta: (
        triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
        * triton.cdiv(b.shape[1], meta["BLOCK_SIZE_N"]),
    )
    k_dim = b.shape[2]
    even_ks = k_dim % config["BLOCK_SIZE_K"] == 0
    out_dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16

    fused_moe_kernel[grid](
        a,
        b,
        c,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        b.shape[1],
        k_dim,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=out_dtype,
        even_Ks=even_ks,
        **config,
    )


def moe_sum_reduce_triton(input_tensor: torch.Tensor, output: torch.Tensor) -> None:
    """沿 top-k 维把 [T, K, D] reduce 成 [T, D]。

    这一步的数学意义非常直白：
    每个 token 会经过 top-k 个 expert，得到 K 份输出；
    这里把这 K 份输出按路由权重加总，恢复成标准的隐藏状态。
    """
    assert input_tensor.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input_tensor.shape
    block_m = 1
    block_dim = 2048
    num_stage = 1
    num_warps = 8
    grid = (
        triton.cdiv(token_num, block_m),
        triton.cdiv(hidden_dim, block_dim),
    )

    moe_sum_reduce_kernel[grid](
        input_tensor,
        *input_tensor.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        NUM_STAGE=num_stage,
        num_warps=num_warps,
    )


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 [T, K] 的 top-k 路由结果重排成适合 grouped GEMM 的布局。

    为什么要做这一步：
    - 如果直接按原始 token 顺序做专家计算，每个 expert 的 token 是稀疏分布的
    - grouped GEMM 更喜欢“同一个 expert 的 token 挨在一起”
    - 同时还希望每个 expert 的 token 数按 block_size 对齐，避免 tile 尾部处理复杂

    返回值的含义：
    - `sorted_token_ids`: 重排后的 token-topk 索引
    - `expert_ids`: 每个 block 属于哪个 expert
    - `num_tokens_post_padded`: padding 之后的总 token 数
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_padded = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.zeros(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    tokens_per_program = math.ceil(topk_ids.numel() / num_experts)

    # 这四个 stage 对应 mini-sglang benchmark 里的 staged Triton 版本：
    # 1. 统计每个 expert 的 token 数
    # 2. 对计数做前缀和
    # 3. 计算 padding 后的 cumsum
    # 4. 写出 sorted_token_ids 和 expert_ids
    moe_align_block_size_stage1[(num_experts,)](
        topk_ids,
        tokens_cnts,
        num_experts=num_experts,
        numel=topk_ids.numel(),
        tokens_per_program=tokens_per_program,
    )
    moe_align_block_size_stage2[(num_experts,)](
        tokens_cnts,
        num_experts=num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        num_tokens_post_padded,
        tokens_cnts,
        cumsum_buffer,
        num_experts=num_experts,
        block_size=block_size,
    )
    moe_align_block_size_stage4[(num_experts,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum_buffer,
        num_experts=num_experts,
        block_size=block_size,
        numel=topk_ids.numel(),
        tokens_per_program=tokens_per_program,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded
