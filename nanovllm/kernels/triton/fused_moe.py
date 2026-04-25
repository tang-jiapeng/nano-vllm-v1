from __future__ import annotations

"""MoE 相关的 Triton kernel 定义。

阅读这个文件时，可以把它分成三部分来看：
1. `moe_sum_reduce_kernel`
   - 负责把 [T, K, D] 沿 K 维求和
2. `fused_moe_kernel`
   - 真正的 grouped GEMM experts kernel
3. `fused_topk_softmax_kernel` 与 `moe_align_block_size_stage*`
   - 分别负责 router 的 top-k 和 token 重排

和 mini-sglang 类似，这里只放“纯 kernel”；
真正怎么 launch、怎么分配 buffer、怎么组织张量，由上层 wrapper 负责。
"""

import triton
import triton.language as tl


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    """把 [T, K, D] reduce 成 [T, D]。

    一个 program 负责：
    - 一小段 token（BLOCK_M）
    - 一小段 hidden dim（BLOCK_DIM）

    对这段区域里的每个 token，都沿 top-k 维把 K 份 expert 输出加起来。
    """
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)
    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)
    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1,
                mask=offs_dim < dim_end,
                other=0.0,
            )
            accumulator += tmp
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """MoE experts 的 grouped GEMM kernel。

    直观理解：
    - 输入 A 不是普通 [T, D] 的 token 顺序
    - 而是通过 `sorted_token_ids + expert_ids` 定义出来的“按 expert 分桶后”的逻辑矩阵
    - 每个 M-tile 都只属于一个 expert，因此同一个 tile 会只读取该 expert 的权重

    这里的 `top_k` 只影响 `offs_token // top_k` 这一步：
    - `sorted_token_ids` 里存的是“token-topk 对”的展平索引
    - 除以 top_k 后才能回到原始 token 行号
    """
    pid = tl.program_id(axis=0)

    # 这里的 grouped ordering 主要是为了提升同一 expert 权重在 L2 / cache 中的复用。
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 通过 `sorted_token_ids` 做间接寻址：
    # 当前 tile 里的每一行，真实对应哪个 token-topk 对，是运行时决定的。
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
            )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        # 有些实现喜欢把 router weight 乘在输入上，有些喜欢乘在输出上。
        # 这里用一个 flag 统一支持两种方式。
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0.0
        )
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_topk_softmax_kernel(
    gating_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    renormalize: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """单行 `softmax + top-k` kernel。

    一个 program 负责一行 gating logits（也就是一个 token）。
    流程是：
    1. 读整行 expert logits
    2. 做 softmax
    3. 重复 top-k 次取最大值
    4. 如有需要，对 top-k 权重再归一化

    这版实现的思路更偏教学和可读性，
    没有像 vLLM CUDA kernel 那样极致追求 warp 内手写规约。
    """
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts

    logits = tl.load(
        gating_ptr + pid * stride_gm + offs_e * stride_ge,
        mask=mask_e,
        other=float("-inf"),
    ).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    probs = tl.exp(logits - row_max)
    probs = tl.where(mask_e, probs, 0.0)
    denom = tl.sum(probs, axis=0)
    probs = probs / denom

    selected_sum = 0.0
    current = probs
    for k_idx in range(top_k):
        max_val = tl.max(current, axis=0)
        max_idx = tl.argmax(current, axis=0)
        tl.store(topk_weights_ptr + pid * stride_wm + k_idx * stride_wk, max_val)
        tl.store(
            topk_ids_ptr + pid * stride_im + k_idx * stride_ik, max_idx.to(tl.int32)
        )
        selected_sum += max_val
        current = tl.where(offs_e == max_idx, float("-inf"), current)

    if renormalize:
        denom_topk = tl.maximum(selected_sum, 1e-8)
        for k_idx in range(top_k):
            ptr = topk_weights_ptr + pid * stride_wm + k_idx * stride_wk
            val = tl.load(ptr)
            tl.store(ptr, val / denom_topk)


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_program: tl.constexpr,
):
    """Stage 1：按 expert 统计 token-topk 对的数量。"""
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_program
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_program):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    """Stage 2：对每个 expert 的计数做分段前缀和。"""
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    """Stage 3：把每个 expert 的 token 数向上对齐到 block_size，并生成 cumsum。"""
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_program: tl.constexpr,
):
    """Stage 4：写出最终的 `sorted_token_ids` 和 `expert_ids`。"""
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    # 先把当前 expert 对应的 block 标记出来。
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_program
    off_t = pid * num_experts

    # 再把属于当前 expert 的 token-topk 对，搬运到 padding 后的全局位置。
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_program, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)
