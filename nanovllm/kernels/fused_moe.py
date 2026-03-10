import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.kernels.awq_gemm import awq_gemm_triton
from nanovllm.utils.context import get_context


# ────────────────────────────────────────────────────────────────────
# Triton fused MoE GEMM kernel（仅 FP prefill 路径使用）
# ────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def fused_moe_kernel(
    A_ptr,  # [M_total, K]  所有 token（已按 expert 排序）
    B_ptr,  # [E, N, K]     堆叠的 expert 权重
    C_ptr,  # [M_total, N]  输出
    # ── 路由信息 ──
    sorted_token_ids_ptr,  # [M_total]     排序后的 token 在原 hidden 中的索引
    expert_ids_ptr,  # [num_tiles_m] 每个 M-tile 属于哪个 expert
    num_valid_tokens_ptr,  # [num_tiles_m] 每个 tile 中有效 token 数
    # ── 维度 ──
    N: tl.constexpr,  # 输出维度（W1: 2*FFN, W2: D）
    K: tl.constexpr,  # 输入维度（W1: D, W2: FFN）
    stride_am,  # A 的 M 维 stride
    stride_ak,  # A 的 K 维 stride
    stride_be,  # B 的 E 维 stride
    stride_bn,  # B 的 N 维 stride
    stride_bk,  # B 的 K 维 stride
    stride_cm,  # C 的 M 维 stride
    stride_cn,  # C 的 N 维 stride
    # ── Tile 大小 ──
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused MoE GEMM Kernel.

    每个 program (pid_m, pid_n) 处理输出矩阵的一个 [BLOCK_M, BLOCK_N] tile.
    通过 expert_ids[pid_m] 查找当前 tile 属于哪个 expert，
    从而选择正确的权重 B[expert_id]

    Grid: (num_tiles_m, ceil(N / BLOCK_N))

    等价于:
        for each tile (m, n):
            expert = expert_ids[m]
            C[m*BM:(m+1)*BM, n*BN:(n+1)*BN] = A[m*BM:(m+1)*BM] @ B[expert].T
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ── 查找当前 tile 属于哪个 expert ──
    expert_id = tl.load(expert_ids_ptr + pid_m)
    num_valid = tl.load(num_valid_tokens_ptr + pid_m)

    # ── 计算偏移 ──
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A 指针: A[offs_m, 0:K]
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B 指针: B[expert_id, offs_n, 0:K]
    b_ptrs = (
        B_ptr
        + expert_id * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )

    # ── 累加器 ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── 沿 K 维分块累加 ──
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # mask: K 维边界
        mask_k = k_offs < K
        # mask: M 维有效 token
        mask_m = tl.arange(0, BLOCK_M) < num_valid

        # 加载 A tile: [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # 加载 B tile: [BLOCK_N, BLOCK_K] (B 布局是 [N, K])
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & mask_k[None, :], other=0.0)
        # GEMM: [BM, BK] @ [BK, BN] → [BM, BN]
        acc += tl.dot(a, tl.trans(b))

        # 推进 K 维指针
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── 写回 C ──
    mask_m = tl.arange(0, BLOCK_M) < num_valid
    mask_n = offs_n < N
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :]
    )


def _prepare_expert_mapping(
    sorted_expert_ids: torch.Tensor,  # [T*K] 已按 expert 排序的 expert id
    tokens_per_expert: torch.Tensor,  # [E] 每个 expert 的 token 数
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    为 Triton kernel 准备 expert_ids 和 num_valid_tokens 映射。

    Returns:
        expert_ids: [num_tiles_m]        每个 M-tile 属于哪个 expert
        num_valid:  [num_tiles_m]        每个 M-tile 中的有效 token 数
    """
    E = tokens_per_expert.shape[0]
    expert_ids_list = []
    num_valid_list = []

    for e in range(E):
        n = tokens_per_expert[e].item()
        if n == 0:
            continue
        full_tiles = n // block_m
        remainder = n % block_m
        expert_ids_list.extend([e] * full_tiles)
        num_valid_list.extend([block_m] * full_tiles)
        if remainder > 0:
            expert_ids_list.append(e)
            num_valid_list.append(remainder)

    device = sorted_expert_ids.device
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=device)
    num_valid = torch.tensor(num_valid_list, dtype=torch.int32, device=device)
    return expert_ids, num_valid


def triton_fused_moe_gemm(
    A: torch.Tensor,  # [M_total, K]
    B: torch.Tensor,  # [E, N, K]
    expert_ids: torch.Tensor,  # [num_tiles_m]
    num_valid: torch.Tensor,  # [num_tiles_m]
) -> torch.Tensor:
    """
    调用 Triton fused_moe_kernel 执行所有 expert 的 GEMM

    等价于:
        for tile_m, expert in enumerate(expert_ids):
            C[tile_m*BM:(tile_m+1)*BM] = A[tile_m*BM:(tile_m+1)*BM] @ B[expert].T
    """
    M_total = A.shape[0]
    E, N, K = B.shape
    num_tiles_m = expert_ids.shape[0]

    C = torch.empty(M_total, N, dtype=A.dtype, device=A.device)

    def grid(meta):
        return num_tiles_m, triton.cdiv(N, meta["BLOCK_N"])

    fused_moe_kernel[grid](
        A,
        B,
        C,
        torch.arange(M_total, device=A.device),  # sorted_token_ids (identity)
        expert_ids,
        num_valid,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
    )
    return C


def _route_tokens(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
):
    """MoE 路由：softmax + topk + 按 expert 排序（不兼容 CUDA Graph）。"""
    T = hidden_states.shape[0]
    E = router_logits.shape[-1]

    routing_weights = torch.softmax(router_logits.float(), dim=-1)  # [T, E]
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)  # [T, K]
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)

    flat_ids = topk_ids.flatten()  # [T*K]
    flat_tok = (
        torch.arange(T, device=hidden_states.device)
        .unsqueeze(1)
        .expand(T, top_k)
        .flatten()
    )  # [T*K]
    flat_w = topk_weights.flatten()  # [T*K]

    sort_idx = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sort_idx]
    sorted_tok = flat_tok[sort_idx]
    sorted_w = flat_w[sort_idx]
    unsorter = torch.argsort(sort_idx)

    tokens_per_expert = torch.bincount(sorted_ids, minlength=E)
    dispatched = hidden_states[sorted_tok]  # [T*K, D]
    return dispatched, sorted_ids, sorted_w, unsorter, tokens_per_expert


def _route_topk_weights(
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    dtype: torch.dtype,
):
    """路由：仅计算 topk_weights 和 topk_ids（CUDA Graph 兼容）。"""
    routing_weights = torch.softmax(router_logits.float(), dim=-1)  # [T, E]
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)  # [T, K]
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    return topk_weights, topk_ids


def _pick_split_k(m: int) -> int:
    if m <= 8:
        return 8
    if m <= 32:
        return 4
    return 1


# ────────────────────────────────────────────────────────────────────
# Prefill 路径：token dispatch + 只算被选中的 expert
#   不兼容 CUDA Graph，使用 bincount / nonzero / .item()
#   适合大 batch prefill，避免对未被选中的 expert 做无效 GEMM
# ────────────────────────────────────────────────────────────────────


def _fused_moe_fp_prefill(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> torch.Tensor:
    """FP16/BF16 prefill MoE（token dispatch，不兼容 CUDA Graph）。"""
    T, D = hidden_states.shape
    E, FFN2, _ = w1.shape
    FFN = FFN2 // 2

    dispatched, sorted_ids, sorted_w, unsorter, tokens_per_expert = _route_tokens(
        hidden_states, router_logits, top_k, norm_topk_prob
    )

    BLOCK_M = 16
    expert_ids, num_valid = _prepare_expert_mapping(
        sorted_ids, tokens_per_expert, BLOCK_M
    )

    intermediate = triton_fused_moe_gemm(dispatched, w1, expert_ids, num_valid)
    gate = intermediate[:, :FFN]
    up = intermediate[:, FFN:]
    inter2 = F.silu(gate) * up
    output_flat = triton_fused_moe_gemm(inter2, w2, expert_ids, num_valid)

    output_flat = output_flat * sorted_w.unsqueeze(-1)
    output_flat = output_flat[unsorter].view(T, top_k, D)
    return output_flat.sum(dim=1)


def _fused_moe_awq_prefill(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    w1_qweight: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_scales: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    """AWQ prefill MoE（token dispatch，不兼容 CUDA Graph）。"""
    T, D = hidden_states.shape
    E = router_logits.shape[-1]
    pack = 8
    FFN = w2_qweight.shape[1]
    qn = FFN // pack

    dispatched, sorted_ids, sorted_w, unsorter, tokens_per_expert = _route_tokens(
        hidden_states, router_logits, top_k, norm_topk_prob
    )

    output_sorted = torch.zeros_like(dispatched)
    prefix = torch.cumsum(tokens_per_expert, dim=0)
    active_experts = torch.nonzero(tokens_per_expert, as_tuple=False).flatten().tolist()

    for e in active_experts:
        end = int(prefix[e].item())
        start = int((prefix[e] - tokens_per_expert[e]).item())
        x = dispatched[start:end]
        m = x.shape[0]
        if m == 0:
            continue

        split_k = _pick_split_k(m)

        gate = awq_gemm_triton(
            x,
            w1_qweight[e, :, :qn].contiguous(),
            w1_scales[e, :, :FFN].contiguous(),
            w1_qzeros[e, :, :qn].contiguous(),
            split_k_iters=split_k,
        )
        up = awq_gemm_triton(
            x,
            w1_qweight[e, :, qn:].contiguous(),
            w1_scales[e, :, FFN:].contiguous(),
            w1_qzeros[e, :, qn:].contiguous(),
            split_k_iters=split_k,
        )

        inter = F.silu(gate) * up
        out = awq_gemm_triton(
            inter,
            w2_qweight[e].contiguous(),
            w2_scales[e].contiguous(),
            w2_qzeros[e].contiguous(),
            split_k_iters=split_k,
        )
        output_sorted[start:end] = out.to(hidden_states.dtype)

    output_sorted = output_sorted * sorted_w.unsqueeze(-1)
    output_flat = output_sorted[unsorter].view(T, top_k, D)
    return output_flat.sum(dim=1)


# ────────────────────────────────────────────────────────────────────
# Decode 路径：逐 expert 循环 + mask weight
#   完全兼容 CUDA Graph（所有操作 shape 固定，无 CPU-GPU 同步）
#   decode 时 T 很小，开销可接受
# ────────────────────────────────────────────────────────────────────


def _fused_moe_fp_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
) -> torch.Tensor:
    """FP16/BF16 decode MoE（CUDA Graph 兼容，逐 expert 循环）。"""
    T, D = hidden_states.shape
    E, FFN2, _ = w1.shape
    FFN = FFN2 // 2

    topk_weights, topk_ids = _route_topk_weights(
        router_logits, top_k, norm_topk_prob, hidden_states.dtype
    )

    output = torch.zeros(T, D, dtype=hidden_states.dtype, device=hidden_states.device)

    for e in range(E):
        mask = topk_ids == e  # [T, K]
        weight = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1)  # [T]

        intermediate = hidden_states @ w1[e].t()  # [T, 2*FFN]
        gate = intermediate[:, :FFN]
        up = intermediate[:, FFN:]
        inter2 = F.silu(gate) * up
        out = inter2 @ w2[e].t()  # [T, D]

        output += out * weight.unsqueeze(-1)

    return output


def _fused_moe_awq_decode(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    w1_qweight: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_scales: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    """AWQ decode MoE（CUDA Graph 兼容，逐 expert 循环）。"""
    T, D = hidden_states.shape
    E = router_logits.shape[-1]
    pack = 8
    FFN = w2_qweight.shape[1]
    qn = FFN // pack

    topk_weights, topk_ids = _route_topk_weights(
        router_logits, top_k, norm_topk_prob, hidden_states.dtype
    )

    output = torch.zeros(T, D, dtype=hidden_states.dtype, device=hidden_states.device)

    for e in range(E):
        mask = topk_ids == e  # [T, K]
        weight = (topk_weights * mask.to(topk_weights.dtype)).sum(dim=-1)  # [T]

        split_k = _pick_split_k(T)

        gate = awq_gemm_triton(
            hidden_states,
            w1_qweight[e, :, :qn].contiguous(),
            w1_scales[e, :, :FFN].contiguous(),
            w1_qzeros[e, :, :qn].contiguous(),
            split_k_iters=split_k,
        )
        up = awq_gemm_triton(
            hidden_states,
            w1_qweight[e, :, qn:].contiguous(),
            w1_scales[e, :, FFN:].contiguous(),
            w1_qzeros[e, :, qn:].contiguous(),
            split_k_iters=split_k,
        )

        inter = F.silu(gate) * up

        out = awq_gemm_triton(
            inter,
            w2_qweight[e].contiguous(),
            w2_scales[e].contiguous(),
            w2_qzeros[e].contiguous(),
            split_k_iters=split_k,
        )

        output += out.to(hidden_states.dtype) * weight.unsqueeze(-1)

    return output


def fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool = True,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    w1_qweight: torch.Tensor | None = None,
    w1_qzeros: torch.Tensor | None = None,
    w1_scales: torch.Tensor | None = None,
    w2_qweight: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w2_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused MoE 前向传播，自动选择 prefill / decode 路径。

    - Prefill（cu_seqlens_q 存在，不走 CUDA Graph）：
      token dispatch 方式，只对被选中的 token-expert 对做计算。
      FP 路径使用 Triton fused_moe_kernel；AWQ 路径逐 active expert 调 awq_gemm。

    - Decode（CUDA Graph 兼容）：
      逐 expert 循环 + mask weight，所有操作 shape 固定，
      无 bincount / nonzero / .item() 等不兼容操作。
      decode 时 T 很小，开销可接受。

    Args:
        hidden_states: [T, D] 输入 hidden states
        router_logits: [T, E] 路由器输出
        top_k: 每个 token 激活的 expert 数
        norm_topk_prob: 是否归一化 top-k 权重
        w1: [E, 2*FFN, D] FP gate+up 权重（堆叠）
        w2: [E, D, FFN] FP down 权重（堆叠）
        w1_qweight/w1_qzeros/w1_scales: AWQ gate+up 量化权重
        w2_qweight/w2_qzeros/w2_scales: AWQ down 量化权重

    Returns:
        [T, D] MoE 层输出
    """
    is_prefill = get_context().is_prefill

    if w1 is not None and w2 is not None:
        if is_prefill:
            return _fused_moe_fp_prefill(
                hidden_states, w1, w2, router_logits, top_k, norm_topk_prob
            )
        return _fused_moe_fp_decode(
            hidden_states, w1, w2, router_logits, top_k, norm_topk_prob
        )

    awq_args = (w1_qweight, w1_qzeros, w1_scales, w2_qweight, w2_qzeros, w2_scales)
    if all(x is not None for x in awq_args):
        if is_prefill:
            return _fused_moe_awq_prefill(
                hidden_states,
                router_logits,
                top_k,
                norm_topk_prob,
                w1_qweight,
                w1_qzeros,
                w1_scales,
                w2_qweight,
                w2_qzeros,
                w2_scales,
            )
        return _fused_moe_awq_decode(
            hidden_states,
            router_logits,
            top_k,
            norm_topk_prob,
            w1_qweight,
            w1_qzeros,
            w1_scales,
            w2_qweight,
            w2_qzeros,
            w2_scales,
        )

    raise ValueError(
        "fused_moe requires either FP weights (w1/w2) or AWQ weights (qweight/qzeros/scales)"
    )
