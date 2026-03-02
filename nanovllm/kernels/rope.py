import torch
import triton
import triton.language as tl


def _rope_configs():
    """生成 autotune 的候选配置矩阵。"""
    configs = []
    for block_t in [1, 2, 4]:
        for block_h in [1, 2, 4, 8, 10, 20, 40]:
            for nw in [1, 2, 4]:
                configs.append(
                    triton.Config(
                        {"BLOCK_T": block_t, "BLOCK_H": block_h}, num_warps=nw
                    )
                )
    return configs


@triton.autotune(
    configs=_rope_configs(),
    key=["num_tokens", "num_q_heads", "num_kv_heads", "half_dim"],
    restore_value=["Q_ptr", "K_ptr"],
)
@triton.jit
def fused_qk_rope_kernel(
    Q_ptr,  # [num_tokens, num_q_heads, head_dim]
    K_ptr,  # [num_tokens, num_kv_heads, head_dim]
    positions_ptr,  # [num_tokens]
    cos_sin_ptr,  # [max_seq_len, 1, head_dim] — 前半 cos, 后半 sin
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_cos_pos,
    stride_cos_dim,
    num_tokens,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,  # >= half_dim 的最小 2 的幂次
    BLOCK_T: tl.constexpr,  # 每个 program 沿 token 维度处理的 token 数
    BLOCK_H: tl.constexpr,  # 每个 program 沿 head 维度处理的 head 数
):
    """
    Fused QK-RoPE Triton Kernel (2D Grid).

    grid = (cdiv(num_tokens, BLOCK_T), cdiv(num_q_heads + num_kv_heads, BLOCK_H))

    设计要点:
      1. 2D grid: program_id(0) 切分 token, program_id(1) 切分 head,
         在两个维度上同时提供并行度。
      2. cos/sin 在 token 级加载一次, 被 BLOCK_H 个 head 共享复用,
         最大限度减少 cos_sin_cache 的显存流量。
      3. Q 和 K 按连续 head 索引统一编号 (0..total_heads-1),
         通过编译期常量 num_q_heads 区分, 两段各用独立 stride 和指针。
      4. autotune 自动搜索最优 (BLOCK_T, BLOCK_H, num_warps) 组合:
         - 小 token 数 → BLOCK_T=1, BLOCK_H=1~2, 最高 GPU 并行度
         - 大 token 数 → BLOCK_T=2~4, BLOCK_H=20~40, 最优 cos/sin 摊销
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    token_start = pid_t * BLOCK_T
    head_start = pid_h * BLOCK_H
    total_heads = num_q_heads + num_kv_heads

    dim_offs = tl.arange(0, BLOCK_D)
    dim_mask = dim_offs < half_dim

    for t in range(BLOCK_T):
        token_idx = token_start + t
        if token_idx < num_tokens:
            # 每个 token 加载一次 cos/sin，被 BLOCK_H 个 head 共享
            pos = tl.load(positions_ptr + token_idx)
            cos_base = cos_sin_ptr + pos * stride_cos_pos
            cos = tl.load(cos_base + dim_offs * stride_cos_dim, mask=dim_mask).to(
                tl.float32
            )
            sin = tl.load(
                cos_base + (half_dim + dim_offs) * stride_cos_dim, mask=dim_mask
            ).to(tl.float32)

            # 处理本 block 负责的 BLOCK_H 个 head
            for i in range(BLOCK_H):
                head_idx = head_start + i
                if head_idx < total_heads:
                    if head_idx < num_q_heads:
                        # Query head
                        base = (
                            Q_ptr + token_idx * stride_q_tok + head_idx * stride_q_head
                        )
                        x = tl.load(base + dim_offs * stride_q_dim, mask=dim_mask).to(
                            tl.float32
                        )
                        y = tl.load(
                            base + (half_dim + dim_offs) * stride_q_dim,
                            mask=dim_mask,
                        ).to(tl.float32)
                        tl.store(
                            base + dim_offs * stride_q_dim,
                            (x * cos - y * sin).to(Q_ptr.dtype.element_ty),
                            mask=dim_mask,
                        )
                        tl.store(
                            base + (half_dim + dim_offs) * stride_q_dim,
                            (y * cos + x * sin).to(Q_ptr.dtype.element_ty),
                            mask=dim_mask,
                        )
                    else:
                        # Key head
                        k_idx = head_idx - num_q_heads
                        base = K_ptr + token_idx * stride_k_tok + k_idx * stride_k_head
                        x = tl.load(base + dim_offs * stride_k_dim, mask=dim_mask).to(
                            tl.float32
                        )
                        y = tl.load(
                            base + (half_dim + dim_offs) * stride_k_dim,
                            mask=dim_mask,
                        ).to(tl.float32)
                        tl.store(
                            base + dim_offs * stride_k_dim,
                            (x * cos - y * sin).to(K_ptr.dtype.element_ty),
                            mask=dim_mask,
                        )
                        tl.store(
                            base + (half_dim + dim_offs) * stride_k_dim,
                            (y * cos + x * sin).to(K_ptr.dtype.element_ty),
                            mask=dim_mask,
                        )


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
    """
    Fused QK-RoPE, 支持 GQA (Grouped-Query Attention)
    原地修改 q 和 k

    Args:
        q: [num_tokens, num_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        positions: [num_tokens]
        cos_sin_cache: [max_seq_len, 1, head_dim] — 前半 cos, 后半 sin (float32)
    """
    assert q.is_contiguous() and k.is_contiguous() and positions.is_contiguous()

    num_tokens, num_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    total_heads = num_heads + num_kv_heads
    half_dim = head_dim // 2
    BLOCK_D = triton.next_power_of_2(half_dim)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(num_tokens, meta["BLOCK_T"]),
        triton.cdiv(total_heads, meta["BLOCK_H"]),
    )

    fused_qk_rope_kernel[grid](
        q,
        k,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(2),
        num_tokens=num_tokens,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        half_dim=half_dim,
        BLOCK_D=BLOCK_D,
    )

    return q, k
