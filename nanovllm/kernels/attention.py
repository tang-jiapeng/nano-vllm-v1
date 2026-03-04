import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["head_dim"],
)
@triton.jit
def flash_attn_prefill_varlen_kernel(
    Q,
    K,
    V,
    O,
    cu_seqlens,  # 序列长度的累加数组 [0, seq1_len, seq1_len+seq2_len, ...]
    scale,
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # (2D Grid)
    start_m = tl.program_id(0)
    off_h_seq = tl.program_id(1)

    off_h = off_h_seq % num_heads  # 当前 Query Head
    seq_idx = off_h_seq // num_heads  # 当前 Sequence

    # GQA: 映射 Q head -> KV head
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    seq_start = tl.load(cu_seqlens + seq_idx)
    seq_end = tl.load(cu_seqlens + seq_idx + 1)
    seq_len = seq_end - seq_start

    start_m_idx = start_m * BLOCK_M
    if start_m_idx >= seq_len:
        return

    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # 物理地址 = 序列起始偏移 + 块内偏移 + Head 偏移 + 维度偏移
    q_ptrs = (
        Q
        + (seq_start + offs_m[:, None]) * stride_q_tok
        + off_h * stride_q_head
        + offs_d[None, :] * stride_q_dim
    )
    # 加载 Q 块
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)  # Pre-scale Q，避免每个 K 块都乘 scale

    # 初始化 Online Softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # 初始化 K 和 V 的指针
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = (
        K
        + (seq_start + offs_n[None, :]) * stride_k_tok
        + kv_head_idx * stride_k_head
        + offs_d[:, None] * stride_k_dim
    )
    v_ptrs = (
        V
        + (seq_start + offs_n[:, None]) * stride_v_tok
        + kv_head_idx * stride_v_head
        + offs_d[None, :] * stride_v_dim
    )

    # ─── Phase 1: Non-Causal 区域 (全在对角线下方，无需任何 mask) ───
    # K[j] < start_m_idx ≤ min(Q[i])，因果条件始终满足
    # 省去: causal_mask 计算、tl.where、边界 mask、masked load
    for start_n in range(0, start_m_idx, BLOCK_N):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_k_tok
        v_ptrs += BLOCK_N * stride_v_tok

    # ─── Phase 2: Causal 对角区域 (需要因果 mask + 边界 mask) ───
    for start_n in range(start_m_idx, start_m_idx + BLOCK_M, BLOCK_N):
        mask_n = (start_n + offs_n) < seq_len
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k)

        causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

        k_ptrs += BLOCK_N * stride_k_tok
        v_ptrs += BLOCK_N * stride_v_tok

    # 写回 HBM
    acc = acc / l_i[:, None]
    o_ptrs = (
        O
        + (seq_start + offs_m[:, None]) * stride_o_tok
        + off_h * stride_o_head
        + offs_d[None, :] * stride_o_dim
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_atten_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:

    output = torch.empty_like(q)

    # 提取最大的序列长度，用于计算 M 维度的网格数量
    max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    num_seqs = cu_seqlens.shape[0] - 1

    # Grid 设计：(总切块数, 总Head数 * 总句子数)
    grid = lambda meta: (
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        num_heads * num_seqs,
    )

    flash_attn_prefill_varlen_kernel[grid](
        q,
        k,
        v,
        output,
        cu_seqlens,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    return output


# =============== PREFILL: Paged KV-Cache (Prefix Caching 支持) ===============


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["head_dim"],
)
@triton.jit
def flash_attn_prefill_paged_kernel(
    Q,
    O,
    K_cache,
    V_cache,
    block_tables_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    scale,
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    stride_kc_b,
    stride_kc_t,
    stride_kc_h,
    stride_kc_d,
    stride_vc_b,
    stride_vc_t,
    stride_vc_h,
    stride_vc_d,
    stride_bt_b,
    stride_bt_block,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Paged Prefill Flash Attention Kernel（支持 prefix caching）。

    Q 仅包含新增 token（cu_seqlens_q），K/V 从 paged KV cache 中
    通过 block_tables 间接寻址读取完整序列（cu_seqlens_k）。
    因果偏移 q_offset = k_len - q_len，保证正确的 causal mask。
    """
    start_m = tl.program_id(0)
    off_h_seq = tl.program_id(1)

    off_h = off_h_seq % num_heads
    seq_idx = off_h_seq // num_heads
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Q 范围（仅新增 token）
    q_seq_start = tl.load(cu_seqlens_q + seq_idx)
    q_seq_end = tl.load(cu_seqlens_q + seq_idx + 1)
    q_len = q_seq_end - q_seq_start

    # K/V 范围（完整序列 = cached + new）
    k_len = tl.load(cu_seqlens_k + seq_idx + 1) - tl.load(cu_seqlens_k + seq_idx)

    # 因果偏移：Q 在完整序列中的起始位置
    q_offset = k_len - q_len

    start_m_idx = start_m * BLOCK_M
    if start_m_idx >= q_len:
        return

    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, BLOCK_N)

    # 加载 Q 块: [BLOCK_M, head_dim]
    q_ptrs = (
        Q
        + (q_seq_start + offs_m[:, None]) * stride_q_tok
        + off_h * stride_q_head
        + offs_d[None, :] * stride_q_dim
    )
    mask_m = offs_m < q_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    # Q 在完整序列中的绝对位置
    q_positions = q_offset + offs_m

    # Online Softmax 状态
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # ─── Phase 1: Non-Causal (所有 K 位置 < 所有 Q 位置, 无需 mask) ───
    # 安全边界: BLOCK_N 对齐的 q_offset + start_m_idx
    causal_start = q_offset + start_m_idx
    aligned_noncausal_end = (causal_start // BLOCK_N) * BLOCK_N

    for start_n in range(0, aligned_noncausal_end, BLOCK_N):
        n_positions = start_n + offs_n

        logical_block_idx = n_positions // kv_block_size
        offset_in_block = n_positions % kv_block_size
        physical_block_idx = tl.load(
            block_tables_ptr
            + seq_idx * stride_bt_b
            + logical_block_idx * stride_bt_block,
        )

        # K 转置: [head_dim, BLOCK_N]
        k_offset = (
            physical_block_idx[None, :] * stride_kc_b
            + offset_in_block[None, :] * stride_kc_t
            + kv_head_idx * stride_kc_h
            + offs_d[:, None] * stride_kc_d
        )
        k = tl.load(K_cache + k_offset)
        qk = tl.dot(q, k)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]

        # V: [BLOCK_N, head_dim]
        v_offset = (
            physical_block_idx[:, None] * stride_vc_b
            + offset_in_block[:, None] * stride_vc_t
            + kv_head_idx * stride_vc_h
            + offs_d[None, :] * stride_vc_d
        )
        v = tl.load(V_cache + v_offset)
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # ─── Phase 2: Causal 区域 (需 causal mask + 边界 mask) ───
    for start_n in range(aligned_noncausal_end, k_len, BLOCK_N):
        n_positions = start_n + offs_n
        mask_n = n_positions < k_len

        logical_block_idx = n_positions // kv_block_size
        offset_in_block = n_positions % kv_block_size
        physical_block_idx = tl.load(
            block_tables_ptr
            + seq_idx * stride_bt_b
            + logical_block_idx * stride_bt_block,
            mask=mask_n,
            other=0,
        )

        # K 转置: [head_dim, BLOCK_N]
        k_offset = (
            physical_block_idx[None, :] * stride_kc_b
            + offset_in_block[None, :] * stride_kc_t
            + kv_head_idx * stride_kc_h
            + offs_d[:, None] * stride_kc_d
        )
        k = tl.load(K_cache + k_offset, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, k)

        # 因果 + 边界 mask
        causal_mask = q_positions[:, None] >= n_positions[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]

        v_offset = (
            physical_block_idx[:, None] * stride_vc_b
            + offset_in_block[:, None] * stride_vc_t
            + kv_head_idx * stride_vc_h
            + offs_d[None, :] * stride_vc_d
        )
        v = tl.load(V_cache + v_offset, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # 写回 HBM
    acc = acc / l_i[:, None]
    o_ptrs = (
        O
        + (q_seq_start + offs_m[:, None]) * stride_o_tok
        + off_h * stride_o_head
        + offs_d[None, :] * stride_o_dim
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_atten_prefill_paged(
    q: torch.Tensor,  # [total_q_tokens, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1]
    cu_seqlens_k: torch.Tensor,  # [num_seqs + 1]
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Paged Prefill Flash Attention：Q 仅含新增 token，K/V 从 paged cache 读取。"""
    output = torch.empty_like(q)

    max_q_len = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    num_seqs = cu_seqlens_q.shape[0] - 1
    kv_block_size = k_cache.shape[1]

    grid = lambda meta: (
        triton.cdiv(max_q_len, meta["BLOCK_M"]),
        num_heads * num_seqs,
    )

    flash_attn_prefill_paged_kernel[grid](
        q,
        output,
        k_cache,
        v_cache,
        block_tables,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_block_size=kv_block_size,
    )

    return output


# ========================== DECODE: Paged Attention ==========================


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["head_dim", "block_size"],
)
@triton.jit
def paged_atten_decode_kernel(
    Out_ptr,
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kc_b,
    stride_kc_t,
    stride_kc_h,
    stride_kc_d,
    stride_vc_b,
    stride_vc_t,
    stride_vc_h,
    stride_vc_d,
    stride_bt_b,
    stride_bt_block,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + batch_idx)

    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * stride_qt + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(Q_ptr + q_offset).to(tl.float32) * scale  # Pre-scale Q

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    num_logical_blocks = tl.cdiv(context_len, block_size)
    offs_block = tl.arange(0, block_size)

    for logical_block_idx in range(num_logical_blocks):
        physical_block_idx = tl.load(
            block_tables_ptr
            + batch_idx * stride_bt_b
            + logical_block_idx * stride_bt_block
        )

        k_offset = (
            physical_block_idx * stride_kc_b
            + kv_head_idx * stride_kc_h
            + offs_block[:, None] * stride_kc_t
            + offs_d[None, :] * stride_kc_d
        )
        v_offset = (
            physical_block_idx * stride_vc_b
            + kv_head_idx * stride_vc_h
            + offs_block[:, None] * stride_vc_t
            + offs_d[None, :] * stride_vc_d
        )

        start_token_idx = logical_block_idx * block_size
        mask = (start_token_idx + offs_block) < context_len

        k = tl.load(K_cache_ptr + k_offset, mask=mask[:, None], other=0.0)
        v = tl.load(V_cache_ptr + v_offset, mask=mask[:, None], other=0.0)

        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(mask, qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)

        acc = acc * alpha
        acc += tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_i_new

    acc = acc / l_i
    out_offset = batch_idx * stride_qt + head_idx * stride_qh + offs_d * stride_qd
    tl.store(Out_ptr + out_offset, acc.to(Out_ptr.dtype.element_ty))


def paged_attention_decode(
    q: torch.Tensor,  # [batch_size, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch_size, max_num_blocks_per_seq]
    context_lens: torch.Tensor,  # [batch_size]
    scale: float = None,
) -> torch.Tensor:
    q = q.contiguous()
    batch_size, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(q)
    grid = (batch_size, num_heads)

    paged_atten_decode_kernel[grid](
        output,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    return output
