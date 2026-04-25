import torch
import triton

from nanovllm.kernels.triton.awq_gemm import awq_gemm_kernel

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(
    input: torch.Tensor,  # 输入张量 [M, K]
    qweight: torch.Tensor,  # 量化权重张量 [K, N // 8]，每个int32包含8个4位权重
    scales: torch.Tensor,  # 缩放因子张量 [K // G, N]
    qzeros: torch.Tensor,  # 量化零点张量 [K // G, N // 8]
    split_k_iters: int,  # K维度并行迭代次数，必须是2的幂，最大32
    block_size_m: int = 32,  # M维度块大小
    block_size_n: int = 32,  # N维度块大小
    block_size_k: int = 32,  # K维度块大小
) -> torch.Tensor:
    """AWQ量化矩阵乘法的包装函数

    该函数为Triton内核设置参数并启动计算。

    张量形状说明：
    - input:   [M, K] - 输入激活值矩阵
    - qweight: [K, N // 8] - 量化权重，打包到int32中
    - qzeros:  [K // G, N // 8] - 量化零点，打包到int32中
    - scales:  [K // G, N] - 缩放因子，FP16类型
    - 输出:    [M, N] - 结果矩阵

    其中：
    - M = batch_size × seq_len（批量大小 × 序列长度）
    - K = input_dim（输入维度）
    - N = output_dim（输出维度）
    - G = group_size（分组大小）

    Args:
        input: 输入张量 [M, K]
        qweight: 量化权重张量 [K, N // 8]
        scales: 缩放因子张量 [K // G, N]
        qzeros: 量化零点张量 [K // G, N // 8]
        split_k_iters: K维度并行迭代次数（必须是2的幂）
        block_size_m: M维度块大小（默认32）
        block_size_n: N维度块大小（默认32）
        block_size_k: K维度块大小（默认32）

    Returns:
        输出张量 [M, N]
    """
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    # ========== 形状验证 ==========
    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    # split_k_iters必须是2的幂且非零
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    # 分组大小必须是支持的值之一
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # 定义网格函数：确定内核启动的块数量
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    # 分配结果张量 [split_k_iters, M, N]
    # 使用K维度分割，需要在最后对split_k_iters维度求和
    result = torch.zeros((split_k_iters, M, N), dtype=scales.dtype, device=input.device)

    # ========== 启动Triton内核 ==========
    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )

    # 对K维度分割的部分求和，得到最终结果 [M, N]
    result = result.sum(0)

    return result
