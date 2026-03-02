import torch
import triton
import triton.testing

from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.kernels.rope import fused_rope

# 使用 Triton 自带的 Benchmark 工具
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_tokens'],               # X 轴：要测试的 Token 数量
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192], 
        line_arg='provider',                  
        line_vals=['torch', 'triton'],        
        line_names=['PyTorch (Out-of-place)', 'Triton (In-place)'], 
        styles=[('blue', '-'), ('green', '-')],   
        ylabel='GB/s',                        # Y 轴：显存带宽
        plot_name='rope-performance',     
        args={'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 128} # 模拟 Qwen3 GQA 架构
    )
)
def benchmark(num_tokens, num_heads, num_kv_heads, head_dim, provider):
    # 初始化数据
    q = torch.randn(num_tokens, num_heads, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    positions = torch.randint(0, 4096, (num_tokens,), device='cuda', dtype=torch.int32)
    
    # 实例化 PyTorch 的 RoPE 层，获取预计算好的 cos_sin_cache
    rope_layer = RotaryEmbedding(
        head_size=head_dim, 
        rotary_dim=head_dim, 
        max_position_embeddings=8192, 
        base=10000.0
    ).cuda()
    cos_sin_cache = rope_layer.cos_sin_cache

    # 预热并测量耗时
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        # PyTorch 会创建并返回新的 q 和 k 张量 (Out-of-place)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rope_layer(positions, q, k), quantiles=quantiles
        )
    elif provider == 'triton':
        # Triton 会直接在传入的 q 和 k 上原地修改 (In-place)
        # 数学冷知识：因为 RoPE 是纯粹的旋转操作（保范数），所以即使 do_bench 把这段数据
        # 原地旋转几千次，它的数值也永远不会溢出变成 NaN，极其适合直接压测！
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fused_rope(q, k, positions, cos_sin_cache), quantiles=quantiles
        )

    # -------------------------------------------------------------
    # 计算理论显存访问量 (Bytes)
    # 读: Q, K, positions, 以及对应的 cos_sin
    # 写: Q_out, K_out
    # -------------------------------------------------------------
    q_bytes = q.numel() * q.element_size()
    k_bytes = k.numel() * k.element_size()
    pos_bytes = positions.numel() * positions.element_size()
    # 提取的 cos 和 sin 大小：每个 token 提取 128 个 float16
    cos_sin_bytes = num_tokens * head_dim * 2 
    
    # 总访问字节数
    total_bytes = 2 * q_bytes + 2 * k_bytes + pos_bytes + cos_sin_bytes
    
    gbps = lambda ms: total_bytes / (ms * 1e-3) / 1e9
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='.')