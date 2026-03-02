import torch
import pytest
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.kernels.rope import fused_rope

def test_rope_correctness():
    torch.manual_seed(42)
    
    num_tokens = 128
    num_heads = 32
    num_kv_heads = 8  # 模拟 GQA 场景
    head_dim = 128
    max_pos = 4096
    
    # 1. 准备数据
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    # 随机生成一些 token 位置索引
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device="cuda")
    
    # 2. 获取 PyTorch 参考实现的 cos/sin 缓存
    # 使用 nanovllm 的 RotaryEmbedding 层初始化缓存
    rope_layer = RotaryEmbedding(head_size=head_dim, rotary_dim=head_dim, max_position_embeddings=max_pos, base=10000.0)
    rope_layer = rope_layer.cuda()
    
    # 3. 计算 PyTorch 参考结果
    # 注意：需要克隆 q 和 k，因为 Triton kernel 会原地修改它们
    ref_q, ref_k = rope_layer(positions, q.clone(), k.clone())
    
    # 4. 计算 Triton 结果 (原地修改)
    tri_q, tri_k = fused_rope(q, k, positions, rope_layer.cos_sin_cache)
    
    # 5. 对比验证
    torch.testing.assert_close(tri_q, ref_q, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tri_k, ref_k, atol=1e-3, rtol=1e-3)
    
    print("✅ 正确性验证通过！Fused RoPE (Triton) 与 PyTorch 原生实现完全一致。")

if __name__ == "__main__":
    test_rope_correctness()