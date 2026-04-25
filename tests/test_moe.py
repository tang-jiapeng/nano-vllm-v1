import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nanovllm.kernels.moe_impl import moe_align_block_size_triton
from nanovllm.layers.moe.fused import FusedMoe, fused_topk
from nanovllm.utils.loader import load_model

HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not HAS_CUDA, reason="requires cuda")
def test_fused_topk_and_align_block_size():
    hidden_states = torch.zeros(3, 4, dtype=torch.float16, device="cuda")
    router_logits = torch.tensor(
        [
            [1.0, 2.0, 0.5],
            [0.2, 0.1, 0.9],
            [3.0, -1.0, 0.0],
        ],
        dtype=torch.float16,
        device="cuda",
    )
    topk_weights, topk_ids = fused_topk(
        hidden_states, router_logits, top_k=2, renormalize=True
    )

    assert topk_weights.shape == (3, 2)
    assert topk_ids.shape == (3, 2)
    torch.testing.assert_close(topk_weights.sum(dim=-1), torch.ones(3))

    block_size = 4
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_triton(
        topk_ids, block_size=block_size, num_experts=3
    )

    assert sorted_token_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_padded.dtype == torch.int32
    assert int(num_tokens_post_padded.item()) % block_size == 0
    assert sorted_token_ids.numel() >= int(num_tokens_post_padded.item())

    num_valid = topk_ids.numel()
    valid_prefix = sorted_token_ids[: int(num_tokens_post_padded.item())]
    assert torch.all((valid_prefix >= 0) & (valid_prefix <= num_valid))


def test_fused_moe_weight_loader_tp_shards(monkeypatch):
    import nanovllm.layers.moe.fused as moe_layer

    monkeypatch.setattr(moe_layer.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(moe_layer.dist, "get_rank", lambda: 1)

    layer = nn.Module()
    moe = FusedMoe()
    moe.create_weights(layer, num_experts=2, hidden_size=3, intermediate_size=4)

    gate_weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    up_weight = torch.arange(100, 112, dtype=torch.float32).view(4, 3)
    down_weight = torch.arange(12, dtype=torch.float32).view(3, 4)

    layer.w1.weight_loader(layer.w1, gate_weight, expert_id=0, loaded_shard_id=0)
    layer.w1.weight_loader(layer.w1, up_weight, expert_id=0, loaded_shard_id=1)
    layer.w2.weight_loader(layer.w2, down_weight, expert_id=0)

    expected_gate_local = gate_weight.chunk(2, dim=0)[1]
    expected_up_local = up_weight.chunk(2, dim=0)[1]
    expected_down_local = down_weight.chunk(2, dim=1)[1]

    torch.testing.assert_close(layer.w1[0, :2], expected_gate_local)
    torch.testing.assert_close(layer.w1[0, 2:], expected_up_local)
    torch.testing.assert_close(layer.w2[0], expected_down_local)


class _DummyMoEModel(nn.Module):
    packed_modules_mapping = {}

    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.mlp = nn.Module()


def test_moe_safetensors_loader_maps_expert_weights_with_tp_shards(
    monkeypatch, tmp_path
):
    import nanovllm.layers.moe.fused as moe_layer

    monkeypatch.setattr(moe_layer.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(moe_layer.dist, "get_rank", lambda: 1)

    model = _DummyMoEModel()
    moe = FusedMoe()
    moe.create_weights(model.block.mlp, num_experts=2, hidden_size=3, intermediate_size=4)

    tensors = {
        "block.mlp.experts.0.gate_proj.weight": torch.arange(12, dtype=torch.float32).view(4, 3),
        "block.mlp.experts.0.up_proj.weight": torch.arange(100, 112, dtype=torch.float32).view(4, 3),
        "block.mlp.experts.0.down_proj.weight": torch.arange(12, dtype=torch.float32).view(3, 4),
        "block.mlp.experts.1.gate_proj.weight": torch.arange(200, 212, dtype=torch.float32).view(4, 3),
        "block.mlp.experts.1.up_proj.weight": torch.arange(300, 312, dtype=torch.float32).view(4, 3),
        "block.mlp.experts.1.down_proj.weight": torch.arange(24, 36, dtype=torch.float32).view(3, 4),
    }
    save_file(tensors, tmp_path / "model.safetensors")

    load_model(model, str(tmp_path))

    torch.testing.assert_close(
        model.block.mlp.w1[0, :2], tensors["block.mlp.experts.0.gate_proj.weight"].chunk(2, dim=0)[1]
    )
    torch.testing.assert_close(
        model.block.mlp.w1[0, 2:], tensors["block.mlp.experts.0.up_proj.weight"].chunk(2, dim=0)[1]
    )
    torch.testing.assert_close(
        model.block.mlp.w2[0], tensors["block.mlp.experts.0.down_proj.weight"].chunk(2, dim=1)[1]
    )
    torch.testing.assert_close(
        model.block.mlp.w1[1, :2], tensors["block.mlp.experts.1.gate_proj.weight"].chunk(2, dim=0)[1]
    )
    torch.testing.assert_close(
        model.block.mlp.w1[1, 2:], tensors["block.mlp.experts.1.up_proj.weight"].chunk(2, dim=0)[1]
    )
    torch.testing.assert_close(
        model.block.mlp.w2[1], tensors["block.mlp.experts.1.down_proj.weight"].chunk(2, dim=1)[1]
    )
