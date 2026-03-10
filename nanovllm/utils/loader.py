import os
import re
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器：直接 copy 权重到参数。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 safetensors 文件加载权重，支持 packed modules 和 tensor parallel 分片。"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    moe_expert_pattern = re.compile(
        r"^(.*)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|qweight|qzeros|scales)$"
    )

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)

                # MoE fused 权重映射：
                # experts.i.{gate,up,down}_proj.{weight|qweight|qzeros|scales}
                # -> mlp.{w1|w2}* 的 expert 维切片
                m = moe_expert_pattern.match(weight_name)
                if m is not None:
                    prefix, expert_id_str, proj_name, tensor_name = m.groups()
                    expert_id = int(expert_id_str)
                    layer_mlp_prefix = f"{prefix}.mlp"

                    try:
                        if proj_name in ("gate_proj", "up_proj"):
                            target_name = (
                                f"{layer_mlp_prefix}.w1"
                                if tensor_name == "weight"
                                else f"{layer_mlp_prefix}.w1_{tensor_name}"
                            )
                            loaded_shard_id = 0 if proj_name == "gate_proj" else 1
                            param = model.get_parameter(target_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(
                                param,
                                loaded_weight,
                                expert_id,
                                loaded_shard_id,
                            )
                            continue

                        target_name = (
                            f"{layer_mlp_prefix}.w2"
                            if tensor_name == "weight"
                            else f"{layer_mlp_prefix}.w2_{tensor_name}"
                        )
                        param = model.get_parameter(target_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, expert_id)
                        continue
                    except Exception:
                        # 目标参数不存在时回退到默认加载流程（兼容旧实现）
                        pass

                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


def print_model(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                print(f"{weight_name} {f.get_tensor(weight_name).shape}")


if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser(description="nano vllm")
    argparse.add_argument("--model-path", type=str, default="./models/Qwen3-0.6B")
    args = argparse.parse_args()
    print_model(args.model_path)
