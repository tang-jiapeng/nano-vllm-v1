import json
import os
from dataclasses import dataclass

from typing import Optional

@dataclass
class AWQConfig:
    """AWQ 量化配置。"""

    weight_bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"

    def __post_init__(self):
        if self.weight_bits != 4:
            raise ValueError(
                f"Only AWQ int4 is supported, got weight_bits={self.weight_bits}"
            )
        self.pack_factor = 32 // self.weight_bits  # 8

    @classmethod
    def from_json(cls, model_path: str) -> Optional["AWQConfig"]:
        """
        扫描模型目录，检测 AWQ 量化配置，返回 AWQConfig 或 None。

        检测顺序：
        1. 独立量化配置文件：quant_config.json / quantize_config.json
        2. 嵌入在 config.json 的 quantization_config 字段（HuggingFace 新格式）
        """
        candidate_files = ["quant_config.json", "quantize_config.json"]
        cfg = None
        for filename in candidate_files:
            path = os.path.join(model_path, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                break

        # 回退：检查 config.json 中嵌入的 quantization_config（HuggingFace 格式）
        if cfg is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    hf_cfg = json.load(f)
                cfg = hf_cfg.get("quantization_config")

        if cfg is None:
            return None

        method = str(cfg.get("quant_method", cfg.get("quant_type", ""))).lower()
        if "awq" not in method:
            return None

        return cls(
            weight_bits=int(cfg.get("w_bit", cfg.get("bits", 4))),
            group_size=int(cfg.get("q_group_size", cfg.get("group_size", 128))),
            zero_point=bool(cfg.get("zero_point", True)),
            version=str(cfg.get("version", "GEMM")),
        )