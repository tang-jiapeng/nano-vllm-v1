import logging
import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.layers.quantization.awq_config import AWQConfig

try:
    import flash_attn as _  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """推理配置，包含模型路径、调度参数、tensor parallel 及 KV-cache 等运行时选项。"""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    chunked_prefill: bool = False
    hf_config: AutoConfig | None = None
    awq_config: AWQConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if HAS_FLASH_ATTN:
            assert (
                self.kvcache_block_size % 256 == 0
            ), "flash-attn requires kvcache_block_size to be a multiple of 256"
        else:
            assert (
                self.kvcache_block_size >= 16
                and (self.kvcache_block_size & (self.kvcache_block_size - 1)) == 0
            ), "kvcache_block_size must be a power of 2 and >= 16"
            logger.warning(
                "flash-attn is not installed, using Triton attention kernel. "
                "Install flash-attn for best performance: pip install flash-attn"
            )
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

        self.awq_config = AWQConfig.from_json(self.model)
        if self.awq_config is not None:
            logger.info(
                f"Detected AWQ quantized model (weight_bits={self.awq_config.weight_bits}, "
                f"group_size={self.awq_config.group_size})"
            )
