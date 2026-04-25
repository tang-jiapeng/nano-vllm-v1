from dataclasses import dataclass
import math


@dataclass
class SamplingParams:
    """采样参数，控制 temperature、最大生成长度及是否忽略 EOS 等生成行为。"""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert math.isfinite(self.temperature), "temperature must be finite"
        assert self.temperature >= 0.0, "temperature must be non-negative"
