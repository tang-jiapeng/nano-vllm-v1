import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def test_sampling_params_allow_zero_temperature():
    params = SamplingParams(temperature=0.0, max_tokens=8)
    assert params.temperature == 0.0


def test_sampler_zero_temperature_is_greedy():
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 2.0, -1.0],
            [5.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.0], dtype=torch.float32)

    tokens = sampler(logits, temperatures)
    expected = logits.argmax(dim=-1)

    torch.testing.assert_close(tokens, expected)


def test_sampler_supports_mixed_greedy_and_sampling_batch():
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.8], dtype=torch.float32)

    tokens = sampler(logits, temperatures)

    assert tokens.shape == (2,)
    assert tokens[0].item() == 1
    assert 0 <= tokens[1].item() < logits.shape[1]
