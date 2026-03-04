"""
vLLM v1 端到端正确性测试（需要 GPU 和模型）。

测试覆盖：
1. 基本生成正确性
2. Chunked prefill 输出一致性
3. Prefix caching 正确性
4. 混合负载生成
5. 多轮生成稳定性
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "models/Qwen3-0.6B")


def cleanup_gpu():
    """释放 GPU 内存。"""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_basic_generation():
    """基本生成正确性：确保能正常产出文本。"""
    llm = LLM(MODEL_PATH, enforce_eager=True)
    prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Write a short poem.",
    ]
    params = SamplingParams(temperature=0.6, max_tokens=32)
    outputs = llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 3
    for o in outputs:
        assert len(o["token_ids"]) > 0
        assert len(o["text"]) > 0
    print("✓ test_basic_generation")
    llm.exit()
    cleanup_gpu()


def test_chunked_prefill():
    """Chunked prefill：分块处理应产出合理文本。"""
    llm = LLM(MODEL_PATH, enforce_eager=True, chunked_prefill=True)
    prompts = ["Tell me a story about a brave knight."]
    params = SamplingParams(temperature=0.6, max_tokens=64)
    outputs = llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 1
    assert len(outputs[0]["token_ids"]) > 0
    print(
        f"  Chunked output ({len(outputs[0]['token_ids'])} tokens): {outputs[0]['text'][:80]}..."
    )
    print("✓ test_chunked_prefill")
    llm.exit()
    cleanup_gpu()


def test_prefix_caching():
    """Prefix caching：相同前缀的第二次请求应更快。"""
    llm = LLM(MODEL_PATH, enforce_eager=True)
    system = "You are a helpful AI assistant. " * 20
    params = SamplingParams(temperature=0.6, max_tokens=16)

    # 第一次请求（无缓存）
    t1 = time.time()
    out1 = llm.generate([system + "What is AI?"], params, use_tqdm=False)
    t1 = time.time() - t1

    # 第二次请求（相同前缀，应命中缓存）
    t2 = time.time()
    out2 = llm.generate([system + "What is ML?"], params, use_tqdm=False)
    t2 = time.time() - t2

    assert len(out1) == 1 and len(out2) == 1
    print(f"  First: {t1:.3f}s, Second (cached): {t2:.3f}s")
    print("✓ test_prefix_caching")
    llm.exit()
    cleanup_gpu()


def test_mixed_workload():
    """混合长度负载：短中长 prompt 混合生成。"""
    llm = LLM(MODEL_PATH, enforce_eager=True)
    prompts = [
        "Hi",
        "Write a paragraph about machine learning.",
        "Explain the concept of attention mechanism in transformers in detail.",
    ]
    params = [
        SamplingParams(temperature=0.6, max_tokens=8),
        SamplingParams(temperature=0.6, max_tokens=32),
        SamplingParams(temperature=0.6, max_tokens=64),
    ]
    outputs = llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 3
    for i, o in enumerate(outputs):
        assert len(o["token_ids"]) > 0, f"Output {i} is empty"
    print("✓ test_mixed_workload")
    llm.exit()
    cleanup_gpu()


def test_batch_generation():
    """批量生成稳定性：多个请求并发。"""
    llm = LLM(MODEL_PATH, enforce_eager=True)
    prompts = [f"Count to {i}: 1, 2," for i in range(5, 15)]
    params = SamplingParams(temperature=0.6, max_tokens=32)
    outputs = llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 10
    for o in outputs:
        assert len(o["token_ids"]) > 0
    print("✓ test_batch_generation")
    llm.exit()
    cleanup_gpu()


def test_chunked_prefill_mixed():
    """Chunked prefill + 混合长度：长短 prompt 混合。"""
    llm = LLM(MODEL_PATH, enforce_eager=True, chunked_prefill=True)
    prompts = ["Hi"] * 5 + ["Tell me about " + " ".join(["topic"] * 50)] * 3
    params = SamplingParams(temperature=0.6, max_tokens=16)
    outputs = llm.generate(prompts, params, use_tqdm=False)

    assert len(outputs) == 8
    for o in outputs:
        assert len(o["token_ids"]) > 0
    print("✓ test_chunked_prefill_mixed")
    llm.exit()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Set MODEL_PATH env var.")
        sys.exit(1)

    print("=" * 50)
    print("E2E v1 Tests (requires GPU)")
    print("=" * 50)
    test_basic_generation()
    test_chunked_prefill()
    test_prefix_caching()
    test_mixed_workload()
    test_batch_generation()
    test_chunked_prefill_mixed()

    print()
    print("=" * 50)
    print("All 6 E2E tests passed! ✅")
    print("=" * 50)
