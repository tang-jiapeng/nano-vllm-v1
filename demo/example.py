"""
nano-vllm 推理示例
=================

支持 FP16 和 AWQ-INT4 量化模型，AWQ 量化会根据模型目录下的
config.json 自动检测，无需额外参数。

使用方式:
  # FP16 模型
  python example.py --model-path ./models/Qwen3-0.6B

  # AWQ-INT4 量化模型（自动检测，enforce_eager 也会自动开启）
  python example.py --model-path ./models/Qwen3-0.6B-AWQ

  # 多卡张量并行
  python example.py --model-path ./models/Qwen3-0.6B --tp 2
"""

import argparse
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    # AWQ 量化模型由 Config.__post_init__ 自动检测，
    # 并会自动启用 enforce_eager=True，无需手动指定。
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "你知道原神吗，请你介绍一下这款游戏",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-vllm inference example")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/Qwen3-0.6B",
        help="模型目录路径，支持 FP16 或 AWQ-INT4 量化模型",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="张量并行数",
    )
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    main(args)
