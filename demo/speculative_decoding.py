"""
nano-vllm 推测解码 (Speculative Decoding) 示例
=============================================

使用小模型（draft）快速生成候选 token，再由大模型（target）一次验证，
以空间换时间加速推理。

前置条件：
  - 准备 target 模型和 draft 模型（draft 需与 target 同词表）

使用方式:
  # 基础用法：Qwen3-4B-AWQ(target) + Qwen3-0.6B(draft), K=5
  python demo/speculative_decoding.py \
      --target-model ./models/Qwen3-4B-AWQ \
      --draft-model  ./models/Qwen3-0.6B \
      --num-speculative-tokens 5

  # 使用 eager 模式（不捕获 CUDA Graph，方便调试）
  python demo/speculative_decoding.py \
      --target-model ./models/Qwen3-4B-AWQ \
      --draft-model  ./models/Qwen3-0.6B \
      --num-speculative-tokens 5 \
      --enforce-eager

  # 对比：不启用推测解码（去掉 --draft-model 即可）
  python demo/speculative_decoding.py \
      --target-model ./models/Qwen3-4B-AWQ
"""

import argparse
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(args):
    target_path = os.path.expanduser(args.target_model)
    draft_path = os.path.expanduser(args.draft_model) if args.draft_model else None
    tokenizer = AutoTokenizer.from_pretrained(target_path)

    llm_kwargs = dict(
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, 4096),
    )
    if draft_path:
        llm_kwargs["speculative_model"] = draft_path
        llm_kwargs["num_speculative_tokens"] = args.num_speculative_tokens

    llm = LLM(target_path, **llm_kwargs)

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
            enable_thinking=False,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    total_proposed = 0
    total_accepted = 0
    for prompt, output in zip(prompts, outputs):
        print("\n" + "=" * 60)
        print(f"Prompt:     {prompt[:80]!r}...")
        print(f"Completion: {output['text']!r}")
        proposed = output.get("proposed", 0)
        accepted = output.get("accepted", 0)
        total_proposed += proposed
        total_accepted += accepted
        if proposed > 0:
            print(f"Accept rate: {accepted}/{proposed} = {accepted/proposed:.2%}")

    if total_proposed > 0:
        print("\n" + "=" * 60)
        print(
            f"Overall accept rate: {total_accepted}/{total_proposed} "
            f"= {total_accepted/total_proposed:.2%}"
        )
    else:
        print("\n(Speculative decoding not enabled — no accept rate to report)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="nano-vllm speculative decoding example"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="./models/Qwen3-1.7B",
        help="Target（大）模型路径",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="./models/Qwen3-0.6B",
        help="Draft（小）模型路径，不设则关闭推测解码",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        "-K",
        type=int,
        default=5,
        help="每轮推测的候选 token 数",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="禁用 CUDA Graph，方便调试",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    main(args)
