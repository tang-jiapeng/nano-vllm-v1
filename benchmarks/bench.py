import os
import sys
import time
from random import randint, seed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm import LLM, SamplingParams

# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    path = os.path.expanduser("~/nano-vllm/models/Qwen3-0.6B/")
    # 支持命令行参数覆盖默认值
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=path)
    parser.add_argument("--num-seqs", type=int, default=num_seqs)
    parser.add_argument("--max-input", type=int, default=max_input_len)
    parser.add_argument("--max-output", type=int, default=max_output_len)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=4096,
        chunked_prefill=args.chunked_prefill,
    )

    seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, args.max_output),
        )
        for _ in range(args.num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    mode = "chunked" if args.chunked_prefill else "standard"
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, "
        f"Throughput: {throughput:.2f}tok/s ({mode})"
    )


if __name__ == "__main__":
    main()
