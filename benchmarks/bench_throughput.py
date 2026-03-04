"""
吞吐量基准测试 - 在多种工作负载下测量 nano-vllm 的吞吐量。

工作负载场景：
  1. short-short  : 短输入(64-128)  + 短输出(32-64)    → 聊天场景
  2. short-long   : 短输入(64-128)  + 长输出(256-512)   → 纯生成场景
  3. long-short   : 长输入(256-512) + 短输出(32-64)     → 文档处理场景
  4. long-long    : 长输入(256-512) + 长输出(256-512)   → 重度工作负载
  5. variable     : 变长输入输出(64-512)                 → 真实场景模拟
  6. high-concur  : 高并发(256路), 均匀长度              → 并发压力测试

输出：
  - 终端打印吞吐量汇总表
  - 保存 benchmarks/throughput_results.csv
"""

import csv
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/Qwen3-0.6B")

# (name, num_seqs, input_len_range, output_len_range, description)
WORKLOADS = [
    ("short-short", 256, (64, 128), (32, 64), "短输入+短输出(聊天)"),
    ("short-long", 128, (64, 128), (256, 512), "短输入+长输出(生成)"),
    ("long-short", 128, (256, 512), (32, 64), "长输入+短输出(处理)"),
    ("long-long", 64, (256, 512), (256, 512), "长输入+长输出(重载)"),
    ("variable", 256, (64, 512), (64, 512), "变长混合(真实场景)"),
    ("high-concur", 256, (192, 256), (192, 256), "高并发均匀负载"),
]


def run_workload(
    llm: LLM, num_seqs: int, in_range: tuple, out_range: tuple, rng_seed: int = 42
) -> dict:
    seed(rng_seed)
    input_lens = [randint(*in_range) for _ in range(num_seqs)]
    output_lens = [randint(*out_range) for _ in range(num_seqs)]
    prompts = [[randint(0, 10000) for _ in range(l)] for l in input_lens]
    params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=l)
        for l in output_lens
    ]

    # 单次 warmup，避免 JIT/graph 捕获污染计时
    llm.generate(
        [prompts[0]],
        [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)],
        use_tqdm=False,
    )

    t0 = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    in_toks = sum(input_lens)
    out_toks = sum(output_lens)
    return {
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "elapsed_s": round(elapsed, 3),
        "prefill_toks_s": round(in_toks / elapsed),
        "decode_toks_s": round(out_toks / elapsed),
        "total_toks_s": round((in_toks + out_toks) / elapsed),
        "req_s": round(num_seqs / elapsed, 2),
    }


def main():
    print("=" * 72)
    print("  nano-vllm 吞吐量基准测试")
    print(f"  模型: Qwen3-0.6B   enforce_eager=True    max_model_len=2048")
    print("=" * 72)

    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

    rows = []
    for name, n, in_r, out_r, desc in WORKLOADS:
        print(f"\n  ▶ {name:<14} ({desc}, {n} 请求) ...")
        r = run_workload(llm, n, in_r, out_r)
        rows.append({"workload": name, "description": desc, "num_seqs": n, **r})
        print(
            f"     Prefill {r['prefill_toks_s']:>7} tok/s  |"
            f"  Decode {r['decode_toks_s']:>7} tok/s  |"
            f"  Total {r['total_toks_s']:>7} tok/s  |"
            f"  {r['req_s']:>5.2f} req/s"
        )

    # ── 汇总表 ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(
        f"  {'工作负载':<14}  {'N':>4}  {'Prefill':>9}  {'Decode':>9}  {'Total':>9}  {'Req/s':>7}"
    )
    print(f"  {'':14}  {'':4}  {'(tok/s)':>9}  {'(tok/s)':>9}  {'(tok/s)':>9}  {'':>7}")
    print("  " + "-" * 68)
    for r in rows:
        print(
            f"  {r['workload']:<14}  {r['num_seqs']:>4}"
            f"  {r['prefill_toks_s']:>9}"
            f"  {r['decode_toks_s']:>9}"
            f"  {r['total_toks_s']:>9}"
            f"  {r['req_s']:>7.2f}"
        )
    print("=" * 72)

    # ── CSV ───────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "throughput_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
