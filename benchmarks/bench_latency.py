"""
延迟基准测试 - 测量 TTFT / TPOT / E2E 延迟的百分位分布。

指标定义：
  TTFT : Time To First Token  — 从提交请求到首个输出 token 就绪的时间
  TPOT : Time Per Output Token — 输出阶段每个 token 的平均间隔时间
  E2E  : End-to-End Latency   — 从提交到生成完毕的总时间

测试维度：
  ① 并发规模：concurrency = 1 / 4 / 16 / 64（固定 input=256, output=128）
  ② 输出长度：output_len  = 32 / 128 / 512（固定 input=128, concurrency=32）

说明：
  本测试通过直接调用 scheduler.schedule() 实现逐步仪表化，
  可精确追踪每条请求的 prefill 完成时间和 finish 时间。

输出：
  - 终端打印延迟分位数表
  - 保存 benchmarks/latency_results.csv
"""

import csv
import math
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/Qwen3-0.6B")


def percentile(data: list[float], p: float) -> float:
    """计算第 p 百分位数（线性插值）。"""
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def measure_latency(
    llm: LLM, num_requests: int, input_len: int, output_len: int, rng_seed: int = 42
) -> dict:
    """
    提交 num_requests 个请求，通过步进循环记录每条请求的
    prefill 完成时刻（→ TTFT）和 finish 时刻（→ E2E）。
    """
    seed(rng_seed)
    prompts = [
        [randint(0, 10000) for _ in range(input_len)] for _ in range(num_requests)
    ]
    params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_requests)
    ]

    # 提交全部请求
    t_submit = time.perf_counter()
    for p, sp in zip(prompts, params):
        llm.add_request(p, sp)

    # per-seq 时间戳
    prefill_done: dict[int, float] = {}  # seq_id → first-token 就绪时刻
    finish_info: dict[int, tuple] = {}  # seq_id → (finish_time, n_output_tokens)

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        t_infer = time.perf_counter()

        # prefill 步结束 → 每条序列的第一个 token 已就绪
        if is_prefill:
            for seq in seqs:
                prefill_done.setdefault(seq.seq_id, t_infer)

        finished_flags = llm.scheduler.postprocess(seqs, token_ids)
        t_post = time.perf_counter()

        for seq, done in zip(seqs, finished_flags):
            if done and seq.seq_id not in finish_info:
                finish_info[seq.seq_id] = (t_post, seq.num_completion_tokens)

    # ── 计算各条请求的延迟 ────────────────────────────────────────────────
    ttfts, tpots, e2es = [], [], []
    for sid, (t_fin, n_out) in finish_info.items():
        t_first = prefill_done.get(sid, t_submit)
        ttft_ms = (t_first - t_submit) * 1000
        e2e_ms = (t_fin - t_submit) * 1000
        ttfts.append(ttft_ms)
        e2es.append(e2e_ms)
        if n_out > 1:
            tpot_ms = (t_fin - t_first) / (n_out - 1) * 1000
            tpots.append(tpot_ms)

    return {
        "num_requests": num_requests,
        "input_len": input_len,
        "output_len": output_len,
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p90_ms": round(percentile(ttfts, 90), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p90_ms": round(percentile(tpots, 90), 2),
        "tpot_p99_ms": round(percentile(tpots, 99), 2),
        "e2e_p50_ms": round(percentile(e2es, 50), 1),
        "e2e_p90_ms": round(percentile(e2es, 90), 1),
        "e2e_p99_ms": round(percentile(e2es, 99), 1),
    }


def print_table(title: str, rows: list[dict], key_col: str, key_label: str):
    print(f"\n  ── {title} " + "─" * (56 - len(title)))
    hdr = (
        f"  {key_label:<10}  "
        f"{'TTFT P50':>9}  {'P90':>9}  {'P99':>9}  "
        f"{'TPOT P50':>9}  {'P90':>9}  "
        f"{'E2E P50':>9}  {'P90':>9}  {'P99':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {str(r[key_col]):<10}  "
            f"{r['ttft_p50_ms']:>8.1f}ms  "
            f"{r['ttft_p90_ms']:>8.1f}ms  "
            f"{r['ttft_p99_ms']:>8.1f}ms  "
            f"{r['tpot_p50_ms']:>8.2f}ms  "
            f"{r['tpot_p90_ms']:>8.2f}ms  "
            f"{r['e2e_p50_ms']:>8.1f}ms  "
            f"{r['e2e_p90_ms']:>8.1f}ms  "
            f"{r['e2e_p99_ms']:>8.1f}ms"
        )


def main():
    print("=" * 72)
    print("  nano-vllm 延迟基准测试（TTFT / TPOT / E2E）")
    print(f"  模型: Qwen3-0.6B   enforce_eager=True    max_model_len=2048")
    print("=" * 72)

    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

    # Warmup
    llm.generate(
        [[1, 2, 3, 4, 5]],
        [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)],
        use_tqdm=False,
    )

    all_rows = []

    # ── ① 并发规模研究（input=256, output=128）────────────────────────────
    print("\n  [研究①] 不同并发数对延迟的影响  (input=256, output=128)")
    concur_rows = []
    for concurrency in [1, 4, 16, 64]:
        print(f"    concurrency={concurrency} ...", end=" ", flush=True)
        r = measure_latency(llm, concurrency, input_len=256, output_len=128)
        concur_rows.append(r)
        all_rows.append({"scenario": "concurrency_study", **r})
        print(f"TTFT_P50={r['ttft_p50_ms']}ms  E2E_P50={r['e2e_p50_ms']}ms")

    print_table(
        "并发数 vs 延迟  (input=256, output=128)",
        concur_rows,
        "num_requests",
        "并发数",
    )

    # ── ② 输出长度研究（concurrency=32, input=128）────────────────────────
    print("\n  [研究②] 不同输出长度对延迟的影响  (input=128, concurrency=32)")
    outlen_rows = []
    for output_len in [32, 128, 512]:
        print(f"    output_len={output_len} ...", end=" ", flush=True)
        r = measure_latency(llm, num_requests=32, input_len=128, output_len=output_len)
        outlen_rows.append(r)
        all_rows.append({"scenario": "output_len_study", **r})
        print(f"TTFT_P50={r['ttft_p50_ms']}ms  TPOT_P50={r['tpot_p50_ms']}ms")

    print_table(
        "输出长度 vs 延迟  (input=128, concurrency=32)",
        outlen_rows,
        "output_len",
        "output_len",
    )

    # ── 总结 ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)

    # ── CSV ───────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "latency_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"  结果已保存: {out}")


if __name__ == "__main__":
    main()
