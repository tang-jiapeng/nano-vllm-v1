"""
调度器性能深度分析 - 逐步仪表化，记录调度机制的内部运行指标。

采集指标（逐步）：
  - 推理时间 / 调度时间 / 后处理时间（耗时分解）
  - 批次类型（prefill / decode）及批次大小
  - Prefill 批次利用率（实际新 token 数 / max_batched_tokens）
  - Prefix Cache 命中率（命中 token / 总 prompt token）
  - KV-cache 利用率（已用块 / 总块数）
  - 抢占（Preemption）次数

汇总指标：
  - 时间分解百分比
  - Prefill / Decode 步比例
  - 平均批次大小和利用率
  - 总 Prefix Cache 命中率
  - 总抢占次数及抢占率

输出：
  - 终端打印分析报告
  - 保存 benchmarks/scheduler_steps.csv    （逐步原始数据）
  - 保存 benchmarks/scheduler_summary.csv  （汇总指标）
"""

import csv
import os
import statistics
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/Qwen3-0.6B")


def run_scheduler_bench(
    llm: LLM,
    num_seqs: int = 256,
    input_range: tuple = (100, 512),
    output_range: tuple = (100, 256),
    rng_seed: int = 42,
) -> list[dict]:
    """
    向调度器提交 num_seqs 个请求，通过直接调用 scheduler.schedule() 收集
    逐步指标，返回每步的详细记录列表。
    """
    seed(rng_seed)

    # ── 注入 preempt 计数器 ────────────────────────────────────────────────
    _step_preemptions = [0]
    _orig_preempt = llm.scheduler.preempt

    def _patched_preempt(seq):
        _step_preemptions[0] += 1
        return _orig_preempt(seq)

    llm.scheduler.preempt = _patched_preempt

    # ── 准备请求 ──────────────────────────────────────────────────────────
    prompts = [
        [randint(0, 10000) for _ in range(randint(*input_range))]
        for _ in range(num_seqs)
    ]
    params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(*output_range),
        )
        for _ in range(num_seqs)
    ]
    for p, sp in zip(prompts, params):
        llm.add_request(p, sp)

    total_kv_blocks = llm.model_runner.config.num_kvcache_blocks
    max_batched_tokens = llm.model_runner.config.max_num_batched_tokens

    steps = []

    while not llm.is_finished():
        _step_preemptions[0] = 0

        # ── 调度 ──────────────────────────────────────────────────────────
        t_s0 = time.perf_counter()
        seqs, is_prefill = llm.scheduler.schedule()
        t_s1 = time.perf_counter()

        # ── 采集调度后统计（prefill 才有 prefix cache 信息）──────────────
        if is_prefill:
            new_tokens = sum(len(s) - s.num_cached_tokens for s in seqs)
            prefix_hits = sum(s.num_cached_tokens for s in seqs)
            total_prompt = new_tokens + prefix_hits
            batch_util = new_tokens / max_batched_tokens if max_batched_tokens else 0
        else:
            new_tokens = len(seqs)
            prefix_hits = 0
            total_prompt = 0
            batch_util = 0.0

        # ── 推理 ──────────────────────────────────────────────────────────
        t_i0 = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        t_i1 = time.perf_counter()

        # ── 后处理 ────────────────────────────────────────────────────────
        llm.scheduler.postprocess(seqs, token_ids)
        t_p1 = time.perf_counter()

        # ── KV-cache 利用率 ───────────────────────────────────────────────
        kv_used = len(llm.scheduler.block_manager.used_block_ids)

        steps.append(
            {
                "step": len(steps),
                "is_prefill": int(is_prefill),
                "num_seqs": len(seqs),
                "new_tokens": new_tokens,
                "prefix_hit_tokens": prefix_hits,
                "total_prompt_tokens": total_prompt,
                "batch_utilization": round(batch_util, 4),
                "kv_used_blocks": kv_used,
                "kv_total_blocks": total_kv_blocks,
                "kv_utilization": round(kv_used / total_kv_blocks, 4),
                "schedule_ms": round((t_s1 - t_s0) * 1000, 3),
                "infer_ms": round((t_i1 - t_i0) * 1000, 3),
                "post_ms": round((t_p1 - t_i1) * 1000, 3),
                "total_step_ms": round((t_p1 - t_s0) * 1000, 3),
                "preemptions": _step_preemptions[0],
            }
        )

    llm.scheduler.preempt = _orig_preempt  # 还原
    return steps


def build_summary(steps: list[dict], num_seqs: int) -> dict:
    """从逐步数据计算汇总指标。"""
    prefill_steps = [s for s in steps if s["is_prefill"]]
    decode_steps = [s for s in steps if not s["is_prefill"]]

    total_infer_ms = sum(s["infer_ms"] for s in steps)
    total_sched_ms = sum(s["schedule_ms"] for s in steps)
    total_post_ms = sum(s["post_ms"] for s in steps)
    total_ms = sum(s["total_step_ms"] for s in steps)

    total_new_tokens = sum(s["new_tokens"] for s in prefill_steps)
    total_prefix_hits = sum(s["prefix_hit_tokens"] for s in prefill_steps)
    total_prompt_toks = total_new_tokens + total_prefix_hits
    total_preemptions = sum(s["preemptions"] for s in steps)

    batch_utils = [
        s["batch_utilization"] for s in prefill_steps if s["batch_utilization"] > 0
    ]
    decode_sizes = [s["num_seqs"] for s in decode_steps]
    kv_utils = [s["kv_utilization"] for s in steps]

    return {
        "num_seqs": num_seqs,
        "total_steps": len(steps),
        "prefill_steps": len(prefill_steps),
        "decode_steps": len(decode_steps),
        "total_wall_ms": round(total_ms, 1),
        # 时间分解
        "infer_pct": round(100 * total_infer_ms / total_ms, 1),
        "sched_pct": round(100 * total_sched_ms / total_ms, 1),
        "post_pct": round(100 * total_post_ms / total_ms, 1),
        "avg_infer_ms": round(total_infer_ms / len(steps), 2),
        "avg_sched_ms": round(total_sched_ms / len(steps), 3),
        "avg_post_ms": round(total_post_ms / len(steps), 3),
        # Prefill 效率
        "avg_prefill_batch_util": round(
            statistics.mean(batch_utils) if batch_utils else 0, 4
        ),
        "min_prefill_batch_util": round(min(batch_utils) if batch_utils else 0, 4),
        # Prefix cache
        "prefix_cache_hit_rate": (
            round(total_prefix_hits / total_prompt_toks, 4) if total_prompt_toks else 0
        ),
        "total_prefix_hits": total_prefix_hits,
        "total_new_tokens": total_new_tokens,
        # Decode 效率
        "avg_decode_batch_size": round(
            statistics.mean(decode_sizes) if decode_sizes else 0, 1
        ),
        "min_decode_batch_size": min(decode_sizes) if decode_sizes else 0,
        "max_decode_batch_size": max(decode_sizes) if decode_sizes else 0,
        # KV-cache
        "avg_kv_utilization": round(statistics.mean(kv_utils), 4),
        "peak_kv_utilization": round(max(kv_utils), 4),
        # 抢占
        "total_preemptions": total_preemptions,
        "preemption_rate": round(total_preemptions / num_seqs, 4),
    }


def print_report(summary: dict):
    W = 64
    print()
    print("═" * W)
    print("  nano-vllm 调度器性能分析报告")
    print("═" * W)

    print(f"\n  [总览]")
    print(f"    请求总数      : {summary['num_seqs']}")
    print(
        f"    总步数        : {summary['total_steps']}"
        f"  (Prefill: {summary['prefill_steps']}, Decode: {summary['decode_steps']})"
    )
    print(f"    总计算时间    : {summary['total_wall_ms']:.1f} ms")
    print(
        f"    总抢占次数    : {summary['total_preemptions']}"
        f"  (抢占率 {summary['preemption_rate']:.1%})"
    )

    print(f"\n  [时间分解（平均每步）]")
    print(
        f"    推理  Inference : {summary['avg_infer_ms']:>8.2f} ms  ({summary['infer_pct']:.1f}%)"
    )
    print(
        f"    调度  Schedule  : {summary['avg_sched_ms']:>8.3f} ms  ({summary['sched_pct']:.1f}%)"
    )
    print(
        f"    后处理 Post     : {summary['avg_post_ms']:>8.3f} ms  ({summary['post_pct']:.1f}%)"
    )

    print(f"\n  [Prefill 批次效率]")
    print(
        f"    平均批次利用率  : {summary['avg_prefill_batch_util']:.1%}"
        f"  (最低 {summary['min_prefill_batch_util']:.1%})"
    )
    if summary["total_prefix_hits"] + summary["total_new_tokens"] > 0:
        print(f"\n  [Prefix Cache 命中率]")
        print(f"    命中 token    : {summary['total_prefix_hits']:>8}")
        print(f"    未命中 token  : {summary['total_new_tokens']:>8}")
        print(f"    命中率        : {summary['prefix_cache_hit_rate']:.1%}")

    print(f"\n  [Decode 批次大小]")
    print(
        f"    平均 : {summary['avg_decode_batch_size']:.1f}"
        f"  最小 : {summary['min_decode_batch_size']}"
        f"  最大 : {summary['max_decode_batch_size']}"
    )

    print(f"\n  [KV-Cache 利用率]")
    print(
        f"    平均 : {summary['avg_kv_utilization']:.1%}"
        f"  峰值 : {summary['peak_kv_utilization']:.1%}"
    )

    print()
    print("═" * W)


def main():
    print("=" * 64)
    print("  nano-vllm 调度器内部性能基准")
    print(f"  模型: Qwen3-0.6B   enforce_eager=True    max_model_len=2048")
    print("=" * 64)

    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=2048)

    # Warmup
    llm.generate(
        [[1, 2, 3, 4, 5]],
        [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8)],
        use_tqdm=False,
    )

    NUM_SEQS = 256
    print(f"\n  开始采集（{NUM_SEQS} 个请求，变长输入/输出）...\n")
    steps = run_scheduler_bench(
        llm,
        num_seqs=NUM_SEQS,
        input_range=(100, 512),
        output_range=(100, 256),
    )

    summary = build_summary(steps, NUM_SEQS)
    print_report(summary)

    # ── 保存逐步原始数据 ──────────────────────────────────────────────────
    steps_csv = os.path.join(os.path.dirname(__file__), "scheduler_steps.csv")
    with open(steps_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=steps[0].keys())
        w.writeheader()
        w.writerows(steps)
    print(f"  逐步数据已保存: {steps_csv}  ({len(steps)} 行)")

    # ── 保存汇总指标 ──────────────────────────────────────────────────────
    summary_csv = os.path.join(os.path.dirname(__file__), "scheduler_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    print(f"  汇总指标已保存: {summary_csv}")


if __name__ == "__main__":
    main()
