"""
vLLM v1 调度器性能基准测试。

测试场景：
1. Chunked Prefill vs Non-Chunked 吞吐量对比
2. Prefix Caching 效果测量
3. 混合负载下的 TTFT / TPOT 延迟分析
4. 调度器内部指标（per-step 时间、batch 利用率）
5. 不同负载模式下的可扩展性测试

用法：
    python benchmarks/bench_v1_scheduler.py --model models/Qwen3-0.6B
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def bench_throughput(model_path, chunked_prefill=False, num_prompts=50, max_tokens=64):
    """测量总吞吐量（tok/s）。"""
    label = "chunked" if chunked_prefill else "standard"
    print(f"\n{'='*60}")
    print(
        f"Throughput Benchmark ({label}, {num_prompts} prompts, max_tokens={max_tokens})"
    )
    print(f"{'='*60}")

    llm = LLM(model_path, enforce_eager=True, chunked_prefill=chunked_prefill)

    random.seed(42)
    prompts = []
    for _ in range(num_prompts):
        length = random.choice([32, 64, 128, 256])
        prompts.append([random.randint(100, 10000) for _ in range(length)])

    params = SamplingParams(temperature=0.6, max_tokens=max_tokens)

    start = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - start

    total_prompt_tokens = sum(len(p) for p in prompts)
    total_gen_tokens = sum(len(o["token_ids"]) for o in outputs)
    total_tokens = total_prompt_tokens + total_gen_tokens

    print(f"\n  Elapsed:       {elapsed:.2f}s")
    print(f"  Prompt tokens: {total_prompt_tokens}")
    print(f"  Gen tokens:    {total_gen_tokens}")
    print(f"  Throughput:    {total_tokens/elapsed:.1f} tok/s")
    print(f"  Gen speed:     {total_gen_tokens/elapsed:.1f} tok/s")

    llm.exit()
    return {
        "mode": label,
        "elapsed": elapsed,
        "prompt_tokens": total_prompt_tokens,
        "gen_tokens": total_gen_tokens,
        "throughput": total_tokens / elapsed,
        "gen_speed": total_gen_tokens / elapsed,
    }


def bench_prefix_caching(model_path, num_prompts=20, max_tokens=32):
    """测量 prefix caching 加速效果。"""
    print(f"\n{'='*60}")
    print(f"Prefix Caching Benchmark ({num_prompts} prompts)")
    print(f"{'='*60}")

    llm = LLM(model_path, enforce_eager=True)

    # 公共前缀（~200 tokens）
    system = "You are a helpful assistant. " * 20
    params = SamplingParams(temperature=0.6, max_tokens=max_tokens)

    # 第一批：无缓存
    prompts_1 = [system + f"Question {i}: What is {i}?" for i in range(num_prompts)]
    t1 = time.time()
    out1 = llm.generate(prompts_1, params, use_tqdm=False)
    t1 = time.time() - t1

    # 第二批：相同前缀，应命中缓存
    prompts_2 = [
        system + f"Question {i}: Tell me about {i}." for i in range(num_prompts)
    ]
    t2 = time.time()
    out2 = llm.generate(prompts_2, params, use_tqdm=False)
    t2 = time.time() - t2

    speedup = t1 / max(t2, 1e-6)
    print(f"\n  First batch (no cache): {t1:.3f}s")
    print(f"  Second batch (cached):  {t2:.3f}s")
    print(f"  Speedup:                {speedup:.2f}x")

    llm.exit()
    return {"first": t1, "second": t2, "speedup": speedup}


def bench_scheduler_metrics(model_path, num_prompts=30, max_tokens=64):
    """收集调度器内部指标。"""
    print(f"\n{'='*60}")
    print(f"Scheduler Metrics ({num_prompts} prompts)")
    print(f"{'='*60}")

    llm = LLM(model_path, enforce_eager=True, chunked_prefill=True)

    random.seed(123)
    prompts = []
    for _ in range(num_prompts):
        length = random.choice([32, 64, 128, 256, 512])
        prompts.append([random.randint(100, 10000) for _ in range(length)])

    params = SamplingParams(temperature=0.6, max_tokens=max_tokens)

    for prompt, sp in zip(prompts, [params] * len(prompts)):
        llm.add_request(prompt, sp)

    step_metrics = []
    step_idx = 0

    while not llm.is_finished():
        t_start = time.time()

        seqs = llm.scheduler.schedule()
        if not seqs:
            continue

        t_sched = time.time()

        num_prefill = sum(s.num_new_tokens for s in seqs if s.num_new_tokens > 1)
        num_decode = sum(1 for s in seqs if s.num_new_tokens == 1)
        batch_size = len(seqs)

        token_ids, seq_need = llm.model_runner.call("run", seqs)
        t_infer = time.time()

        llm.scheduler.postprocess(seqs, token_ids, seq_need)
        t_post = time.time()

        step_metrics.append(
            {
                "step": step_idx,
                "batch_size": batch_size,
                "prefill_tokens": num_prefill,
                "decode_tokens": num_decode,
                "sched_ms": (t_sched - t_start) * 1000,
                "infer_ms": (t_infer - t_sched) * 1000,
                "post_ms": (t_post - t_infer) * 1000,
                "total_ms": (t_post - t_start) * 1000,
            }
        )
        step_idx += 1

    # 汇总统计
    total_steps = len(step_metrics)
    avg_batch = sum(m["batch_size"] for m in step_metrics) / max(total_steps, 1)
    avg_sched = sum(m["sched_ms"] for m in step_metrics) / max(total_steps, 1)
    avg_infer = sum(m["infer_ms"] for m in step_metrics) / max(total_steps, 1)
    avg_post = sum(m["post_ms"] for m in step_metrics) / max(total_steps, 1)
    avg_total = sum(m["total_ms"] for m in step_metrics) / max(total_steps, 1)

    prefill_steps = [m for m in step_metrics if m["prefill_tokens"] > 0]
    decode_only_steps = [m for m in step_metrics if m["prefill_tokens"] == 0]
    mixed_steps = [
        m for m in step_metrics if m["prefill_tokens"] > 0 and m["decode_tokens"] > 0
    ]

    print(f"\n  Total steps:           {total_steps}")
    print(f"  Avg batch size:        {avg_batch:.1f}")
    print(f"  Avg schedule time:     {avg_sched:.2f} ms")
    print(f"  Avg inference time:    {avg_infer:.2f} ms")
    print(f"  Avg postprocess time:  {avg_post:.2f} ms")
    print(f"  Avg total step time:   {avg_total:.2f} ms")
    print(f"  Prefill steps:         {len(prefill_steps)}")
    print(f"  Decode-only steps:     {len(decode_only_steps)}")
    print(f"  Mixed steps:           {len(mixed_steps)}")

    # 保存详细步骤数据
    csv_path = os.path.join(os.path.dirname(__file__), "v1_scheduler_steps.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
        writer.writeheader()
        writer.writerows(step_metrics)
    print(f"\n  Step metrics saved to {csv_path}")

    llm.exit()
    return {
        "total_steps": total_steps,
        "avg_batch": avg_batch,
        "avg_sched_ms": avg_sched,
        "avg_infer_ms": avg_infer,
        "mixed_steps": len(mixed_steps),
    }


def bench_scalability(model_path, max_tokens=32):
    """不同并发数下的吞吐量可扩展性。"""
    print(f"\n{'='*60}")
    print(f"Scalability Benchmark")
    print(f"{'='*60}")

    concurrency_levels = [1, 5, 10, 20, 50]
    results = []

    for n in concurrency_levels:
        llm = LLM(model_path, enforce_eager=True, chunked_prefill=True)

        random.seed(42)
        prompts = [[random.randint(100, 10000) for _ in range(64)] for _ in range(n)]
        params = SamplingParams(temperature=0.6, max_tokens=max_tokens)

        start = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - start

        gen_tokens = sum(len(o["token_ids"]) for o in outputs)
        throughput = gen_tokens / elapsed

        results.append(
            {
                "concurrency": n,
                "elapsed": elapsed,
                "gen_tokens": gen_tokens,
                "throughput": throughput,
            }
        )
        print(f"  Concurrency={n:3d}: {throughput:.1f} tok/s ({elapsed:.2f}s)")
        llm.exit()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM v1 Scheduler Benchmarks")
    parser.add_argument("--model", default="models/Qwen3-0.6B", help="Model path")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (fewer prompts)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        sys.exit(1)

    n = 10 if args.quick else 50

    all_results = {}

    # 1. 吞吐量对比
    all_results["standard"] = bench_throughput(
        args.model, chunked_prefill=False, num_prompts=n
    )
    all_results["chunked"] = bench_throughput(
        args.model, chunked_prefill=True, num_prompts=n
    )

    # 2. Prefix caching
    all_results["prefix"] = bench_prefix_caching(args.model, num_prompts=max(n // 5, 5))

    # 3. 调度器指标
    all_results["scheduler"] = bench_scheduler_metrics(
        args.model, num_prompts=max(n // 2, 10)
    )

    # 4. 可扩展性
    all_results["scalability"] = bench_scalability(args.model)

    # 总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    s = all_results["standard"]
    c = all_results["chunked"]
    print(f"  Standard throughput: {s['throughput']:.1f} tok/s")
    print(f"  Chunked throughput:  {c['throughput']:.1f} tok/s")
    print(f"  Prefix speedup:     {all_results['prefix']['speedup']:.2f}x")
    print(f"  Mixed steps:        {all_results['scheduler']['mixed_steps']}")

    # 保存汇总
    csv_path = os.path.join(os.path.dirname(__file__), "v1_benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["standard_throughput", f"{s['throughput']:.1f}"])
        writer.writerow(["chunked_throughput", f"{c['throughput']:.1f}"])
        writer.writerow(["prefix_speedup", f"{all_results['prefix']['speedup']:.2f}"])
        writer.writerow(
            ["avg_sched_ms", f"{all_results['scheduler']['avg_sched_ms']:.3f}"]
        )
        writer.writerow(["mixed_steps", all_results["scheduler"]["mixed_steps"]])
    print(f"\n  Summary saved to {csv_path}")
