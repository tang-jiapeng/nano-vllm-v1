#!/usr/bin/env python3
"""
nano-vllm 客户端压测工具
=======================

向运行中的 nano-vllm API Server 发送并发请求，
收集 TTFT、TBT、E2E 延迟、吞吐量等指标，写入 JSON 日志。

前置条件：
  先启动服务端：
    python -m nanovllm.entrypoints.serve --model models/Qwen3-0.6B --enforce-eager

用法：
  # 默认 8 并发 × 20 请求
  python client_bench.py

  # 自定义参数
  python client_bench.py --concurrency 16 --num-requests 50 --max-tokens 128

  # 指定输出日志
  python client_bench.py --output bench_results/run1.json

依赖：
  pip install httpx   # 必需（异步 HTTP 客户端）
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import httpx

# ── 预置 Prompt 池 ──
PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main differences between TCP and UDP?",
    "介绍一下中国的长城。",
    "Write a haiku about artificial intelligence.",
    "What is the time complexity of merge sort and why?",
    "Explain how a transformer neural network works.",
    "列举世界上最高的五座山峰。",
    "Write a short story about a robot who learns to paint.",
    "What are the benefits and risks of nuclear energy?",
    "Describe the process of photosynthesis step by step.",
    "用Python写一个快速排序算法。",
    "What is the difference between machine learning and deep learning?",
    "Explain quantum computing to a 10-year-old.",
    "Write a limerick about a programmer who loves coffee.",
    "How does the HTTP protocol work?",
    "介绍一下量子计算的基本原理。",
    "What are the key principles of object-oriented programming?",
    "Describe the water cycle in detail.",
    "Write a recipe for chocolate chip cookies.",
]


@dataclass
class RequestMetrics:
    """单次请求的度量数据。"""

    request_id: int
    prompt: str
    prompt_len: int = 0  # prompt 字符数
    completion_len: int = 0  # 生成 token 数
    ttft_ms: float = 0.0  # Time To First Token (ms)
    e2e_ms: float = 0.0  # 端到端延迟 (ms)
    tbt_ms: list[float] = field(default_factory=list)  # 每个 token 间延迟
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkReport:
    """整体压测报告。"""

    timestamp: str
    server_url: str
    model_name: str
    concurrency: int
    num_requests: int
    max_tokens: int
    temperature: float
    # ── 汇总指标 ──
    total_time_s: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    # 延迟
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_e2e_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    avg_tbt_ms: float = 0.0
    p50_tbt_ms: float = 0.0
    p99_tbt_ms: float = 0.0
    # 吞吐
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    requests_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    # 原始数据
    per_request: list[dict] = field(default_factory=list)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    idx = min(idx, len(data_sorted) - 1)
    return data_sorted[idx]


async def send_one_request(
    client: httpx.AsyncClient,
    url: str,
    req_id: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestMetrics:
    """发送一个流式 chat 请求并收集度量。"""
    metrics = RequestMetrics(request_id=req_id, prompt=prompt, prompt_len=len(prompt))

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_last_token = None
    token_count = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=180.0) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue

                now = time.perf_counter()
                token_count += 1

                if token_count == 1:
                    metrics.ttft_ms = (now - t_start) * 1000
                else:
                    metrics.tbt_ms.append((now - t_last_token) * 1000)

                t_last_token = now

        metrics.e2e_ms = (time.perf_counter() - t_start) * 1000
        metrics.completion_len = token_count
        metrics.success = True

    except Exception as e:
        metrics.e2e_ms = (time.perf_counter() - t_start) * 1000
        metrics.success = False
        metrics.error = str(e)

    return metrics


async def run_benchmark(args):
    base_url = args.url.rstrip("/")
    chat_url = f"{base_url}/v1/chat/completions"

    # ── 检查连接 ──
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{base_url}/health", timeout=5.0)
            r.raise_for_status()
    except Exception:
        print(f"✗ 无法连接到 {base_url}，请确认服务已启动。")
        return

    # 获取模型名称
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{base_url}/v1/models", timeout=5.0)
            model_name = r.json()["data"][0]["id"]
    except Exception:
        model_name = "unknown"

    # ── 生成请求列表 ──
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_requests)]

    print(f"\n{'='*60}")
    print(f"  nano-vllm Client Benchmark")
    print(f"  Server:      {base_url}")
    print(f"  Model:       {model_name}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Requests:    {args.num_requests}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"{'='*60}\n")

    # ── 并发发送 ──
    semaphore = asyncio.Semaphore(args.concurrency)
    all_metrics: list[RequestMetrics] = []
    completed = 0

    async def bounded_request(client, idx, prompt):
        nonlocal completed
        async with semaphore:
            m = await send_one_request(
                client,
                chat_url,
                idx,
                prompt,
                args.max_tokens,
                args.temperature,
            )
            completed += 1
            status = "✓" if m.success else "✗"
            print(
                f"\r  [{completed}/{args.num_requests}] "
                f"{status} req#{idx:03d}  "
                f"TTFT={m.ttft_ms:6.0f}ms  "
                f"E2E={m.e2e_ms:7.0f}ms  "
                f"tokens={m.completion_len:3d}",
                end="",
                flush=True,
            )
            return m

    t_total_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client, i, p) for i, p in enumerate(prompts)]
        all_metrics = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - t_total_start
    print()  # 换行

    # ── 汇总 ──
    successful = [m for m in all_metrics if m.success]
    failed = [m for m in all_metrics if not m.success]

    ttfts = [m.ttft_ms for m in successful if m.ttft_ms > 0]
    e2es = [m.e2e_ms for m in successful if m.e2e_ms > 0]
    all_tbt = [t for m in successful for t in m.tbt_ms]
    total_completion = sum(m.completion_len for m in successful)
    total_prompt = sum(m.prompt_len for m in successful)

    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        server_url=base_url,
        model_name=model_name,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        total_time_s=round(total_time, 2),
        successful_requests=len(successful),
        failed_requests=len(failed),
        # TTFT
        avg_ttft_ms=round(statistics.mean(ttfts), 2) if ttfts else 0,
        p50_ttft_ms=round(_percentile(ttfts, 50), 2),
        p99_ttft_ms=round(_percentile(ttfts, 99), 2),
        # E2E
        avg_e2e_ms=round(statistics.mean(e2es), 2) if e2es else 0,
        p50_e2e_ms=round(_percentile(e2es, 50), 2),
        p99_e2e_ms=round(_percentile(e2es, 99), 2),
        # TBT
        avg_tbt_ms=round(statistics.mean(all_tbt), 2) if all_tbt else 0,
        p50_tbt_ms=round(_percentile(all_tbt, 50), 2),
        p99_tbt_ms=round(_percentile(all_tbt, 99), 2),
        # throughput
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        requests_per_sec=(
            round(len(successful) / total_time, 2) if total_time > 0 else 0
        ),
        tokens_per_sec=round(total_completion / total_time, 2) if total_time > 0 else 0,
        # per-request detail
        per_request=[
            {
                "id": m.request_id,
                "prompt": m.prompt[:60],
                "completion_tokens": m.completion_len,
                "ttft_ms": round(m.ttft_ms, 2),
                "e2e_ms": round(m.e2e_ms, 2),
                "avg_tbt_ms": round(statistics.mean(m.tbt_ms), 2) if m.tbt_ms else 0,
                "success": m.success,
                "error": m.error,
            }
            for m in all_metrics
        ],
    )

    # ── 打印摘要 ──
    print(f"\n{'─'*60}")
    print(f"  Benchmark Results")
    print(f"{'─'*60}")
    print(f"  Total time:          {report.total_time_s:.1f}s")
    print(
        f"  Successful / Failed: {report.successful_requests} / {report.failed_requests}"
    )
    print(f"")
    print(
        f"  TTFT  (avg/p50/p99): {report.avg_ttft_ms:>8.1f} / {report.p50_ttft_ms:>8.1f} / {report.p99_ttft_ms:>8.1f} ms"
    )
    print(
        f"  TBT   (avg/p50/p99): {report.avg_tbt_ms:>8.1f} / {report.p50_tbt_ms:>8.1f} / {report.p99_tbt_ms:>8.1f} ms"
    )
    print(
        f"  E2E   (avg/p50/p99): {report.avg_e2e_ms:>8.1f} / {report.p50_e2e_ms:>8.1f} / {report.p99_e2e_ms:>8.1f} ms"
    )
    print(f"")
    print(f"  Throughput:          {report.tokens_per_sec:.1f} tokens/s")
    print(f"  Requests/s:          {report.requests_per_sec:.2f}")
    print(f"  Total tokens:        {report.total_completion_tokens}")
    print(f"{'─'*60}")

    # ── 写入日志 ──
    output_path = args.output
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 详细报告已写入: {output_path}\n")

    # ── 同时拉取服务端指标 ──
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{base_url}/v1/metrics", timeout=5.0)
            server_metrics = r.json()
        server_log = output_path.replace(".json", "_server_metrics.json")
        with open(server_log, "w", encoding="utf-8") as f:
            json.dump(server_metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 服务端指标已写入: {server_log}\n")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="nano-vllm 客户端压测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API 服务地址 (default: http://localhost:8000)",
    )
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="并发数")
    parser.add_argument("-n", "--num-requests", type=int, default=20, help="总请求数")
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="每个请求的最大生成 token 数"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/client_bench.json",
        help="结果输出路径 (default: benchmarks/results/client_bench.json)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
