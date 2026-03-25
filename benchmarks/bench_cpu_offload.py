"""
CPU KV offload 基准测试。

设计目标：
1. 支持对比 offload 开 / 关在相同压力下的行为差异。
2. 统计 swap_in / swap_out 次数、CPU/GPU KV block 使用率峰值。

用法示例：
  # 
  python benchmarks/bench_cpu_offload.py --model ./models/Qwen3-0.6B --preset realistic

  # 极限 thrash 压测
  python benchmarks/bench_cpu_offload.py --model ./models/Qwen3-0.6B --preset stress

  # 只跑单个 case
  python benchmarks/bench_cpu_offload.py \
    --model ./models/Qwen3-4B-AWQ \
    --num-seqs 256 \
    --input-len 255 \
    --output-len 256 \
    --override-gpu-blocks 4 \
    --override-cpu-blocks 24 \
    --compare-offload
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager


DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "../models/Qwen3-0.6B")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "cpu_offload_results.jsonl"
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchmarkCase:
    model: str
    num_seqs: int
    input_len: int
    output_len: int
    offload: bool
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 256
    max_model_len: int = 4096
    chunked_prefill: bool = True
    enforce_eager: bool = False
    cpu_offload_gb: float = 8.0
    cpu_offload_safety_margin_gb: float = 1.0
    cpu_offload_watermark_blocks: int = 0
    override_gpu_blocks: int = 0
    override_cpu_blocks: int = 0
    warmup_iters: int = 1
    seed: int = 42


def save_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_prompts(num_seqs: int, input_len: int, rng_seed: int) -> list[list[int]]:
    """构造固定长度的随机 token prompts。"""
    random.seed(rng_seed)
    return [[random.randint(100, 10000) for _ in range(input_len)] for _ in range(num_seqs)]


def build_sampling_params(num_seqs: int, output_len: int) -> list[SamplingParams]:
    """为每条请求构造统一采样参数。"""
    return [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]


def override_block_manager_if_needed(llm: LLM, case: BenchmarkCase):
    """按需缩小逻辑 block 池，便于稳定制造 offload 压力。"""
    if case.override_gpu_blocks <= 0 and case.override_cpu_blocks <= 0:
        return

    old_bm = llm.scheduler.block_manager
    gpu_blocks = old_bm.total_blocks
    cpu_blocks = llm.model_runner.config.num_cpu_kvcache_blocks

    if case.override_gpu_blocks > 0:
        gpu_blocks = min(case.override_gpu_blocks, gpu_blocks)
    if case.offload and case.override_cpu_blocks > 0:
        cpu_blocks = min(case.override_cpu_blocks, cpu_blocks)
    elif not case.offload:
        cpu_blocks = 0

    if gpu_blocks <= 0:
        raise ValueError("override_gpu_blocks results in zero available GPU blocks.")
    if case.offload and cpu_blocks <= 0:
        raise ValueError("override_cpu_blocks results in zero available CPU blocks.")

    llm.scheduler.block_manager = BlockManager(
        gpu_blocks,
        llm.model_runner.config.kvcache_block_size,
        speculative_decoding=llm.scheduler.speculative_decoding,
        num_speculative_tokens=llm.scheduler.num_speculative_tokens,
        num_cpu_blocks=cpu_blocks,
        cpu_offload_watermark_blocks=case.cpu_offload_watermark_blocks,
    )


def run_engine_loop(llm: LLM) -> dict:
    """手动步进调度器，收集 CPU offload 相关统计。"""
    generated_tokens = 0
    finished_seq_ids = set()
    total_steps = 0
    scheduled_seq_count = 0
    prefill_steps = 0
    decode_steps = 0
    swap_out_steps = 0
    swap_in_steps = 0
    swap_out_blocks = 0
    swap_in_blocks = 0
    max_gpu_kv_usage = 0.0
    max_cpu_kv_usage = 0.0

    while not llm.is_finished():
        step = llm.scheduler.schedule()
        if not step.seqs and not step.swap_in_map and not step.swap_out_map:
            break

        total_steps += 1
        scheduled_seq_count += len(step.seqs)
        if any(seq.num_new_tokens > 1 for seq in step.seqs):
            prefill_steps += 1
        if any(seq.num_new_tokens == 1 for seq in step.seqs):
            decode_steps += 1
        if step.swap_out_map:
            swap_out_steps += 1
            swap_out_blocks += len(step.swap_out_map)
        if step.swap_in_map:
            swap_in_steps += 1
            swap_in_blocks += len(step.swap_in_map)

        token_ids, seq_need = llm.model_runner.call(
            "run", step.seqs, step.swap_in_map, step.swap_out_map
        )
        if step.seqs:
            llm.scheduler.postprocess(step.seqs, token_ids, seq_need)
            generated_tokens += len(token_ids or [])
            for seq in step.seqs:
                if seq.is_finished:
                    finished_seq_ids.add(seq.seq_id)

        bm = llm.scheduler.block_manager
        max_gpu_kv_usage = max(max_gpu_kv_usage, bm.usage_ratio)
        max_cpu_kv_usage = max(max_cpu_kv_usage, bm.cpu_usage_ratio)

    avg_batch_size = scheduled_seq_count / max(total_steps, 1)
    return {
        "generated_tokens": generated_tokens,
        "finished_requests": len(finished_seq_ids),
        "total_steps": total_steps,
        "avg_batch_size": avg_batch_size,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "swap_out_steps": swap_out_steps,
        "swap_in_steps": swap_in_steps,
        "swap_out_blocks": swap_out_blocks,
        "swap_in_blocks": swap_in_blocks,
        "max_gpu_kv_usage": max_gpu_kv_usage,
        "max_cpu_kv_usage": max_cpu_kv_usage,
    }


def run_case(case: BenchmarkCase) -> dict:
    """执行单个 benchmark case 并返回统计结果。"""
    print(
        "\n--- Running case: "
        f"num_seqs={case.num_seqs}, inlen={case.input_len}, "
        f"outlen={case.output_len}, offload={case.offload}, "
        f"override_gpu_blocks={case.override_gpu_blocks}, "
        f"override_cpu_blocks={case.override_cpu_blocks} ---"
    )

    llm = None
    try:
        llm = LLM(
            case.model,
            tensor_parallel_size=case.tensor_parallel_size,
            enforce_eager=case.enforce_eager,
            chunked_prefill=case.chunked_prefill,
            max_num_batched_tokens=case.max_num_batched_tokens,
            max_num_seqs=case.max_num_seqs,
            max_model_len=case.max_model_len,
            enable_kv_offload=case.offload,
            cpu_offload_gb=case.cpu_offload_gb,
            cpu_offload_safety_margin_gb=case.cpu_offload_safety_margin_gb,
            cpu_offload_watermark_blocks=case.cpu_offload_watermark_blocks,
        )
        override_block_manager_if_needed(llm, case)

        warmup_prompts = build_prompts(2, min(case.input_len, 32), case.seed + 999)
        warmup_params = build_sampling_params(2, min(case.output_len, 16))
        for _ in range(case.warmup_iters):
            llm.generate(warmup_prompts, warmup_params, use_tqdm=False)
            cuda_sync()

        prompts = build_prompts(case.num_seqs, case.input_len, case.seed)
        sampling_params = build_sampling_params(case.num_seqs, case.output_len)
        for prompt, sp in zip(prompts, sampling_params):
            llm.add_request(prompt, sp)

        prompt_tokens = case.num_seqs * case.input_len
        start = time.perf_counter()
        loop_stats = run_engine_loop(llm)
        cuda_sync()
        elapsed = time.perf_counter() - start

        total_tokens = prompt_tokens + loop_stats["generated_tokens"]
        return {
            "latency_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": loop_stats["generated_tokens"],
            "generated_tokens": loop_stats["generated_tokens"],
            "total_tokens": total_tokens,
            "request_throughput_rps": case.num_seqs / max(elapsed, 1e-6),
            "token_throughput_tps": total_tokens / max(elapsed, 1e-6),
            "throughput_tps": loop_stats["generated_tokens"] / max(elapsed, 1e-6),
            "gen_throughput_tps": loop_stats["generated_tokens"] / max(elapsed, 1e-6),
            **loop_stats,
        }
    finally:
        if llm is not None:
            llm.exit()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def benchmark_worker(case_dict: dict, result_path: str):
    """子进程入口：执行 case 并将结果写入 JSONL。"""
    case = BenchmarkCase(**case_dict)
    stats = run_case(case)
    payload = {
        "timestamp": time.time(),
        "case": asdict(case),
        "stats": stats,
    }
    save_jsonl(result_path, payload)

    print("\n=== Benchmark Results ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def spawn_case(case: BenchmarkCase, result_path: str):
    """独立进程运行单个 case，避免显存和进程状态互相污染。"""
    p = mp.Process(
        target=benchmark_worker,
        kwargs={"case_dict": asdict(case), "result_path": result_path},
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        print(f"Warning: benchmark process exited with code {p.exitcode}")
    else:
        print("Benchmark process finished successfully.")


def build_matrix_cases(args) -> list[BenchmarkCase]:
    """构造一组默认对比 case。"""
    common = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        chunked_prefill=args.chunked_prefill,
        enforce_eager=args.enforce_eager,
        cpu_offload_gb=args.cpu_offload_gb,
        cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
        cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
        override_gpu_blocks=args.override_gpu_blocks,
        override_cpu_blocks=args.override_cpu_blocks,
        warmup_iters=args.warmup_iters,
        seed=args.seed,
    )
    return [
        BenchmarkCase(
            num_seqs=256,
            input_len=255,
            output_len=256,
            offload=False,
            **common,
        ),
        BenchmarkCase(
            num_seqs=256,
            input_len=255,
            output_len=256,
            offload=True,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=255,
            output_len=256,
            offload=False,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=255,
            output_len=256,
            offload=True,
            **common,
        ),
    ]


def build_realistic_cases(args) -> list[BenchmarkCase]:
    """
    - 不人为压小 GPU block
    - 使用较大的 CPU offload 空间
    - 使用较大的 batched token budget，减少调度切分带来的额外偏差
    - 使用 eager 模式
    """
    common = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=1024 * 256,
        max_model_len=4096,
        chunked_prefill=True,
        enforce_eager=True,
        cpu_offload_gb=max(args.cpu_offload_gb, 60.0),
        cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
        cpu_offload_watermark_blocks=0,
        override_gpu_blocks=0,
        override_cpu_blocks=0,
        warmup_iters=2,
        seed=args.seed,
    )
    return [
        BenchmarkCase(
            num_seqs=384,
            input_len=256,
            output_len=640,
            offload=False,
            max_num_seqs=384,
            **common,
        ),
        BenchmarkCase(
            num_seqs=384,
            input_len=256,
            output_len=640,
            offload=True,
            max_num_seqs=384,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=1024,
            output_len=2048,
            offload=False,
            max_num_seqs=512,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=1024,
            output_len=2048,
            offload=True,
            max_num_seqs=512,
            **common,
        ),
        BenchmarkCase(
            num_seqs=64,
            input_len=2048,
            output_len=4096,
            offload=False,
            max_num_seqs=64,
            **common,
        ),
        BenchmarkCase(
            num_seqs=64,
            input_len=2048,
            output_len=4096,
            offload=True,
            max_num_seqs=64,
            **common,
        ),
    ]


def build_stress_cases(args) -> list[BenchmarkCase]:
    """
    极限 CPU offload 压测口径：
    - 人为压小 GPU / CPU block 池
    - prompt 接近 block 边界，便于频繁触发 swap
    """
    common = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        chunked_prefill=args.chunked_prefill,
        enforce_eager=args.enforce_eager,
        cpu_offload_gb=args.cpu_offload_gb,
        cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
        cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
        override_gpu_blocks=args.override_gpu_blocks if args.override_gpu_blocks > 0 else 4,
        override_cpu_blocks=args.override_cpu_blocks if args.override_cpu_blocks > 0 else 24,
        warmup_iters=args.warmup_iters,
        seed=args.seed,
    )
    return [
        BenchmarkCase(
            num_seqs=256,
            input_len=255,
            output_len=256,
            offload=False,
            **common,
        ),
        BenchmarkCase(
            num_seqs=256,
            input_len=255,
            output_len=256,
            offload=True,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=255,
            output_len=256,
            offload=False,
            **common,
        ),
        BenchmarkCase(
            num_seqs=512,
            input_len=255,
            output_len=256,
            offload=True,
            **common,
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="CPU KV offload benchmark")
    parser.add_argument("--model", default=os.path.expanduser(DEFAULT_MODEL))
    parser.add_argument(
        "--preset",
        choices=["realistic", "stress"],
        default=None,
        help="预设 benchmark 口径：realistic 做参考口径对比，stress 做极限压测。",
    )
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--input-len", type=int, default=255)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--cpu-offload-gb", type=float, default=8.0)
    parser.add_argument("--cpu-offload-safety-margin-gb", type=float, default=1.0)
    parser.add_argument("--cpu-offload-watermark-blocks", type=int, default=0)
    parser.add_argument(
        "--override-gpu-blocks",
        type=int,
        default=4,
        help="压小逻辑 GPU block 数，便于稳定触发 offload；0 表示不覆盖。",
    )
    parser.add_argument(
        "--override-cpu-blocks",
        type=int,
        default=24,
        help="压小逻辑 CPU block 数；0 表示不覆盖。",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare-offload",
        action="store_true",
        help="用当前输入规模分别跑 offload 开 / 关两组 case。",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="运行预设 case 矩阵（offload 开/关对比）。",
    )
    parser.add_argument(
        "--result-path",
        default=RESULTS_PATH,
        help="JSONL 结果保存路径。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    if args.preset == "realistic":
        cases = build_realistic_cases(args)
    elif args.preset == "stress":
        cases = build_stress_cases(args)
    elif args.matrix:
        cases = build_matrix_cases(args)
    elif args.compare_offload:
        common = dict(
            model=args.model,
            num_seqs=args.num_seqs,
            input_len=args.input_len,
            output_len=args.output_len,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            chunked_prefill=args.chunked_prefill,
            enforce_eager=args.enforce_eager,
            cpu_offload_gb=args.cpu_offload_gb,
            cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
            cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
            override_gpu_blocks=args.override_gpu_blocks,
            override_cpu_blocks=args.override_cpu_blocks,
            warmup_iters=args.warmup_iters,
            seed=args.seed,
        )
        cases = [
            BenchmarkCase(offload=False, **common),
            BenchmarkCase(offload=True, **common),
        ]
    else:
        cases = [
            BenchmarkCase(
                model=args.model,
                num_seqs=args.num_seqs,
                input_len=args.input_len,
                output_len=args.output_len,
                offload=True,
                tensor_parallel_size=args.tensor_parallel_size,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                chunked_prefill=args.chunked_prefill,
                enforce_eager=args.enforce_eager,
                cpu_offload_gb=args.cpu_offload_gb,
                cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
                cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
                override_gpu_blocks=args.override_gpu_blocks,
                override_cpu_blocks=args.override_cpu_blocks,
                warmup_iters=args.warmup_iters,
                seed=args.seed,
            )
        ]

    print("Running CPU KV offload benchmark...")
    print(f"Result path: {args.result_path}")
    for case in cases:
        print(f"Starting benchmark process: {asdict(case)}")
        spawn_case(case, args.result_path)


if __name__ == "__main__":
    main()
