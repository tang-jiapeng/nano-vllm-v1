"""
nano-vllm CPU KV offload 演示
============================

这个脚本会逐步打印调度状态，方便观察：
- waiting / running 队列变化
- CPU offload 的 swap out / swap in
- GPU / CPU KV block 使用情况
- 每一步有哪些序列在 prefill / decode

使用方式:
  python demo/cpu_offload_demo.py --model-path ./models/Qwen3-0.6B

建议：
  - 先用较小模型验证
  - 默认会主动将 block manager 的 GPU / CPU block 数压小，
    以便更容易触发 offload
"""

import argparse
import os
from collections import Counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import CacheResidency


def _short(text: str, limit: int = 80) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_residency_counts(seqs) -> str:
    counts = Counter(seq.residency.name for seq in seqs)
    if not counts:
        return "none"
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def _print_step_header(step_idx: int):
    print()
    print("=" * 80)
    print(f"Step {step_idx}")
    print("=" * 80)


def _print_queue_state(llm):
    waiting = list(llm.scheduler.waiting)
    running = list(llm.scheduler.running)
    bm = llm.scheduler.block_manager

    print(
        f"Queues: waiting={len(waiting)}, running={len(running)} | "
        f"waiting residency: {_format_residency_counts(waiting)}"
    )
    print(
        f"GPU KV blocks: used={bm.used_blocks}/{bm.total_blocks} "
        f"({bm.usage_ratio:.1%}) | "
        f"CPU KV blocks: used={bm.used_cpu_blocks}/{bm.total_cpu_blocks} "
        f"({bm.cpu_usage_ratio:.1%})"
    )


def _decode_incremental(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True)


def _build_demo_prompts(
    tokenizer, num_prompts: int, target_prompt_len: int
) -> list[list[int]]:
    """构造接近 block 边界的 prompt token ids，稳定触发 offload。"""
    user_prompts = [
        "请详细介绍一下操作系统中虚拟内存、页表、TLB 与缺页异常之间的关系。",
        "请从训练、推理、显存占用、通信开销几个角度解释 MoE 模型的特点。",
        "请分步骤讲解 CUDA Graph 为什么能降低 decode 阶段的开销。",
        "请系统解释 KV cache、paged attention 与 continuous batching 的关系。",
        "请讲解 Transformer 推理阶段为什么会产生 KV cache。",
        "请解释连续批处理为什么能提升 LLM 服务吞吐。",
        "请说明 paged attention 为什么能缓解显存碎片。",
        "请解释 MoE 模型在部署时为什么会更复杂。",
    ]
    filler = " 请详细、系统、分步骤地展开说明，并给出结构化总结。"

    prompts: list[list[int]] = []
    for idx in range(num_prompts):
        base_prompt = user_prompts[idx % len(user_prompts)]
        # 通过扩充 user content 来逼近目标长度，避免把 filler 追加到 assistant
        # generation prompt 之后，污染最终回答。
        repeat = 0
        best_ids = None
        best_gap = None

        while repeat <= 128:
            content = base_prompt + filler * repeat
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            gap = abs(len(token_ids) - target_prompt_len)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_ids = token_ids
            if len(token_ids) >= target_prompt_len:
                break
            repeat += 1

        assert best_ids is not None
        prompts.append(best_ids)
    return prompts


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    max_num_batched_tokens = max(args.max_num_batched_tokens, args.max_model_len)
    if max_num_batched_tokens != args.max_num_batched_tokens:
        print(
            "Adjusting max_num_batched_tokens to satisfy "
            f"max_num_batched_tokens >= max_model_len: "
            f"{args.max_num_batched_tokens} -> {max_num_batched_tokens}"
        )

    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        chunked_prefill=args.chunked_prefill,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_kv_offload=True,
        cpu_offload_gb=args.cpu_offload_gb,
        cpu_offload_safety_margin_gb=args.cpu_offload_safety_margin_gb,
        cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
    )

    # 为了让 demo 更容易稳定触发 offload，这里主动把 block manager 的逻辑块数压小。
    old_bm = llm.scheduler.block_manager
    demo_gpu_blocks = min(args.demo_gpu_blocks, old_bm.total_blocks)
    demo_cpu_blocks = min(
        args.demo_cpu_blocks, llm.model_runner.config.num_cpu_kvcache_blocks
    )
    if demo_gpu_blocks <= 0 or demo_cpu_blocks <= 0:
        raise ValueError(
            "Demo block counts must be positive. Please increase "
            "--demo-gpu-blocks / --demo-cpu-blocks or CPU offload space."
        )
    llm.scheduler.block_manager = BlockManager(
        demo_gpu_blocks,
        llm.model_runner.config.kvcache_block_size,
        speculative_decoding=llm.scheduler.speculative_decoding,
        num_speculative_tokens=llm.scheduler.num_speculative_tokens,
        num_cpu_blocks=demo_cpu_blocks,
        cpu_offload_watermark_blocks=args.cpu_offload_watermark_blocks,
    )

    print("=" * 80)
    print("nano-vllm CPU KV offload demo")
    print("=" * 80)
    print(f"Model path:              {path}")
    print(f"Tensor parallel size:    {args.tensor_parallel_size}")
    print(f"Chunked prefill:         {args.chunked_prefill}")
    print(f"Max model len:           {args.max_model_len}")
    print(f"Max batched tokens:      {max_num_batched_tokens}")
    print(f"Max num seqs:            {args.max_num_seqs}")
    print(f"Configured CPU offload:  {args.cpu_offload_gb} GB")
    print(f"Original GPU blocks:     {old_bm.total_blocks}")
    print(f"Original CPU blocks:     {llm.model_runner.config.num_cpu_kvcache_blocks}")
    print(f"Demo GPU blocks:         {demo_gpu_blocks}")
    print(f"Demo CPU blocks:         {demo_cpu_blocks}")
    print(f"CPU offload watermark:   {args.cpu_offload_watermark_blocks}")
    print(f"Num prompts:             {args.num_prompts}")
    print(f"Target prompt len:       {args.target_prompt_len} tokens")

    prompts = _build_demo_prompts(
        tokenizer,
        num_prompts=args.num_prompts,
        target_prompt_len=args.target_prompt_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    print()
    print("Submitting prompts:")
    for idx, prompt in enumerate(prompts):
        seq_id = llm.add_request(prompt, sampling_params)
        print(f"  - req={idx} -> seq={seq_id}, prompt_tokens={len(prompt)}")

    cumulative = {}
    step_idx = 0

    while not llm.is_finished():
        _print_step_header(step_idx)
        _print_queue_state(llm)

        step = llm.scheduler.schedule()
        seqs = step.seqs

        print(
            f"Scheduled: seqs={len(seqs)}, "
            f"swap_out_blocks={len(step.swap_out_map)}, "
            f"swap_in_blocks={len(step.swap_in_map)}"
        )

        if step.swap_out_map:
            print(f"  swap_out_map: {step.swap_out_map}")
        if step.swap_in_map:
            print(f"  swap_in_map:  {step.swap_in_map}")

        if seqs:
            seq_summary = ", ".join(
                (
                    f"seq={seq.seq_id}"
                    f"/{'D' if seq.num_new_tokens == 1 else 'P'}"
                    f"/n={seq.num_new_tokens}"
                    f"/{seq.residency.name}"
                )
                for seq in seqs
            )
            print(f"  scheduled: {seq_summary}")

        if seqs:
            for seq in seqs:
                if seq.residency == CacheResidency.CPU:
                    print(
                        f"  seq={seq.seq_id} is still CPU-offloaded before this step"
                    )

        token_ids, seq_need = llm.model_runner.call(
            "run", seqs, step.swap_in_map, step.swap_out_map
        )

        if seqs:
            llm.scheduler.postprocess(seqs, token_ids, seq_need)

        # 打印本步产生的新 token
        if seqs and seq_need:
            print("New tokens:")
            for seq_index, token_id in zip(seq_need, token_ids):
                seq = seqs[seq_index]
                piece = _decode_incremental(tokenizer, token_id)
                cumulative.setdefault(seq.seq_id, [])
                cumulative[seq.seq_id].append(token_id)
                print(
                    f"  - seq={seq.seq_id:>3} token_id={token_id:<8} "
                    f"text={piece!r}"
                )

        finished = [seq for seq in seqs if seq.is_finished]
        if finished:
            print("Finished sequences:")
            for seq in finished:
                text = tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True)
                print(f"  - seq={seq.seq_id:>3} completion={_short(text, 120)!r}")

        step_idx += 1

    print()
    print("=" * 80)
    print("Final outputs")
    print("=" * 80)
    for seq_id in sorted(cumulative):
        text = tokenizer.decode(cumulative[seq_id], skip_special_tokens=True)
        print(f"seq={seq_id:>3} -> {text!r}")

    llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-vllm CPU KV offload demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/Qwen3-0.6B",
        help="模型目录路径",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        type=int,
        default=1,
        help="张量并行数",
    )
    parser.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=8,
        help="提交请求数。更多请求更容易触发 offload。",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=8.0,
        help="CPU offload 空间（GB）",
    )
    parser.add_argument(
        "--cpu-offload-safety-margin-gb",
        type=float,
        default=1.0,
        help="CPU 可用内存安全余量（GB）",
    )
    parser.add_argument(
        "--cpu-offload-watermark-blocks",
        type=int,
        default=0,
        help="恢复时额外保留的 GPU block 数，减少来回震荡",
    )
    parser.add_argument(
        "--demo-gpu-blocks",
        type=int,
        default=6,
        help="仅用于 demo：主动压小逻辑 GPU block 数，方便触发 offload",
    )
    parser.add_argument(
        "--demo-cpu-blocks",
        type=int,
        default=24,
        help="仅用于 demo：主动压小逻辑 CPU block 数",
    )
    parser.add_argument(
        "--target-prompt-len",
        type=int,
        default=255,
        help="目标 prompt token 长度。设为接近 256 的倍数更容易触发 offload。",
    )
    args = parser.parse_args()

    main(args)
