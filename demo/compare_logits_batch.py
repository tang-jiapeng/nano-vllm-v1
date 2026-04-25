import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.utils.context import get_context, reset_context


PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "你知道原神吗，请你介绍一下这款游戏",
]


def build_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def format_topk(tokenizer, logits: torch.Tensor, k: int) -> str:
    values, indices = torch.topk(logits, k)
    parts = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        token_str = tokenizer.decode([idx]).replace("\n", "\\n")
        parts.append(f"{idx}:{token_str!r}:{value:.4f}")
    return " | ".join(parts)


@torch.inference_mode()
def hf_next_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    return out.logits[0, -1].float().cpu()


@torch.inference_mode()
def nano_step_logits(llm: LLM):
    seqs = llm.scheduler.schedule()
    input_ids, positions = llm.model_runner.prepare_model_input(seqs)
    logits = llm.model_runner.run_model(input_ids, positions).float().cpu()
    seq_need_compute_logits = get_context().seq_need_compute_logits.tolist()
    reset_context()
    return seqs, logits, seq_need_compute_logits


def main(args):
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [build_prompt(tokenizer, p, args.enable_thinking) for p in PROMPTS[: args.num_prompts]]

    llm = LLM(model_path, enforce_eager=args.enforce_eager)
    seq_id_to_prompt_idx = {}
    for prompt_idx, prompt in enumerate(prompts):
        seq_id = llm.add_request(
            prompt,
            SamplingParams(temperature=0.0, max_tokens=args.steps, ignore_eos=True),
        )
        seq_id_to_prompt_idx[seq_id] = prompt_idx

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda",
    )
    hf_input_ids = [
        tokenizer(prompt, return_tensors="pt").input_ids.to("cuda") for prompt in prompts
    ]

    print("=" * 120)
    print(f"Batch prompts ({len(prompts)}):")
    for idx, prompt in enumerate(prompts):
        print(f"[{idx}] {prompt!r}")
    print("=" * 120)

    try:
        for step in range(args.steps):
            seqs, nano_logits, seq_need_compute_logits = nano_step_logits(llm)
            if not seq_need_compute_logits:
                print(f"Step {step}: no logits to compare")
                break

            forced_tokens = []
            forced_indices = []

            print(f"\nStep {step}")
            print("-" * 120)
            for local_idx in seq_need_compute_logits:
                seq = seqs[local_idx]
                prompt_idx = seq_id_to_prompt_idx[seq.seq_id]
                hf_logits = hf_next_logits(model, hf_input_ids[prompt_idx])
                curr_nano_logits = nano_logits[len(forced_tokens)]

                nano_argmax = int(curr_nano_logits.argmax().item())
                hf_argmax = int(hf_logits.argmax().item())
                max_abs_diff = float((curr_nano_logits - hf_logits).abs().max().item())
                mean_abs_diff = float((curr_nano_logits - hf_logits).abs().mean().item())

                print(
                    f"[prompt {prompt_idx} / seq {seq.seq_id}] "
                    f"nano={nano_argmax}({tokenizer.decode([nano_argmax])!r}) "
                    f"hf={hf_argmax}({tokenizer.decode([hf_argmax])!r}) "
                    f"same={nano_argmax == hf_argmax} "
                    f"max_abs_diff={max_abs_diff:.6f} "
                    f"mean_abs_diff={mean_abs_diff:.6f}"
                )
                print(f"  nano top{args.topk}: {format_topk(tokenizer, curr_nano_logits, args.topk)}")
                print(f"  hf   top{args.topk}: {format_topk(tokenizer, hf_logits, args.topk)}")

                forced_token = hf_argmax
                forced_tokens.append(forced_token)
                forced_indices.append(local_idx)
                hf_input_ids[prompt_idx] = torch.cat(
                    [
                        hf_input_ids[prompt_idx],
                        torch.tensor([[forced_token]], device=hf_input_ids[prompt_idx].device),
                    ],
                    dim=1,
                )

            llm.scheduler.postprocess(seqs, forced_tokens, forced_indices)

            if llm.is_finished():
                break
    finally:
        llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare nano-vllm batched logits against HuggingFace per-request logits"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    main(parser.parse_args())
