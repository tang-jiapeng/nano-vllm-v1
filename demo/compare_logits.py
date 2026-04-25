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
def nano_next_logits(llm: LLM) -> tuple[torch.Tensor, object]:
    seqs = llm.scheduler.schedule()
    input_ids, positions = llm.model_runner.prepare_model_input(seqs)
    logits = llm.model_runner.run_model(input_ids, positions)
    seq_need_compute_logits = get_context().seq_need_compute_logits.tolist()
    reset_context()
    assert seq_need_compute_logits == [0], f"Unexpected seq_need_compute_logits={seq_need_compute_logits}"
    return logits[0].float().cpu(), seqs[0]


def main(args):
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = build_prompt(tokenizer, args.prompt, args.enable_thinking)

    llm = LLM(model_path, enforce_eager=args.enforce_eager)
    llm.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=args.steps))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda",
    )
    hf_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    print("=" * 100)
    print(f"Prompt: {prompt!r}")
    print("=" * 100)

    try:
        for step in range(args.steps):
            nano_logits, nano_seq = nano_next_logits(llm)
            hf_logits = hf_next_logits(model, hf_input_ids)

            nano_argmax = int(nano_logits.argmax().item())
            hf_argmax = int(hf_logits.argmax().item())
            max_abs_diff = float((nano_logits - hf_logits).abs().max().item())
            mean_abs_diff = float((nano_logits - hf_logits).abs().mean().item())

            print(f"\nStep {step}")
            print(
                f"argmax nano={nano_argmax}({tokenizer.decode([nano_argmax])!r}) "
                f"hf={hf_argmax}({tokenizer.decode([hf_argmax])!r}) "
                f"same={nano_argmax == hf_argmax}"
            )
            print(f"max_abs_diff={max_abs_diff:.6f}  mean_abs_diff={mean_abs_diff:.6f}")
            print(f"nano top{args.topk}: {format_topk(tokenizer, nano_logits, args.topk)}")
            print(f"hf   top{args.topk}: {format_topk(tokenizer, hf_logits, args.topk)}")

            # Force both sides to consume the HF greedy token so subsequent contexts stay aligned.
            forced_token = hf_argmax
            llm.scheduler.postprocess([nano_seq], [forced_token], [0])
            hf_input_ids = torch.cat(
                [
                    hf_input_ids,
                    torch.tensor([[forced_token]], device=hf_input_ids.device),
                ],
                dim=1,
            )

            if nano_seq.is_finished:
                break
    finally:
        llm.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare nano-vllm logits against HuggingFace step by step"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=PROMPTS[0])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    main(parser.parse_args())
