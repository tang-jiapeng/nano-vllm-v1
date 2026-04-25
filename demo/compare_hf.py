import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams


PROMPTS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "你知道原神吗，请你介绍一下这款游戏",
]


def build_chat_prompts(tokenizer, enable_thinking: bool) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in PROMPTS
    ]


def run_nanovllm(model_path: str, prompts: list[str], max_tokens: int) -> list[str]:
    llm = LLM(model_path, enforce_eager=True)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    return [o["text"] for o in outputs]


@torch.inference_mode()
def run_hf(
    model_path: str,
    tokenizer,
    prompts: list[str],
    max_tokens: int,
) -> list[str]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda",
    )

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0, input_len:]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
    return outputs


def main(args):
    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = build_chat_prompts(tokenizer, args.enable_thinking)

    print("Running nano-vllm...")
    nano_outputs = run_nanovllm(model_path, prompts, args.max_tokens)

    print("Running HuggingFace...")
    hf_outputs = run_hf(model_path, tokenizer, prompts, args.max_tokens)

    for prompt, nano_text, hf_text in zip(prompts, nano_outputs, hf_outputs):
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt!r}")
        print(f"nano-vllm: {nano_text!r}")
        print(f"HF       : {hf_text!r}")
        print(f"same     : {nano_text == hf_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare nano-vllm Qwen3-MoE outputs against HuggingFace"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-tokens", type=int, default=128)
    main(parser.parse_args())
