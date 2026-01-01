import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exps_research.nested_subnet.subnet_factory import (
    SubnetSpec,
    build_subnet,
    maybe_load_peft_adapter,
    set_adapter_enabled,
)


def load_math_questions(path: str, max_samples: int) -> List[dict]:
    obj = json.loads(Path(path).read_text())
    examples = obj.get("examples", [])
    if max_samples > 0:
        examples = examples[:max_samples]
    return examples


def format_prompt(question: str) -> List[dict]:
    return [{"role": "user", "content": question}]


def generate_answer(model, tokenizer, messages, temperature, max_new_tokens, seed):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split(messages[-1]["content"])[-1].strip()


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", add_eos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = maybe_load_peft_adapter(base_model, args.adapter_path)
    base_model.to(args.device)

    if args.policy in {"sub_only", "disagreement_escalate"} and args.mode != "baseline_full":
        sub_model = build_subnet(base_model, args.sub_layers)
    else:
        sub_model = None

    if args.adapter_path:
        set_adapter_enabled(base_model, args.mode != "baseline_full")

    questions = load_math_questions(args.eval_path, args.max_eval_samples)
    results = []
    for idx, sample in enumerate(questions):
        messages = format_prompt(sample["question"])
        if args.mode == "baseline_full":
            answer = generate_answer(
                base_model,
                tokenizer,
                messages,
                args.temperature,
                args.max_new_tokens,
                args.seed + idx,
            )
            used = "full"
        elif args.policy == "sub_only":
            answer = generate_answer(
                sub_model,
                tokenizer,
                messages,
                args.temperature,
                args.max_new_tokens,
                args.seed + idx,
            )
            used = "subnet"
        elif args.policy == "disagreement_escalate":
            candidates = []
            for k in range(args.K):
                candidates.append(
                    generate_answer(
                        sub_model,
                        tokenizer,
                        messages,
                        args.temperature,
                        args.max_new_tokens,
                        args.seed + idx + k,
                    )
                )
            unique = {c.strip() for c in candidates if c.strip()}
            if len(unique) > args.tau:
                answer = generate_answer(
                    base_model,
                    tokenizer,
                    messages,
                    args.temperature,
                    args.max_new_tokens,
                    args.seed + idx + args.K,
                )
                used = "full"
            else:
                answer = candidates[0]
                used = "subnet"
        else:
            answer = generate_answer(
                base_model,
                tokenizer,
                messages,
                args.temperature,
                args.max_new_tokens,
                args.seed + idx,
            )
            used = "full"
        results.append({"id": sample["id"], "answer": answer, "used": used})

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} outputs to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--mode", default="baseline_full", choices=["baseline_full", "subnet_only", "joint_preserve"])
    parser.add_argument("--policy", default="sub_only", choices=["sub_only", "disagreement_escalate"])
    parser.add_argument("--sub_layers", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_eval_samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_path", default="data_processor/math_dataset/test/math_500_20250414.json")
    parser.add_argument("--output_path", default="training_outputs/nested_subnet/toy_eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
