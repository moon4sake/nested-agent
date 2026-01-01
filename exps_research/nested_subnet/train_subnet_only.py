import argparse
import os
import random
from typing import List

import torch
from peft import LoraConfig, get_peft_model

from exps_research.nested_subnet.subnet_factory import (
    SubnetSpec,
    build_subnet,
    load_full_model_and_tokenizer,
    set_adapter_enabled,
)
from exps_research.nested_subnet.train_utils import (
    load_toy_tool_dataset,
    save_training_metadata,
    tokenize_messages,
)


def _row_to_messages(row: dict) -> List[dict]:
    if "messages" in row:
        return row["messages"]
    if "prompt" in row and "response" in row:
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    if "question" in row and "answer" in row:
        return [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
    return [{"role": "user", "content": str(row)}]


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    spec = SubnetSpec(
        model_name=args.model_name,
        sub_layers=args.sub_layers,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    full_model, tokenizer = load_full_model_and_tokenizer(spec)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    full_model = get_peft_model(full_model, peft_config)
    full_model.print_trainable_parameters()

    subnet = build_subnet(
        full_model,
        args.sub_layers,
        start_layer=args.sub_start,
        layer_stride=args.sub_stride,
    )
    set_adapter_enabled(full_model, True)

    train_data = load_toy_tool_dataset(args.max_train_samples)
    optimizer = torch.optim.AdamW(
        [p for p in full_model.parameters() if p.requires_grad], lr=args.lr
    )
    full_model.train()
    subnet.train()

    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    for step in range(args.train_steps):
        row = train_data[step % len(train_data)]
        messages = _row_to_messages(row)
        batch = tokenize_messages(tokenizer, messages, args.max_length)
        batch = batch.input_ids.to(spec.device), batch.attention_mask.to(spec.device), batch.labels.to(spec.device)

        optimizer.zero_grad()
        outputs = subnet(
            input_ids=batch[0],
            attention_mask=batch[1],
            labels=batch[2],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if (step + 1) % args.log_every == 0:
            print(f"step={step+1} loss={loss.item():.4f}")

    full_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_training_metadata(
        os.path.join(args.output_dir, "subnet_metadata.json"),
        {
            "model_name": args.model_name,
            "sub_layers": args.sub_layers,
            "sub_start": args.sub_start,
            "sub_stride": args.sub_stride,
            "variant": "subnet_only_lora",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sub_layers", type=int, default=8)
    parser.add_argument("--sub_start", type=int, default=0)
    parser.add_argument("--sub_stride", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=10)
    main(parser.parse_args())
