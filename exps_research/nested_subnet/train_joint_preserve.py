import argparse
import os
import random
from typing import List

import torch

from exps_research.nested_subnet.subnet_factory import (
    SubnetSpec,
    build_subnet,
    load_full_model_and_tokenizer,
)
from exps_research.nested_subnet.train_utils import (
    compute_kl_divergence,
    load_toy_general_dataset,
    load_toy_tool_dataset,
    save_training_metadata,
    tokenize_messages,
    tokenize_text,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = SubnetSpec(
        model_name=args.model_name,
        sub_layers=args.sub_layers,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    full_model, tokenizer = load_full_model_and_tokenizer(spec)
    subnet = build_subnet(full_model, args.sub_layers)

    ref_model, _ = load_full_model_and_tokenizer(spec)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    tool_data = load_toy_tool_dataset(args.max_train_samples)
    general_data = load_toy_general_dataset(args.gen_dataset_name, args.max_gen_samples)

    optimizer = torch.optim.AdamW(full_model.parameters(), lr=args.lr)
    full_model.train()
    subnet.train()

    os.makedirs(args.output_dir, exist_ok=True)

    for step in range(args.train_steps):
        tool_row = tool_data[step % len(tool_data)]
        messages = _row_to_messages(tool_row)
        tool_batch = tokenize_messages(tokenizer, messages, args.max_length)
        tool_batch = tool_batch.input_ids.to(device), tool_batch.attention_mask.to(device), tool_batch.labels.to(device)

        optimizer.zero_grad()

        sub_outputs = subnet(
            input_ids=tool_batch[0],
            attention_mask=tool_batch[1],
            labels=tool_batch[2],
        )
        loss = sub_outputs.loss

        if args.alpha_kd > 0:
            with torch.no_grad():
                full_outputs = full_model(
                    input_ids=tool_batch[0],
                    attention_mask=tool_batch[1],
                )
            kd_loss = compute_kl_divergence(full_outputs.logits, sub_outputs.logits)
            loss = loss + args.alpha_kd * kd_loss

        if step % args.preserve_every == 0:
            gen_text = general_data[step % len(general_data)]
            gen_batch = tokenize_text(tokenizer, gen_text, args.max_length)
            gen_batch = gen_batch.input_ids.to(device), gen_batch.attention_mask.to(device)
            with torch.no_grad():
                ref_logits = ref_model(
                    input_ids=gen_batch[0],
                    attention_mask=gen_batch[1],
                ).logits
            full_logits = full_model(
                input_ids=gen_batch[0],
                attention_mask=gen_batch[1],
            ).logits
            preserve_loss = compute_kl_divergence(ref_logits, full_logits)
            loss = loss + args.beta_preserve * preserve_loss

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
            "variant": "joint_preserve",
            "beta_preserve": args.beta_preserve,
            "alpha_kd": args.alpha_kd,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sub_layers", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=64)
    parser.add_argument("--max_gen_samples", type=int, default=1000)
    parser.add_argument("--gen_dataset_name", default="wikitext", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta_preserve", type=float, default=0.1)
    parser.add_argument("--alpha_kd", type=float, default=0.0)
    parser.add_argument("--preserve_every", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=10)
    main(parser.parse_args())
