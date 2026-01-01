import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset


@dataclass
class ToyBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def load_toy_tool_dataset(max_samples: int) -> List[dict]:
    try:
        dataset = load_dataset("agent-distillation/smolagents-toy-code")["train"]
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset.to_list()
    except Exception:
        fallback = load_dataset("gsm8k", "main", split="train")
        if max_samples > 0:
            fallback = fallback.select(range(min(max_samples, len(fallback))))
        return [{"question": row["question"], "answer": row["answer"]} for row in fallback]


def load_toy_general_dataset(name: str, max_samples: int) -> List[str]:
    if name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        dataset = load_dataset(name, split="train")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [row["text"] for row in dataset]


def tokenize_messages(tokenizer, messages: List[dict], max_length: int) -> ToyBatch:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokens["input_ids"].clone()
    labels[tokens["attention_mask"] == 0] = -100
    return ToyBatch(tokens["input_ids"], tokens["attention_mask"], labels)


def tokenize_text(tokenizer, text: str, max_length: int) -> ToyBatch:
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokens["input_ids"].clone()
    labels[tokens["attention_mask"] == 0] = -100
    return ToyBatch(tokens["input_ids"], tokens["attention_mask"], labels)


def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    return F.kl_div(log_q, p, reduction="batchmean", log_target=False)


def save_training_metadata(path: str, metadata: dict) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
