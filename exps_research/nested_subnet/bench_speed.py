import argparse
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from exps_research.nested_subnet.subnet_factory import build_subnet, maybe_load_peft_adapter


def measure_tokens_per_sec(model, tokenizer, prompt: str, max_new_tokens: int) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
    return tokens / max(elapsed, 1e-6)


def load_model(model_name, adapter_path=None):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if adapter_path:
        model = maybe_load_peft_adapter(model, adapter_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def main(args):
    cfg = yaml.safe_load(open(args.config, "r"))
    prompt = cfg.get("prompt", "Action: compute 2+2 using python")
    max_new_tokens = cfg.get("max_new_tokens", 64)
    model_name = cfg.get("model_name")
    subnet_layers = cfg.get("sub_layers", 8)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", add_eos_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []

    base_model = load_model(model_name)
    rows.append(("base_full", measure_tokens_per_sec(base_model, tokenizer, prompt, max_new_tokens)))

    agent_path = cfg.get("agent_distilled_path")
    if agent_path:
        agent_model = load_model(model_name, adapter_path=agent_path)
        rows.append(("agent_distilled_full", measure_tokens_per_sec(agent_model, tokenizer, prompt, max_new_tokens)))

    subnet_adapter = cfg.get("subnet_adapter_path")
    if subnet_adapter:
        subnet_model = load_model(model_name, adapter_path=subnet_adapter)
    else:
        subnet_model = load_model(model_name)
    subnet_model = build_subnet(subnet_model, subnet_layers)
    rows.append(("subnet", measure_tokens_per_sec(subnet_model, tokenizer, prompt, max_new_tokens)))

    joint_path = cfg.get("joint_model_path")
    if joint_path:
        joint_model = load_model(joint_path)
        rows.append(("joint_full", measure_tokens_per_sec(joint_model, tokenizer, prompt, max_new_tokens)))

    print("name\tokens_per_sec\tms_per_token")
    for name, tps in rows:
        ms = 1000.0 / max(tps, 1e-6)
        print(f"{name}\t{tps:.2f}\t{ms:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
