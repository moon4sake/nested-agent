import argparse
import os
import tempfile

from exps_research.nested_subnet.subnet_factory import SubnetSpec, build_subnet, load_full_model_and_tokenizer
from exps_research.nested_subnet.train_subnet_only import main as train_subnet_only
from exps_research.nested_subnet.train_joint_preserve import main as train_joint_preserve


def run_generate(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=8)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    spec = SubnetSpec(model_name="Qwen/Qwen2.5-0.5B-Instruct", sub_layers=2)
    full_model, tokenizer = load_full_model_and_tokenizer(spec)
    sub_model = build_subnet(full_model, spec.sub_layers)

    full_out = run_generate(full_model, tokenizer, "Hello")
    sub_out = run_generate(sub_model, tokenizer, "Hello")
    print("Full output:", full_out[:50])
    print("Subnet output:", sub_out[:50])

    with tempfile.TemporaryDirectory() as tmpdir:
        subnet_dir = os.path.join(tmpdir, "subnet_only")
        joint_dir = os.path.join(tmpdir, "joint")

        train_subnet_only(
            argparse.Namespace(
                model_name=spec.model_name,
                sub_layers=spec.sub_layers,
                output_dir=subnet_dir,
                max_train_samples=4,
                seed=123,
                train_steps=1,
                lr=2e-4,
                lora_rank=4,
                lora_alpha=8,
                lora_dropout=0.05,
                max_length=128,
                log_every=1,
            )
        )
        train_joint_preserve(
            argparse.Namespace(
                model_name=spec.model_name,
                sub_layers=spec.sub_layers,
                output_dir=joint_dir,
                max_train_samples=4,
                max_gen_samples=4,
                gen_dataset_name="wikitext",
                seed=123,
                train_steps=1,
                lr=2e-5,
                beta_preserve=0.1,
                alpha_kd=0.0,
                preserve_every=1,
                max_length=128,
                log_every=1,
            )
        )

        assert os.path.exists(os.path.join(subnet_dir, "subnet_metadata.json"))
        assert os.path.exists(os.path.join(joint_dir, "subnet_metadata.json"))
        print("Sanity test passed.")


if __name__ == "__main__":
    main()
