# Agent Distillation

This repository contains research code for distilling tool-using LLM agents into smaller models, including toy and baseline experiments that reproduce key results from the Agent Distillation paper (arXiv:2505.17612). It builds on the `smolagents` framework and provides scripts for collecting trajectories, fine-tuning student models, and evaluating distilled agents with retrieval and code tools.

## Quick start

```bash
pip install -e .[distill]
```

See the scripts in `scripts/` and `exps_research/` for training, inference, and benchmarking workflows.

# install
pip install -e .[distill]

# (1) baseline initial full
bash scripts/inference/run_nested_toy.sh --mode baseline_full

# (2) baseline agent-distilled full
bash scripts/training/train_agent_toy.sh
bash scripts/inference/run_agent_student_toy.sh

# (3) variant (i) subnet-only LoRA
bash scripts/training/train_subnet_only_toy.sh
bash scripts/inference/run_nested_toy.sh --mode subnet_only

# (4) variant (ii) joint preserve
bash scripts/training/train_joint_preserve_toy.sh
bash scripts/inference/run_nested_toy.sh --mode joint_preserve

# (5) speed benchmark
python exps_research/nested_subnet/bench_speed.py --config exps_research/nested_subnet/configs/toy.yaml

# (6) sanity test (subnet + full + 1-step training for both variants)
python exps_research/nested_subnet/sanity_test_toy.py

python exps_research/nested_subnet/sanity_test_toy.py

