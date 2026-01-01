# Agent Distillation

This repository contains research code for distilling tool-using LLM agents into smaller models, including toy and baseline experiments that reproduce key results from the Agent Distillation paper (arXiv:2505.17612). It builds on the `smolagents` framework and provides scripts for collecting trajectories, fine-tuning student models, and evaluating distilled agents with retrieval and code tools.

## Quick start

```bash
pip install -e .[distill]
```

See the scripts in `scripts/` and `exps_research/` for training, inference, and benchmarking workflows.