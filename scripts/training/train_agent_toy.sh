#!/bin/bash

set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
DATASET=${2:-"agent-distillation/Qwen2.5-32B-Instruct_agent_trajectories_2k"}

python exps_research/finetune_sft.py \
  --model_name "$MODEL" \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --lr 2e-4 \
  --train_filepath "$DATASET" \
  --solution_type agent \
  --max_length 2048 \
  --max_train_samples 64 \
  --max_eval_samples 25 \
  --exp_id toy
