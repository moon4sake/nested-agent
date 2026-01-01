#!/bin/bash

set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
OUTPUT_DIR=${2:-"training_outputs/nested_subnet/subnet_only"}

python exps_research/nested_subnet/train_subnet_only.py \
  --model_name "$MODEL" \
  --sub_layers 8 \
  --output_dir "$OUTPUT_DIR" \
  --max_train_samples 64 \
  --train_steps 200 \
  --lr 2e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --seed 42
