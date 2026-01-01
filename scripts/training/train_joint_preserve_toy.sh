#!/bin/bash

set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
OUTPUT_DIR=${2:-"training_outputs/nested_subnet/joint_preserve"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exps_research/nested_subnet/train_joint_preserve.py \
  --model_name "$MODEL" \
  --sub_layers 8 \
  --output_dir "$OUTPUT_DIR" \
  --max_train_samples 64 \
  --max_gen_samples 1000 \
  --gen_dataset_name wikitext \
  --train_steps 200 \
  --lr 2e-5 \
  --beta_preserve 0.1 \
  --alpha_kd 0.0 \
  --preserve_every 4 \
  --seed 42
