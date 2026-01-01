#!/bin/bash

set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
OUTPUT_DIR=${2:-"training_outputs/nested_subnet/subnet_only"}
SUB_LAYERS=${3:-8}
SUB_START=${4:-0}
SUB_STRIDE=${5:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python exps_research/nested_subnet/train_subnet_only.py \
  --model_name "$MODEL" \
  --sub_layers "$SUB_LAYERS" \
  --sub_start "$SUB_START" \
  --sub_stride "$SUB_STRIDE" \
  --output_dir "$OUTPUT_DIR" \
  --max_train_samples 64 \
  --train_steps 200 \
  --lr 2e-4 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --seed 42
