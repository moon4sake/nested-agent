#!/bin/bash

set -e
set -x

MODE="baseline_full"
if [[ "$1" == "--mode" ]]; then
  MODE="$2"
fi

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SUBNET_ADAPTER="training_outputs/nested_subnet/subnet_only"
JOINT_MODEL="training_outputs/nested_subnet/joint_preserve"

POLICY="sub_only"
ADAPTER_PATH=""
MODEL_PATH="$MODEL"

if [[ "$MODE" == "subnet_only" ]]; then
  POLICY="sub_only"
  ADAPTER_PATH="$SUBNET_ADAPTER"
elif [[ "$MODE" == "joint_preserve" ]]; then
  POLICY="disagreement_escalate"
  MODEL_PATH="$JOINT_MODEL"
elif [[ "$MODE" == "baseline_full" ]]; then
  POLICY="sub_only"
else
  echo "Unknown mode: $MODE"
  exit 1
fi

python exps_research/nested_subnet/disagreement_infer.py \
  --model_name "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --mode "$MODE" \
  --policy "$POLICY" \
  --sub_layers 8 \
  --K 4 \
  --tau 1.5 \
  --temperature 0.4 \
  --max_eval_samples 25 \
  --output_path "training_outputs/nested_subnet/${MODE}_eval.json"
