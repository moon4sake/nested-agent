#!/bin/bash

set -e
set -x

MODEL=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}
ADAPTER_PATH=${2:-"training_outputs/qwen-0.5B-instruct/agent_toy"}

# Add project root to PYTHONPATH so exps_research can be imported
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python exps_research/nested_subnet/disagreement_infer.py \
  --model_name "$MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --mode baseline_full \
  --policy sub_only \
  --sub_layers 8 \
  --max_eval_samples 25 \
  --output_path "training_outputs/nested_subnet/agent_distilled_eval.json"
