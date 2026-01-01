#!/bin/bash
# (3) variant (i) subnet-only LoRA

set -e

# Add project root to PYTHONPATH so exps_research can be imported
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

DEVICE=${CUDA_VISIBLE_DEVICES:-0}
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
# QA_DATASETS=(data_processor/qa_dataset/test/*.json)
MATH_DATASETS=(data_processor/math_dataset/test/*.json)

TOTAL_LAYERS=$(python - <<PY
from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("${MODEL_ID}")
    print(getattr(config, "num_hidden_layers", 0))
except Exception:
    print(0)
PY
)
if [[ -z "$TOTAL_LAYERS" || "$TOTAL_LAYERS" -le 0 ]]; then
  TOTAL_LAYERS=16
fi
HALF_LAYERS=$((TOTAL_LAYERS / 2))
SUBNET_CONFIGS=(
  "first_half:0:1"
  "second_half:${HALF_LAYERS}:1"
  "stepping_half:0:2"
)

score_nested_math() {
  local dataset_path=$1
  local results_path=$2
  python - <<PY
import json
from pathlib import Path

from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal
from exps_research.unified_framework.math_utils.qwen_math_parser import extract_answer

dataset = json.loads(Path("${dataset_path}").read_text())
results = json.loads(Path("${results_path}").read_text())

examples = {row["id"]: row for row in dataset.get("examples", [])}
correct = 0
total = 0

for row in results:
  example = examples.get(row["id"])
  if not example:
    continue
  predicted = row.get("answer", "")
  gold = example.get("answer", "")
  if isinstance(predicted, str) and "boxed" in predicted:
    predicted = extract_answer(predicted)
  correct += int(math_equal(str(predicted), str(gold), timeout=True))
  total += 1

accuracy = (correct / total) if total else 0.0
print(f"Accuracy: {accuracy:.2%} ({correct}/{total}) -> ${results_path}")
PY
}

# Train and evaluate subnet-only LoRA variants
for config in "${SUBNET_CONFIGS[@]}"; do
  IFS=":" read -r name sub_start sub_stride <<< "$config"
  output_dir="training_outputs/nested_subnet/subnet_only_${name}"
  CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/training/train_subnet_only_toy.sh \
    "$MODEL_ID" \
    "$output_dir" \
    "$HALF_LAYERS" \
    "$sub_start" \
    "$sub_stride"
  for dataset in "${MATH_DATASETS[@]}"; do
    dataset_base=$(basename "$dataset" .json)
    output_path="training_outputs/nested_subnet/subnet_only_${name}_${dataset_base}.json"
    CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/nested_subnet/disagreement_infer.py \
      --model_name "$MODEL_ID" \
      --adapter_path "$output_dir" \
      --mode subnet_only \
      --policy sub_only \
      --sub_layers "$HALF_LAYERS" \
      --sub_start "$sub_start" \
      --sub_stride "$sub_stride" \
      --max_eval_samples 0 \
      --eval_path "$dataset" \
      --output_path "$output_path"
    score_nested_math "$dataset" "$output_path"
  done
done

