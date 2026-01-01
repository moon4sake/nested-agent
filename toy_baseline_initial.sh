#!/bin/bash
# (1) baseline initial full
# Full model accuracy on agent-prompt math datasets.

set -e

# Add project root to PYTHONPATH so exps_research can be imported
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

DEVICE=${CUDA_VISIBLE_DEVICES:-0}
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
LOG_FOLDER="training_outputs/baselines/baseline_initial/qa_results"
# QA_DATASETS=(data_processor/qa_dataset/test/*.json)
MATH_DATASETS=(data_processor/math_dataset/test/*.json)

run_unified_eval() {
  local model_id=$1
  local lora_dir=$2

#   for dataset in "${QA_DATASETS[@]}"; do
#     if [[ -n "$lora_dir" ]]; then
#       CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
#         --experiment_type agent \
#         --model_type vllm \
#         --model_id "$model_id" \
#         --fine_tuned \
#         --lora_folder "$lora_dir" \
#         --use_local_model \
#         --data_path "$dataset"
#     else
#       CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
#         --experiment_type agent \
#         --model_type vllm \
#         --model_id "$model_id" \
#         --use_local_model \
#         --data_path "$dataset"
#     fi
#   done

  for dataset in "${MATH_DATASETS[@]}"; do
    if [[ -n "$lora_dir" ]]; then
      CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
        --experiment_type agent \
        --model_type vllm \
        --model_id "$model_id" \
        --fine_tuned \
        --lora_folder "$lora_dir" \
        --use_local_model \
        --log_folder "$LOG_FOLDER" \
        --suffix baseline_initial \
        --data_path "$dataset"
    else
      CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
        --experiment_type agent \
        --model_type vllm \
        --model_id "$model_id" \
        --use_local_model \
        --log_folder "$LOG_FOLDER" \
        --suffix baseline_initial \
        --data_path "$dataset"
    fi
  done
}

# Run baseline initial full model evaluation
run_unified_eval "$MODEL_ID" ""
