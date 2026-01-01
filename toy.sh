# # install
# pip install -e .[distill]

DEVICE=7
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
LORA_DIR="training_outputs/qwen-0.5B-instruct/agent_toy"
QA_DATASETS=(data_processor/qa_dataset/test/*.json)
MATH_DATASETS=(data_processor/math_dataset/test/*.json)

# (1) baseline initial full
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/inference/run_nested_toy.sh --mode baseline_full

# (2) baseline agent-distilled full
CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/training/train_agent_toy.sh

for dataset in "${QA_DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
    --experiment_type agent \
    --model_type vllm \
    --model_id "$MODEL_ID" \
    --fine_tuned \
    --lora_folder "$LORA_DIR" \
    --use_local_model \
    --data_path "$dataset"
done

for dataset in "${MATH_DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$DEVICE python exps_research/unified_framework/run_experiment.py \
    --experiment_type reasoning \
    --task_type math \
    --model_type vllm \
    --model_id "$MODEL_ID" \
    --fine_tuned \
    --lora_folder "$LORA_DIR" \
    --use_local_model \
    --data_path "$dataset"
done

# # (3) variant (i) subnet-only LoRA
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/training/train_subnet_only_toy.sh
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/inference/run_nested_toy.sh --mode subnet_only

# # (4) variant (ii) joint preserve
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/training/train_joint_preserve_toy.sh
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/inference/run_nested_toy.sh --mode joint_preserve

# # (5) speed benchmark
# python exps_research/nested_subnet/bench_speed.py --config exps_research/nested_subnet/configs/toy.yaml

# # (6) sanity test (subnet + full + 1-step training for both variants)
# python exps_research/nested_subnet/sanity_test_toy.py

# python exps_research/nested_subnet/sanity_test_toy.py
