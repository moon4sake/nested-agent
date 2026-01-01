# # install
# pip install -e .[distill]

DEVICE=7

# (1) baseline initial full
# CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/inference/run_nested_toy.sh --mode baseline_full

# (2) baseline agent-distilled full
CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/training/train_agent_toy.sh
CUDA_VISIBLE_DEVICES=$DEVICE bash scripts/inference/run_agent_student_toy.sh

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

