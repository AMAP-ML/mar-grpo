#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4

BASE_CKPT_DIR="$PROJECT_ROOT/workspace_t2i-r1_harmon/t2i-r1-harmon-test-beta0.01_25diff_test6"
SAVE_BASE_DIR="$PROJECT_ROOT/workspace_harmon_new_evaluations/hps/t2i-r1-harmon-test-beta0.01_25diff_test6"

CHECKPOINTS=(100 150 200 250 300 350)

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "=== Running inference for checkpoint-${ckpt} ==="

    torchrun \
        --nnodes=1 \
        --nproc_per_node=4 \
        --node_rank=0 \
        --master_port=29502 \
        infer_hps_harmon.py \
        --transformer_path "${BASE_CKPT_DIR}/checkpoint-${ckpt}" \
        --save_root "${SAVE_BASE_DIR}/iter${ckpt}"

    echo "=== Finished checkpoint-${ckpt} ==="
done
