#!/usr/bin/env bash
set -euo pipefail

########################################
# 可配置项
########################################
# $WORKSPACE_ROOT/outputs_geneval_v3/t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_hps_git_gdino
# $WORKSPACE_ROOT/outputs_geneval_v3/final_results/geneval_reward/t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_woaccu_beta0.03_baseline
EXP_NAME='t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_woaccu_beta0.03_baseline_wokl'
# 模型 checkpoint 根目录（不含 checkpoint-xxx）
# $USER_ROOT/workspace_t2i-r1_train_ckpts/outputs_geneval_v3/t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_woaccu_beta0.03_baseline_wo_entropy
MODEL_ROOT="$USER_ROOT/workspace_t2i-r1_train_ckpts/outputs_geneval_v3/${EXP_NAME}"

# 结果保存根目录（每个 ckpt 会保存到 ${SAVE_BASE}/iter${ckpt}）
SAVE_BASE="$USER_ROOT/workspace_t2i-r1/final_results/geneval_reward/geneval/${EXP_NAME}"
# SAVE_BASE="$USER_ROOT/workspace_t2i-r1/${EXP_NAME}_womask"

# 分布式参数
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=29500

# 若需要固定 GPU 可在此设置，例如：export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

########################################
# 读取 checkpoint 列表
########################################
# 用法1：直接在脚本内设置
DEFAULT_CHECKPOINTS=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600)

# 用法2：运行时传参覆盖（例如 bash run_batch_infer.sh 200 400 900 1300）
if [ "$#" -gt 0 ]; then
  CHECKPOINTS=("$@")
else
  CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

########################################
# 创建日志目录
########################################
LOG_DIR="${SAVE_BASE}/logs"
mkdir -p "${LOG_DIR}"

########################################
# 批量推理
########################################
for ckpt in "${CHECKPOINTS[@]}"; do
  CKPT_DIR="${MODEL_ROOT}/checkpoint-${ckpt}"
  SAVE_DIR="${SAVE_BASE}/iter${ckpt}"
  LOG_FILE="${LOG_DIR}/iter${ckpt}.log"

  if [ ! -d "${CKPT_DIR}" ]; then
    echo "[WARN] Checkpoint not found: ${CKPT_DIR}  (skip)"
    continue
  fi

  mkdir -p "${SAVE_DIR}"

  echo "========================================================" | tee -a "${LOG_FILE}"
  echo "[`date '+%F %T'`] Start inference for checkpoint-${ckpt}" | tee -a "${LOG_FILE}"
  echo "Model Path : ${CKPT_DIR}" | tee -a "${LOG_FILE}"
  echo "Save  Path : ${SAVE_DIR}"  | tee -a "${LOG_FILE}"
  echo "========================================================" | tee -a "${LOG_FILE}"

  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_port="${MASTER_PORT}" \
    reason_inference_geneval.py \
      --model_path "${CKPT_DIR}" \
      --save_root "${SAVE_DIR}" 2>&1 | tee -a "${LOG_FILE}"

  echo "[`date '+%F %T'`] Finished checkpoint-${ckpt}" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"
done

echo "All done. Logs in: ${LOG_DIR}"
