#!/usr/bin/env bash
set -euo pipefail

# 用法检查与帮助
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <EXP_NAME> [CKPT1 CKPT2 ...]"
  echo "  e.g. $0 exp_name                # 用默认的 checkpoint 列表"
  echo "  e.g. $0 exp_name 200 400 900    # 只跑指定的 checkpoint"
  exit 1
fi

########################################
# 可配置项
########################################
EXP_NAME="$1"
shift  # 剩余参数全部视为 checkpoint 列表（如果有）

# 模型 checkpoint 根目录（不含 checkpoint-xxx）
MODEL_ROOT="$WORKSPACE_ROOT/outputs_geneval_v3/${EXP_NAME}"

# 结果保存根目录（每个 ckpt 会保存到 ${SAVE_BASE}/iter${ckpt}）
SAVE_BASE="$WORKSPACE_ROOT/final_results/geneval_reward/geneval/${EXP_NAME}"

# 分布式参数
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=29500
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 如需固定 GPU

########################################
# checkpoint 列表
########################################
DEFAULT_CHECKPOINTS=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600)

if [[ $# -gt 0 ]]; then
  # 用户显式指定的 ckpt 列表
  CHECKPOINTS=("$@")
else
  # 使用默认 ckpt 列表
  CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

# 调试输出（可选）
echo "[INFO] EXP_NAME      = ${EXP_NAME}"
echo "[INFO] MODEL_ROOT    = ${MODEL_ROOT}"
echo "[INFO] SAVE_BASE     = ${SAVE_BASE}"
echo "[INFO] CHECKPOINTS   = ${CHECKPOINTS[*]}"


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
