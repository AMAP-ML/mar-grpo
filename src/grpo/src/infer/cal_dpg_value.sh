#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash run_batch_eval.sh <SAVE_BASE>                     # 自动评估 SAVE_BASE 下所有 iter*
#   bash run_batch_eval.sh <SAVE_BASE> 200 400 900 1300    # 只评估指定 iter

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <SAVE_BASE> [iter_numbers...]"
  exit 1
fi

SAVE_BASE="$(realpath -m "$1")"; shift
if [[ ! -d "$SAVE_BASE" ]]; then
  echo "[ERROR] SAVE_BASE not found: $SAVE_BASE"; exit 1
fi

# GenEval 目录与脚本（若你用的是 evaluate_images_hl.py，请改成对应脚本路径）
GENEVAL_DIR="$DPG_BENCH_ROOT"
EVAL_SCRIPT="dpg_bench/dist_eval.sh"     # 该脚本支持 --outfile
RESOLUTION=384

# 输出目录
EVAL_LOG_DIR="${SAVE_BASE}/eval_logs"
EVAL_RES_DIR="${SAVE_BASE}/eval_results"
mkdir -p "${EVAL_LOG_DIR}" "${EVAL_RES_DIR}"

# 以 SAVE_BASE 的末级目录名作为区分标识
OUT_TAG="$(basename "${SAVE_BASE}")"

# 解析 iter 列表
ITER_LIST=()
if [[ $# -gt 0 ]]; then
  for n in "$@"; do ITER_LIST+=("$n"); done
else
  shopt -s nullglob
  for d in "${SAVE_BASE}"/iter*/; do
    base="$(basename "$d")"           # e.g. iter200
    num="${base#iter}"                # e.g. 200
    [[ "$num" =~ ^[0-9]+$ ]] && ITER_LIST+=("$num")
  done
  shopt -u nullglob
fi

if [[ ${#ITER_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] No iter folders found under $SAVE_BASE"; exit 1
fi

pushd "${GENEVAL_DIR}" >/dev/null

# ITER_LIST=(200 400 500 900 1000 1100 1200 1300 1600)
# ITER_LIST=(100 300 600 700 800 1000 1100 1200 1300 1600)

for it in "${ITER_LIST[@]}"; do
  IMG_DIR="${SAVE_BASE}/iter${it}"
  LOG_FILE="${EVAL_LOG_DIR}/eval_iter${it}.log"
  OUT_JSON="${EVAL_RES_DIR}/results_${OUT_TAG}_iter${it}.jsonl"
  OUT_SUMMARY_TXT="${EVAL_RES_DIR}/summary_${OUT_TAG}_iter${it}.txt"

  if [[ ! -d "${IMG_DIR}" ]]; then
    echo "[WARN] skip iter ${it}, not found: ${IMG_DIR}"
    continue
  fi

  echo "========================================================" | tee -a "${LOG_FILE}"
  echo "[`date '+%F %T'`] Evaluating iter${it}" | tee -a "${LOG_FILE}"
  echo "Image Dir : ${IMG_DIR}" | tee -a "${LOG_FILE}"
  echo "Out JSON  : ${OUT_JSON}" | tee -a "${LOG_FILE}"
  echo "========================================================" | tee -a "${LOG_FILE}"

  # 直接把 --outfile 指向唯一文件名（绝对路径），避免并发冲突/覆盖
  "${EVAL_SCRIPT}" "${IMG_DIR}" "${RESOLUTION}" 2>&1 | tee -a "${LOG_FILE}"

  echo "[`date '+%F %T'`] Done iter${it}. -> ${OUT_JSON} | ${OUT_SUMMARY_TXT}" | tee -a "${LOG_FILE}"
done

popd >/dev/null

echo "[INFO] All evaluations done."
echo "JSON:    ${EVAL_RES_DIR}/results_${OUT_TAG}_iter*.jsonl"
echo "Summary: ${EVAL_RES_DIR}/summary_${OUT_TAG}_iter*.txt"
echo "Logs:    ${EVAL_LOG_DIR}/eval_iter*.log"
