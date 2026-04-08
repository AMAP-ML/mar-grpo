#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash run_batch_eval.sh <BASE_DIR>                      # 自动评估 BASE_DIR 下所有一级子目录
#   bash run_batch_eval.sh <BASE_DIR> expA foo bar         # 只评估指定子目录（相对 BASE_DIR）
#   bash run_batch_eval.sh <BASE_DIR> /abs/path/to/subdir  # 也可传绝对路径
#   bash run_batch_eval.sh <BASE_DIR> 200 400 900 1300     # 兼容旧用法：数字映射到 BASE_DIR/iter<数字>（若存在）
#
# 说明：
#   每个子目录的评估日志保存为 <BASE_DIR>/<子目录名>.txt
#   结果文件保存为 <BASE_DIR>/results_<子目录名>.jsonl 和 <BASE_DIR>/summary_<子目录名>.txt

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 <BASE_DIR> [SUBDIR_OR_PATH ...]"
  exit 1
fi

BASE_DIR="$1"; shift

GENEVAL_DIR="$GENEVAL_ROOT"
EVAL_SCRIPT="${GENEVAL_DIR}/evaluation/evaluate_images.py"     # 需支持 --outfile
SUMMARY_SCRIPT="${GENEVAL_DIR}/evaluation/summary_scores.py"
PYTHON="python"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "[ERR] BASE_DIR not found: $BASE_DIR"
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERR] evaluate_images.py not found: $EVAL_SCRIPT"
  exit 1
fi
if [[ ! -f "$SUMMARY_SCRIPT" ]]; then
  echo "[ERR] summary_scores.py not found: $SUMMARY_SCRIPT"
  exit 1
fi

echo "[INFO] Evaluating GenEval under: $BASE_DIR"

declare -a TARGET_DIRS=()

# 将用户输入解析为实际目录：
#  - 绝对路径：直接使用
#  - 纯数字 n：若存在 BASE_DIR/itern 则优先用它，否则尝试 BASE_DIR/n
#  - 其他名字：尝试 BASE_DIR/<name>
_resolve_dir() {
  local name="$1"
  if [[ "$name" = /* ]]; then
    [[ -d "$name" ]] && { echo "$name"; return 0; } || return 1
  fi
  if [[ "$name" =~ ^[0-9]+$ ]] && [[ -d "$BASE_DIR/iter${name}" ]]; then
    echo "$BASE_DIR/iter${name}"; return 0
  fi
  if [[ -d "$BASE_DIR/$name" ]]; then
    echo "$BASE_DIR/$name"; return 0
  fi
  return 1
}

if [[ $# -gt 0 ]]; then
  # 显式指定子目录名/绝对路径/数字
  for n in "$@"; do
    if d="$(_resolve_dir "$n")"; then
      TARGET_DIRS+=("$d")
    else
      echo "[WARN] folder not found: $n"
    fi
  done
else
  # 未指定时：自动收集 BASE_DIR 下所有一级子目录
  while IFS= read -r -d '' d; do
    TARGET_DIRS+=("$d")
  done < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  echo "[WARN] No subfolders to evaluate under $BASE_DIR"
  exit 0
fi

for dir in "${TARGET_DIRS[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "[SKIP] Not a directory: $dir"
    continue
  fi

  tag="$(basename "$dir")"                            # 用子目录名
  log_txt="${BASE_DIR}/${tag}.txt"                   # 日志
  out_json="${BASE_DIR}/results_${tag}.jsonl"        # 结果 JSONL
  out_summary="${BASE_DIR}/summary_${tag}.txt"       # 汇总文本

  echo "================ Evaluating ${tag} ================"
  echo "[`date '+%F %T'`] Input Dir : ${dir}"
  echo "[`date '+%F %T'`] Out JSON  : ${out_json}"
  echo "[`date '+%F %T'`] Summary   : ${out_summary}"

  # 跑评估（把 stdout+stderr 也写进日志）
  "${PYTHON}" "${EVAL_SCRIPT}" "${dir}" --outfile "${out_json}" 2>&1 | tee "${log_txt}"

  # 生成汇总
  "${PYTHON}" "${SUMMARY_SCRIPT}" "${out_json}" > "${out_summary}" 2>>"${log_txt}" || true

  echo "[`date '+%F %T'`] Done ${tag}. -> ${out_json} | ${out_summary}"
done

echo "[DONE] Logs:     ${BASE_DIR}/*.txt"
echo "[DONE] Results:  ${BASE_DIR}/results_*.jsonl"
echo "[DONE] Summary:  ${BASE_DIR}/summary_*.txt"
