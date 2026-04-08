#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash run_batch_eval.sh <BASE_DIR>                      # 自动评估 BASE_DIR 下所有一级子目录
#   bash run_batch_eval.sh <BASE_DIR> expA foo bar         # 只评估指定子目录（相对 BASE_DIR）
#   bash run_batch_eval.sh <BASE_DIR> /abs/path/to/subdir  # 也可传绝对路径
#   bash run_batch_eval.sh <BASE_DIR> 200 400 900 1300     # 兼容旧用法：数字会映射到 BASE_DIR/iter<数字>（若存在）
#
# 说明：
#   会把每个子目录的评估日志保存为 <BASE_DIR>/<子目录名>.txt（例如 iter1600.txt / expA.txt）

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 <BASE_DIR> [SUBDIR_OR_PATH ...]"
  exit 1
fi

BASE_DIR="$1"; shift
EVAL_SCRIPT="$PROJECT_ROOT/reference_code/HPSv2/evaluation.py"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "[ERR] BASE_DIR not found: $BASE_DIR"
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERR] evaluation.py not found: $EVAL_SCRIPT"
  exit 1
fi

echo "[INFO] Evaluating HPS under: $BASE_DIR"

declare -a TARGET_DIRS=()

# 将用户输入解析为实际目录：
#  - 绝对路径：直接使用
#  - 纯数字 n：若存在 BASE_DIR/itern 则优先用它，否则尝试 BASE_DIR/n
#  - 其他名字：尝试 BASE_DIR/<name>
_resolve_dir() {
  local name="$1"
  # 绝对路径
  if [[ "$name" = /* ]]; then
    [[ -d "$name" ]] && { echo "$name"; return 0; } || return 1
  fi
  # 兼容旧用法：数字 -> 优先 BASE_DIR/iter<NUM>
  if [[ "$name" =~ ^[0-9]+$ ]] && [[ -d "$BASE_DIR/iter${name}" ]]; then
    echo "$BASE_DIR/iter${name}"; return 0
  fi
  # 尝试 BASE_DIR/<name>
  if [[ -d "$BASE_DIR/$name" ]]; then
    echo "$BASE_DIR/$name"; return 0
  fi
  return 1
}

if [[ $# -gt 0 ]]; then
  # 用户显式指定子目录名或绝对路径
  for n in "$@"; do
    if d="$(_resolve_dir "$n")"; then
      TARGET_DIRS+=("$d")
    else
      echo "[WARN] folder not found: $n"
    fi
  done
else
  # 未指定时，自动收集 BASE_DIR 下所有一级子目录（可按需排除隐藏目录）
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
  tag="$(basename "$dir")"                # 用子目录名作为日志文件名
  out_txt="$BASE_DIR/${tag}.txt"

  echo "================ Evaluating ${tag} ================"
  # 将 stdout+stderr 都写入该目录的日志，同时在屏幕打印
  python "$EVAL_SCRIPT" "$dir" 2>&1 | tee "$out_txt"
done

echo "[DONE] Results saved as: $BASE_DIR/<subdir>.txt"
