#!/usr/bin/env bash
set -euo pipefail

BASE="$WORKSPACE_ROOT/arsample_temperature/hps"

for t in $(seq 0.1 0.1 1.5); do
  t_fmt=$(printf "%.1f" "$t")
  out="${BASE}/temp${t_fmt}"
  mkdir -p "$out"

  echo ">> Running temperature=${t_fmt} -> ${out}"
  torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=29501 \
    reason_inference_hps.py --temperature "${t_fmt}" \
    --save_root "${out}"
done
