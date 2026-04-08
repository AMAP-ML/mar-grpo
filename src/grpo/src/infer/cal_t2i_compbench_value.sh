#!/usr/bin/env bash
set -euo pipefail

# 用法检查
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ROOT_DIR>"
  exit 1
fi

ROOT_DIR="$1"
T2I_COMP_CODE_ROOT="$PROJECT_ROOT/compare_models/T2I-CompBench"

# 如果 split 目录不存在：跳过
# 如果 split 目录存在但结果文件不存在：运行
# 如果结果文件已存在：跳过
run_if_split_exists () {
  local split_dir="$1"     # e.g. texture_val
  local check_rel="$2"     # e.g. annotation_blip/vqa_result.json
  shift 2
  local split_path="${ROOT_DIR}/${split_dir}"
  local check_path="${split_path}/${check_rel}"

  if [[ ! -d "${split_path}" ]]; then
    echo "[SKIP] ${split_dir} dir not found: ${split_path}"
    return 0
  fi

  if [[ -f "${check_path}" ]]; then
    echo "[SKIP] ${split_dir} already has $(basename "${check_rel}")"
    return 0
  fi

  echo "[RUN] ${split_dir} -> $*"
  "$@"
}

# # ## Attribute Binding:
cd "${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
# run_if_split_exists "texture_val" "annotation_blip/vqa_result.json" \
#   python BLIP_vqa.py --out_dir "${ROOT_DIR}/texture_val"

run_if_split_exists "color_val" "annotation_blip/vqa_result.json" \
  python BLIP_vqa.py --out_dir "${ROOT_DIR}/color_val"

# run_if_split_exists "shape_val" "annotation_blip/vqa_result.json" \
#   python BLIP_vqa.py --out_dir "${ROOT_DIR}/shape_val"

# ## 2D-spatial
# cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
# run_if_split_exists "spatial_val" "labels/annotation_obj_detection_2d/vqa_result.json" \
#   python 2D_spatial_eval.py --outpath "${ROOT_DIR}/spatial_val"

# ## numeracy
# cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
# run_if_split_exists "numeracy_val" "labels/annotation_num/vqa_result.json" \
#   python numeracy_eval.py --outpath "${ROOT_DIR}/numeracy_val"

## 3d spatial
cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
run_if_split_exists "3d_spatial_val" "labels/annotation_obj_detection_3d/vqa_result.json" \
  python 3D_spatial_eval.py --outpath "${ROOT_DIR}/3d_spatial_val"

# ## Non-Spatial Relationship
# cd "${T2I_COMP_CODE_ROOT}"
# run_if_split_exists "non_spatial_val" "annotation_clip/vqa_result.json" \
#   python CLIPScore_eval/CLIP_similarity.py --outpath "${ROOT_DIR}/non_spatial_val"

# ## 3-in-1 for Complex Compositions
# cd "${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
# run_if_split_exists "complex_val" "annotation_blip/vqa_result.json" \
#   python BLIP_vqa.py --out_dir "${ROOT_DIR}/complex_val"

# cd "${T2I_COMP_CODE_ROOT}/UniDet_eval"
# run_if_split_exists "complex_val" "labels/annotation_obj_detection_2d/vqa_result.json" \
#   python 2D_spatial_eval.py --outpath "${ROOT_DIR}/complex_val"

# cd "${T2I_COMP_CODE_ROOT}"
# run_if_split_exists "complex_val" "annotation_clip/vqa_result.json" \
#   python CLIPScore_eval/CLIP_similarity.py --outpath "${ROOT_DIR}/complex_val"

# cd "${T2I_COMP_CODE_ROOT}/3_in_1_eval/"
# run_if_split_exists "complex_val" "annotation_3_in_1/vqa_score.txt" \
#   python 3_in_1.py --outpath "${ROOT_DIR}/complex_val"
