
# ROOT_DIR="$WORKSPACE_ROOT/final_results/geneval_reward/t2i_compbench/t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_woaccu_beta0.03_bonusv2_2/iter1200"

ROOT_DIR="$USER_ROOT/workspace_t2i-r1/final_results/geneval_reward/t2i_compbench/t2i-r1-my-geneval-lr5e-6-test-datav3-kl_reweight2_woaccu_beta0.03_baseline_wo_entropy/iter1600"
T2I_COMP_CODE_ROOT="$T2I_COMPBENCH_ROOT"


## Attribute Binding:
export project_dir="${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
cd $project_dir
python BLIP_vqa.py --out_dir $ROOT_DIR/texture_val
python BLIP_vqa.py --out_dir $ROOT_DIR/color_val
python BLIP_vqa.py --out_dir $ROOT_DIR/shape_val

## 2D-spatial
export project_dir="${T2I_COMP_CODE_ROOT}/UniDet_eval"
cd $project_dir
python 2D_spatial_eval.py --outpath $ROOT_DIR/spatial_val


# numeracy
export project_dir="${T2I_COMP_CODE_ROOT}/UniDet_eval"
cd $project_dir
python numeracy_eval.py --outpath $ROOT_DIR/numeracy_val

## 3d spatial
export project_dir="${T2I_COMP_CODE_ROOT}/UniDet_eval"
cd $project_dir
python 3D_spatial_eval.py --outpath $ROOT_DIR/3d_spatial_val


##  Non-Spatial Relationship
export project_dir="${T2I_COMP_CODE_ROOT}"
cd $project_dir
python CLIPScore_eval/CLIP_similarity.py --outpath $ROOT_DIR/non_spatial_val

# 3-in-1 for Complex Compositions
export project_dir="${T2I_COMP_CODE_ROOT}/BLIPvqa_eval/"
cd $project_dir
python BLIP_vqa.py --out_dir $ROOT_DIR/complex_val
export project_dir="${T2I_COMP_CODE_ROOT}/UniDet_eval"
cd $project_dir
python 2D_spatial_eval.py --outpath $ROOT_DIR/complex_val
export project_dir="${T2I_COMP_CODE_ROOT}/3_in_1_eval/"
cd $project_dir
python "3_in_1.py" --outpath $ROOT_DIR/complex_val