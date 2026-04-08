# MAR-GRPO: Stabilizing Reinforcement Learning for Masked Autoregressive Image Generation

Official cleaned release for the paper **MAR-GRPO**.

This repository contains the training code used to apply Group Relative Policy Optimization (GRPO) to masked autoregressive image generation models, with experiments on **NOVA** and **Harmon**.

## Overview

Masked autoregressive image generation models couple an autoregressive backbone with a diffusion-style decoder head. While GRPO can improve these models, naive end-to-end RL introduces severe instability: the diffusion head changes the decoding mapping during training, and stochastic denoising trajectories add high-variance gradients to the autoregressive policy.

MAR-GRPO addresses this with three simple ideas:

- **Fix the diffusion head during RL**, which removes a major source of non-stationarity.
- **Estimate the policy with multiple diffusion trajectories**, which reduces variance caused by diffusion stochasticity.
- **Apply multi-trajectory estimation selectively on uncertain tokens**, improving stability without over-smoothing all updates.

Together, these changes make RL training for masked autoregressive image generation substantially more stable and more effective.

## What Is Included

This cleaned release keeps the main research code paths while removing private local paths and obvious workspace-specific clutter.

It includes:

- GRPO training entrypoints for **NOVA** and **Harmon**
- core trainer implementations used by the paper
- reward modules used in experiments, including HPS, GIT, GroundingDINO, OCR, Geneval, and ORM
- local third-party source trees that some rewards depend on
- example/debug launchers that preserve the original experiment style, but now use generic placeholders instead of personal paths

It does **not** include:

- model checkpoints
- reward checkpoints
- private datasets
- a fully frozen environment image

## Repository Structure

```text
.
├── README.md
├── Install.md
├── data/                        # data utilities and small local assets
├── jobs/                        # original job launch helpers
├── scripts/                     # original debug / experiment launchers
└── src/
    └── grpo/
        └── src/
            ├── open_r1/         # RL entrypoints and trainers
            ├── diffnext/        # NOVA-side generation code
            ├── harmon/          # Harmon-side generation code
            ├── infer/           # evaluation / inference helpers
            ├── nova/            # NOVA-side components
            └── utils/           # reward modules and local third-party deps
```

## Main Training Entry Points

The main training entrypoints used in this release are:

- `src/grpo/src/open_r1/grpo_v3.py`: MAR-GRPO training for **NOVA**
- `src/grpo/src/open_r1/grpo_v1_harmon.py`: MAR-GRPO training for **Harmon**

The corresponding core trainers are:

- `src/grpo/src/open_r1/trainer/grpo_trainer_nova_v3.py`
- `src/grpo/src/open_r1/trainer/grpo_trainer_harmon_v1_2.py`

## How To Read the Placeholder Paths

Many original experiment scripts in the research workspace contained hard-coded private paths. In this release, those paths have been replaced with **generic placeholders**.

The most common placeholders are:

- `$PROJECT_ROOT`: the root of this repository
- `$CHECKPOINT_ROOT`: your local directory containing base model checkpoints
- `$REWARD_MODEL_ROOT`: your local directory containing reward-model checkpoints
- `$WORKSPACE_ROOT`: your own experiment output / scratch directory
- `$ENV_ROOT`: your environment root, if you want to reference tools or interpreters there
- `$USER_ROOT`: a generic placeholder for a personal storage root
- `$GENEVAL_ROOT`, `$DPG_BENCH_ROOT`, `$T2I_COMPBENCH_ROOT`: roots for external evaluation toolkits if you use them
- `$LEGACY_PROJECT_ROOT`: a placeholder used in a few legacy helper scripts that were originally written against an older codebase layout

These are **documentation placeholders only**. Python will not expand `$PROJECT_ROOT` automatically unless you explicitly do so in your own shell or script.

If you want a script to work directly, replace these placeholders with either:

1. real absolute paths in your own environment, or
2. `os.environ[...]` / shell environment variables that you export before launch.

## Recommended Way To Configure Your Own Environment

The cleanest way to use this release is:

1. keep the repository path fixed as your own `PROJECT_ROOT`
2. keep all model checkpoints under one `CHECKPOINT_ROOT`
3. keep reward checkpoints under one `REWARD_MODEL_ROOT`
4. keep outputs under one `WORKSPACE_ROOT`
5. pass all runtime-specific paths through CLI flags or environment variables

For example, suppose your environment looks like this:

```text
/home/you/projects/mar_grpo_release
/home/you/checkpoints
/home/you/reward_models
/home/you/outputs
```

Then you can mentally map:

- `$PROJECT_ROOT -> /home/you/projects/mar_grpo_release`
- `$CHECKPOINT_ROOT -> /home/you/checkpoints`
- `$REWARD_MODEL_ROOT -> /home/you/reward_models`
- `$WORKSPACE_ROOT -> /home/you/outputs`

## Minimal Setup

See [Install.md](./Install.md) for dependency installation.

After installation:

```bash
cd /path/to/mar_grpo_release
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT/src/grpo/src:${PYTHONPATH}
```

This `PYTHONPATH` is important because the main code lives under `src/grpo/src`.

## Training Configuration Checklist

Before you launch training, decide the following:

- which base model you want to train: **NOVA** or **Harmon**
- where that base model checkpoint lives
- which dataset JSON / JSONL file you want to use
- which reward functions you want to enable
- where your output directory should be written
- which reward-model checkpoints are needed for those rewards

At minimum, most runs need:

- base model checkpoint path
- dataset path
- output directory
- HPS checkpoint path if using HPS reward
- CLIP checkpoint path if using HPS / Geneval style scoring

Depending on your reward set, you may also need:

- `GIT_CKPT_PATH`
- `GDINO_CKPT_PATH`
- `GDINO_CONFIG_PATH`
- `ORM_CKPT_PATH`
- `GENEVAL_MMDET_CONFIG`
- `GENEVAL_MMDET_CKPT_DIR`
- `GENEVAL_CLIP_CKPT`

## Placeholder-to-Variable Mapping

A practical way to adapt the cleaned scripts is to define a small set of shell variables first.

Example:

```bash
export PROJECT_ROOT=/path/to/mar_grpo_release
export CHECKPOINT_ROOT=/path/to/checkpoints
export REWARD_MODEL_ROOT=/path/to/reward_models
export WORKSPACE_ROOT=/path/to/outputs
export PYTHONPATH=$PROJECT_ROOT/src/grpo/src:${PYTHONPATH}
```

Then map the model-specific paths:

```bash
export MODEL_PATH=$CHECKPOINT_ROOT/BAAI/nova-d48w1024-sd512
export DATASET_PATH=$PROJECT_ROOT/data/your_train_metadata.json
export OUTPUT_DIR=$WORKSPACE_ROOT/nova_run_001

export HPS_CKPT_PATH=$REWARD_MODEL_ROOT/HPS_v2.1_compressed.pt
export CLIP_CKPT_PATH=$CHECKPOINT_ROOT/timm/vit_large_patch14_clip_224.openai/open_clip_pytorch_model.bin
export ORM_CKPT_PATH=$REWARD_MODEL_ROOT/your_orm_checkpoint
```

For Geneval reward:

```bash
export GENEVAL_MMDET_CONFIG=/path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
export GENEVAL_MMDET_CKPT_DIR=/path/to/mmdet_ckpts
export GENEVAL_CLIP_CKPT=$CLIP_CKPT_PATH
```

For GroundingDINO reward:

```bash
export GDINO_CONFIG_PATH=/path/to/GroundingDINO_config.py
export GDINO_CKPT_PATH=/path/to/groundingdino_swint_ogc.pth
```

For Harmon runs that need an extra transformer checkpoint:

```bash
export TRANSFORMER_PATH=/path/to/harmon_transformer_checkpoint
```

## Example: Running NOVA Training

A simple pattern is:

```bash
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT/src/grpo/src:${PYTHONPATH}

export MODEL_PATH=/path/to/nova_checkpoint
export DATASET_PATH=/path/to/train_metadata_flow_grpo.json
export OUTPUT_DIR=/path/to/output_nova
export HPS_CKPT_PATH=/path/to/HPS_v2.1_compressed.pt
export CLIP_CKPT_PATH=/path/to/open_clip_pytorch_model.bin
export ORM_CKPT_PATH=/path/to/orm_checkpoint

python src/grpo/src/open_r1/grpo_v3.py   --output_dir $OUTPUT_DIR   --model_name_or_path $MODEL_PATH   --dataset_name $DATASET_PATH   --reward_funcs hps orm   --hps_ckpt_path $HPS_CKPT_PATH   --clip_ckpt_path $CLIP_CKPT_PATH   --orm_ckpt_path $ORM_CKPT_PATH
```

If you prefer using the original experiment-style launcher scripts under `scripts/`, replace the placeholder strings in those scripts with your own paths or environment-variable lookups.

## Example: Running Harmon Training

```bash
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT/src/grpo/src:${PYTHONPATH}

export MODEL_PATH=/path/to/harmon_checkpoint
export TRANSFORMER_PATH=/path/to/harmon_transformer_checkpoint
export DATASET_PATH=/path/to/train_metadata_flow_grpo.json
export OUTPUT_DIR=/path/to/output_harmon
export HPS_CKPT_PATH=/path/to/HPS_v2.1_compressed.pt
export CLIP_CKPT_PATH=/path/to/open_clip_pytorch_model.bin

python src/grpo/src/open_r1/grpo_v1_harmon.py   --output_dir $OUTPUT_DIR   --model_name_or_path $MODEL_PATH   --transformer_path $TRANSFORMER_PATH   --dataset_name $DATASET_PATH   --reward_funcs hps   --hps_ckpt_path $HPS_CKPT_PATH   --clip_ckpt_path $CLIP_CKPT_PATH
```

## How To Adapt the Original Debug Scripts

The repository still contains many original `debug_entry_*.py` scripts because they document the experiment style used during research. These scripts are useful references, but they are **not meant to be plug-and-play** in a fresh environment.

To adapt one of them:

1. open the script you want to reuse
2. replace placeholder paths such as `$PROJECT_ROOT/...` or `$CHECKPOINT_ROOT/...`
3. remove any experiment-specific options you do not need
4. confirm that the reward checkpoints referenced in that script actually exist in your environment
5. run it from the repository root so relative paths remain valid

A good practice is to copy one script into your own launcher file and make your local edits there.

## Which Files Were Intentionally Kept

This release is conservative: it keeps the original trainer structure and many helper scripts intact, while only cleaning obvious clutter and personal path references.

That means you may still see:

- legacy experiment launchers
- helper evaluation scripts
- ablation-oriented code branches
- reward-specific utilities that are only needed in some settings

This is intentional. The goal of this release is to preserve the research code path while making it portable and safe to share.

## Common Pitfalls

### 1. `ModuleNotFoundError`

Usually means `PYTHONPATH` is missing or incorrect.

Use:

```bash
export PYTHONPATH=$PROJECT_ROOT/src/grpo/src:${PYTHONPATH}
```

### 2. A script still contains `$PROJECT_ROOT` literally

That means the script is still a template-like launcher. Replace the placeholder with your own path, or change it to read from `os.environ`.

### 3. Reward imports fail

Many reward modules need extra packages. See [Install.md](./Install.md).

The usual missing packages are:

- `hpsv2`
- `open_clip_torch`
- `mmdet`
- `mmcv`
- `clip_benchmark`
- `paddleocr`

### 4. Geneval reward fails at startup

Double-check:

- `GENEVAL_MMDET_CONFIG`
- `GENEVAL_MMDET_CKPT_DIR`
- `GENEVAL_CLIP_CKPT`

### 5. Checkpoint path errors

The cleaned release does not ship checkpoints. Most failures are simply caused by missing or mismatched local checkpoint paths.

## Notes on Privacy Cleanup

Private path roots from the original workspace have been replaced with generic placeholders. Hard-coded personal storage paths and the original hard-coded WandB API key were removed from this release.

If you see `WANDB_API_KEY` in some launcher scripts, it now refers only to the **environment variable name**, not to a bundled secret.

## Acknowledgements

This release builds on several excellent open-source projects, including the base model code and reward-related toolchains used in the paper experiments.

If you use this repository, please also consider citing the corresponding upstream projects such as **NOVA**, **Harmon**, **T2I-R1**, **GroundingDINO**, and **LLaVA-NeXT**.