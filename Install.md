# Install

This document describes a practical setup for running the cleaned MAR-GRPO release.

## 1. Environment

We recommend using a fresh Python environment with CUDA-enabled PyTorch.

Example:

```bash
conda create -n mar_grpo python=3.10 -y
conda activate mar_grpo
```

Install the core stack first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets trl deepspeed sentencepiece protobuf
pip install einops safetensors pillow scipy numpy pandas matplotlib tqdm
```

You may need to adjust the PyTorch CUDA wheel according to your cluster setup.

## 2. Local Source Layout

After cloning or copying the cleaned release:

```bash
cd /path/to/diff_grpo_release_20260406
export PYTHONPATH=$PWD/src:${PYTHONPATH}
```

The main code paths are:

- `src/open_r1`
- `src/diffnext`
- `src/harmon`
- `src/nextstep`
- `src/utils`

## 3. Core External Dependencies

The repository includes the paper code, but several runtime dependencies are still external.

At minimum, prepare:

- PyTorch + CUDA
- Transformers / Accelerate / TRL / DeepSpeed
- The checkpoint dependencies for **NOVA** or **Harmon**
- Reward-model checkpoints used in your experiments

## 4. Reward Dependencies

Different reward functions require different external packages.

### HPS / CLIP reward

Required for:

- `reward_hps.py`
- `reward_clip.py`

Dependency:

```bash
pip install hpsv2 open_clip_torch
```

You also need:

- `HPS_CKPT_PATH`
- `CLIP_CKPT_PATH`

### GIT reward

Required for:

- `reward_git.py`

Dependency is usually covered by `transformers`, but you still need the corresponding checkpoint:

- `GIT_CKPT_PATH`

### GroundingDINO reward

Required for:

- `reward_gdino.py`

The repository vendors a local GroundingDINO copy under:

- `src/utils/GroundingDINO`

You may still need to install its dependencies:

```bash
pip install -r src/utils/GroundingDINO/requirements.txt
pip install -e src/utils/GroundingDINO
```

You also need:

- `GDINO_CKPT_PATH`
- `GDINO_CONFIG_PATH`

### OCR reward

Required for:

- `reward_ocr.py`

Install:

```bash
pip install paddleocr python-Levenshtein
```

### ORM reward

Required for:

- `reward_orm.py`

The repository includes a local `LLaVA-NeXT` copy under:

- `src/utils/LLaVA-NeXT`

Install its dependencies according to your environment, typically:

```bash
pip install -r src/utils/LLaVA-NeXT/requirements.txt
pip install -e src/utils/LLaVA-NeXT
```

You also need:

- `ORM_CKPT_PATH`

### Geneval reward

Required for:

- `reward_geneval.py`

Install:

```bash
pip install open_clip_torch mmengine
pip install mmdet mmcv clip_benchmark
```

You must also set:

```bash
export GENEVAL_MMDET_CONFIG=/path/to/mask2former_config.py
export GENEVAL_MMDET_CKPT_DIR=/path/to/mmdet_ckpts
export GENEVAL_CLIP_CKPT=/path/to/open_clip_pytorch_model.bin
```

Optionally:

```bash
export GENEVAL_OBJECT_NAMES_PATH=/path/to/object_names.txt
```

If not set, `reward_geneval.py` will default to the local file under:

- `src/utils/reward-server/reward_server/object_names.txt`

## 5. Checkpoints

This release does not include model checkpoints. Prepare paths for:

- base generator checkpoint:
  - `MODEL_PATH`
- reward checkpoints:
  - `HPS_CKPT_PATH`
  - `CLIP_CKPT_PATH`
  - `GIT_CKPT_PATH`
  - `GDINO_CKPT_PATH`
  - `ORM_CKPT_PATH`

For Harmon experiments with latent-sim based selection, also prepare:

- `TRANSFORMER_PATH`

## 6. Running Training

Example for NOVA:

```bash
export MODEL_PATH=/path/to/nova
export DATASET_PATH=/path/to/train_metadata_flow_grpo.json
export OUTPUT_DIR=/path/to/output_nova
export HPS_CKPT_PATH=/path/to/HPS_v2.1_compressed.pt
export CLIP_CKPT_PATH=/path/to/open_clip_pytorch_model.bin

bash examples/train_nova.sh
```

Example for Harmon:

```bash
export MODEL_PATH=/path/to/harmon
export DATASET_PATH=/path/to/train_metadata_flow_grpo.json
export OUTPUT_DIR=/path/to/output_harmon
export HPS_CKPT_PATH=/path/to/HPS_v2.1_compressed.pt
export CLIP_CKPT_PATH=/path/to/open_clip_pytorch_model.bin
export TRANSFORMER_PATH=/path/to/checkpoint-400

bash examples/train_harmon.sh
```

## 7. Common Issues

### `ModuleNotFoundError`

Usually means `PYTHONPATH` is missing:

```bash
export PYTHONPATH=$PWD/src:${PYTHONPATH}
```

### Reward import fails

Check whether the reward-specific packages are installed. The main ones that often fail are:

- `hpsv2`
- `open_clip_torch`
- `paddleocr`
- `mmcv`
- `mmdet`
- `clip_benchmark`

### Geneval reward path error

Make sure these are set:

- `GENEVAL_MMDET_CONFIG`
- `GENEVAL_MMDET_CKPT_DIR`
- `GENEVAL_CLIP_CKPT`

### Checkpoint path error

Most paths in this release are passed from CLI flags or environment variables. Double-check your exported paths before launch.
