# harmon_infer_drawbench.py
# Run Harmon inference on DrawBench CSV with DDP sharding (idx % world_size == rank),
# resume support, optional MAR reload from safetensors.
#
# Usage (single node, 8 gpus):
#   torchrun --nproc_per_node=8 harmon_infer_drawbench.py \
#     --save_root ./drawbench_harmon_results \
#     --drawbench_csv_path /path/to/drawbench_data.csv \
#     --model_path /path/to/Harmon-1_5B \
#     --harmon_code_root /path/to/harmon/repo/root \
#     --batch_size 8 --seed 20
#
# Optional:
#   --transformer_path /path/to/ckpt_root_containing_model.safetensors
#   --csv_prompt_key Prompts   (or Prompt)

import os
import sys
sys.path.insert(0,'$PROJECT_ROOT/src/grpo/src/')
import csv
import argparse
import random
from safetensors.torch import load_file

import torch
import torch.distributed as dist

import hpsv2  # not used, but kept if your env expects it

# =========================
# 1) DDP init
# =========================

def setup(rank, world_size, master_addr="127.0.0.1", master_port="29501"):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# =========================
# 2) DrawBench CSV reader
# =========================

def read_drawbench_csv(csv_path, prompt_key_candidates=("Prompts", "Prompt")):
    """
    Returns: list[{"idx": int, "caption": str}]
    idx = row number in csv (starting from 0)
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            caption = ""
            for k in prompt_key_candidates:
                if r.get(k) is not None:
                    caption = (r.get(k) or "").strip()
                    if caption:
                        break
            if caption:
                rows.append({"idx": i, "caption": caption})
    return rows

def shard_and_batch_items(rows, rank, world_size, batch_size, save_root):
    """
    - Resume: skip if save_root/<idx:05d>.jpg exists
    - Shard: idx % world_size == rank
    - Batch into dict: {"caption": [...], "idx": [...]}
    """
    os.makedirs(save_root, exist_ok=True)

    remaining = [
        it for it in rows
        if not os.path.exists(os.path.join(save_root, f"{it['idx']:05d}.jpg"))
    ]

    shard = [it for it in remaining if (it["idx"] % world_size) == rank]

    batched = []
    for s in range(0, len(shard), batch_size):
        batch = shard[s:s + batch_size]
        batched.append({
            "caption": [x["caption"] for x in batch],
            "idx": [x["idx"] for x in batch],
        })
    return remaining, shard, batched

# =========================
# 3) Optional ckpt reload (MAR only)
# =========================

def reload_mar_simple(process_class, ckpt_root, device):
    """
    Minimal version:
    - ckpt_root/model.safetensors
    - only loads process_class.mar
    """
    ckpt_path = os.path.join(ckpt_root, "model.safetensors")
    assert os.path.exists(ckpt_path), f"Not found: {ckpt_path}"

    print(f"[reload_mar] loading {ckpt_path}")

    # 1) load safetensors on CPU
    state = load_file(ckpt_path, device="cpu")

    mar = process_class.mar
    mar_sd = mar.state_dict()

    # 2) extract MAR weights, strip prefixes
    new_sd = {}
    for k, v in state.items():
        if k.startswith("mar."):
            new_k = k[len("mar."):]
        elif k.startswith("module.mar."):
            new_k = k[len("module.mar."):]
        elif k in mar_sd:
            new_k = k
        else:
            continue

        if new_k in mar_sd:
            new_sd[new_k] = v

    # 3) dtype align
    target_dtype = next(mar.parameters()).dtype
    new_sd = {k: t.to(dtype=target_dtype, device="cpu") for k, t in new_sd.items()}

    # 4) load
    missing, unexpected = mar.load_state_dict(new_sd, strict=False)
    print(f"[reload_mar] loaded={len(new_sd)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    # 5) move back to GPU
    mar.to(device)
    process_class.to(device)
    return process_class

# =========================
# 4) Main
# =========================

def main(args):
    # Env / rank
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    setup(rank, world_size, master_port=args.master_port)

    # Determinism
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = torch.device(f"cuda:{rank}")

    # Harmon repo visibility
    sys.path.insert(0, args.harmon_code_root)

    from harmon.modeling_harmon import HarmonModel
    from harmon.mar import MAR  # noqa: F401
    from harmon.harmon_generate_image_with_logp import generate_image_with_logp
    from harmon.text2image_hf import GENERATION_TEMPLATE
    from transformers import AutoTokenizer

    # Load Harmon
    process_class = (
        HarmonModel
        .from_pretrained(args.model_path)
        .to(device)
        .bfloat16()
        .eval()
    )

    # Access MAR
    model = process_class.mar

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True
    )

    # Optional reload MAR weights
    if args.transformer_path is not None:
        process_class = reload_mar_simple(process_class, args.transformer_path, device)
        model = process_class.mar

    # Read DrawBench
    prompt_keys = tuple([k.strip() for k in args.csv_prompt_keys.split(",") if k.strip()])
    rows = read_drawbench_csv(args.drawbench_csv_path, prompt_key_candidates=prompt_keys)

    remaining, shard, batched_data = shard_and_batch_items(
        rows=rows,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        save_root=args.save_root
    )

    if rank == 0:
        print(f"[DrawBench] total rows: {len(rows)}")
        print(f"[DrawBench] remaining after resume filter: {len(remaining)}")
    print(f"[rank {rank}] to-generate: {len(shard)} / {len(remaining)}")

    os.makedirs(args.save_root, exist_ok=True)
    neg_prompt = args.neg_prompt

    # Inference
    for bi, item in enumerate(batched_data):
        text_prompts = item["caption"]
        indices = item["idx"]

        if rank == 0:
            print(f"[rank {rank}] batch {bi+1}/{len(batched_data)} size={len(text_prompts)}")

        image_gen_prompt_list = [
            GENERATION_TEMPLATE.format(text=p)
            for p in text_prompts
        ]

        with torch.no_grad(), torch.inference_mode():
            images, *_ = generate_image_with_logp(
                process_class,
                model,
                image_gen_prompt_list,
                neg_prompt,
                tokenizer,
                selected_mask_steps=None,
            )

        for img, idx in zip(images, indices):
            out_path = os.path.join(args.save_root, f"{idx:05d}.jpg")
            try:
                img.save(out_path)
            except Exception as e:
                print(f"[rank {rank}] save failed idx={idx}: {e}")

    cleanup()

# =========================
# 5) CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="$CHECKPOINT_ROOT/harmon/Harmon-1_5B",
        help="Path to Harmon model checkpoint",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Root dir containing model.safetensors (optional). Only loads mar.* weights.",
    )
    parser.add_argument(
        "--harmon_code_root",
        type=str,
        default="$PROJECT_ROOT/src/grpo/src/",
        help="Root dir containing harmon/ (repo root inserted to sys.path)",
    )

    parser.add_argument("--drawbench_csv_path", type=str, default="$PROJECT_ROOT/data/drawbench_data.csv")
    parser.add_argument(
        "--csv_prompt_keys",
        type=str,
        default="Prompts,Prompt",
        help="CSV column candidates, comma-separated. e.g. 'Prompts,Prompt'",
    )

    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="Output dir, saves <idx:05d>.jpg",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--neg_prompt", type=str, default="Generate an image.")
    parser.add_argument("--master_port", type=str, default="29501")

    args = parser.parse_args()
    main(args)
