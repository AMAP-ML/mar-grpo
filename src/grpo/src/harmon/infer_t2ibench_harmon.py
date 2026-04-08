# infer_compbench_harmon.py
import os
import sys
sys.path.insert(0,'$PROJECT_ROOT/src/grpo/src/')

import time
import argparse
import random
import re
from typing import List, Dict

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import AutoTokenizer

# -------------------------
# DDP utils
# -------------------------

def setup(rank: int, world_size: int, master_port: str = "29502"):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_prompt_data_for_rank(prompt_file: str, rank: int, world_size: int, batch_size: int):
    """Read prompts from txt, shard by rank, then batch."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    all_items = [{"caption": p, "idx": i} for i, p in enumerate(prompt_list)]

    total = len(all_items)
    if rank == 0:
        print(f"[CompBench] total prompts = {total} from {prompt_file}")

    # shard
    per = total // world_size
    start = rank * per
    end = total if rank == world_size - 1 else start + per
    rank_items = all_items[start:end]

    # batch
    batched = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i + batch_size]
        batched.append({
            "caption": [x["caption"] for x in batch],
            "idx": [x["idx"] for x in batch],
        })
    return batched

def sanitize_filename(s: str, max_len: int = 120) -> str:
    """Make prompt safe as filename."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", s)
    return s[:max_len]

# -------------------------
# MAR hot-reload (RL ckpt)
# -------------------------

def reload_mar_simple(process_class, ckpt_root: str, device: torch.device):
    """
    Simple reload:
    - ckpt_root/model.safetensors
    - only load process_class.mar
    """
    ckpt_path = os.path.join(ckpt_root, "model.safetensors")
    assert os.path.exists(ckpt_path), f"Not found: {ckpt_path}"
    print(f"[reload_mar] loading {ckpt_path}")

    state = load_file(ckpt_path, device="cpu")

    mar = process_class.mar
    mar_sd = mar.state_dict()

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

    target_dtype = next(mar.parameters()).dtype
    new_sd = {k: t.to(dtype=target_dtype, device="cpu") for k, t in new_sd.items()}

    missing, unexpected = mar.load_state_dict(new_sd, strict=False)
    print(f"[reload_mar] loaded={len(new_sd)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    mar.to(device)
    process_class.to(device)
    return process_class

# -------------------------
# Main
# -------------------------

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size, master_port=args.master_port)

    # seeds
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = torch.device(f"cuda:{rank}")

    # Import harmon from your code root
    sys.path.insert(0, args.harmon_code_root)
    from harmon.modeling_harmon import HarmonModel
    from harmon.mar import MAR
    from harmon.harmon_generate_image_with_logp import generate_image_with_logp
    from harmon.text2image_hf import GENERATION_TEMPLATE

    # Load model
    process_class = (
        HarmonModel
        .from_pretrained(args.model_path)
        .to(device)
        .bfloat16()
        .eval()
    )

    if args.transformer_path is not None:
        process_class = reload_mar_simple(process_class, args.transformer_path, device)

    model: MAR = process_class.mar

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    neg_prompt = args.neg_prompt

    # CompBench prompts
    prompts_dir = args.prompts_dir

    for bench_type in args.bench_types:
        bench_root = os.path.join(args.save_root, bench_type)
        output_path = os.path.join(bench_root, "samples")
        os.makedirs(output_path, exist_ok=True)

        prompt_file = os.path.join(prompts_dir, f"{bench_type}.txt")
        batched_data = get_prompt_data_for_rank(prompt_file, rank, world_size, args.batch_size)

        for item in batched_data:
            t0 = time.time()

            captions: List[str] = item["caption"]          # batch_size prompts
            idxs: List[int] = item["idx"]

            # 对 batch 内每个 prompt，生成 repeat_size 张
            for caption, pidx in zip(captions, idxs):
                safe_prompt = sanitize_filename(caption)
                first_img_path = os.path.join(output_path, f"{safe_prompt}_{0:05d}.png")
                if os.path.exists(first_img_path):
                    if rank == 0:
                        print(f"[rank {rank}] existed: {first_img_path}")
                    continue

                # Harmon 的接口吃 list[str]，我们构造 repeat_size 条相同 prompt
                image_gen_prompt_list = [
                    GENERATION_TEMPLATE.format(text=caption)
                    for _ in range(args.repeat_size)
                ]

                if rank == 0:
                    print(f"[rank {rank}] {bench_type} | prompt_idx={pidx} | repeat={args.repeat_size}")

                with torch.no_grad(), torch.inference_mode():
                    images, *_ = generate_image_with_logp(
                        process_class,
                        model,
                        image_gen_prompt_list,
                        neg_prompt,
                        tokenizer,
                        selected_mask_steps=None,
                    )

                for ridx, img in enumerate(images):
                    img_path = os.path.join(output_path, f"{safe_prompt}_{ridx:05d}.png")
                    img.save(img_path)

                if rank == 0:
                    print(f"[rank {rank}] saved {args.repeat_size} imgs, cost {time.time()-t0:.2f}s")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Harmon ckpt
    parser.add_argument("--model_path", type=str,
                        default="$CHECKPOINT_ROOT/harmon/Harmon-1_5B")
    parser.add_argument("--harmon_code_root", type=str,
                        default="$PROJECT_ROOT/src/grpo/src/",
                        help="dir that contains harmon/")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="RL ckpt root; should contain model.safetensors (MAR only)")

    # CompBench
    parser.add_argument("--prompts_dir", type=str,
                        default="$PROJECT_ROOT/reference_code/STAGE/data/T2I-CompBench/dataset_val")
    parser.add_argument("--bench_types", nargs="+",
                        default=[
                            "3d_spatial_val","color_val","complex_val","new_objects",
                            "non_spatial_val","numeracy_val","shape_val","spatial_val","texture_val"
                        ])

    # Save
    parser.add_argument("--save_root", type=str,
                        default="$WORKSPACE_ROOT/harmon_origin")

    # Inference cfg
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--repeat_size", type=int, default=10)
    parser.add_argument("--neg_prompt", type=str, default="Generate an image.")
    parser.add_argument("--master_port", type=str, default="29502")

    args = parser.parse_args()
    main(args)
