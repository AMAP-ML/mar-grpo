# harmon_infer_hps.py
import os
import sys
sys.path.insert(0,'$PROJECT_ROOT/src/grpo/src/')

import argparse
import random
from collections import OrderedDict
from safetensors.torch import load_file

import torch
import torch.distributed as dist

from PIL import Image

import hpsv2

# =========================
# 1. 分布式初始化
# =========================

def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# =========================
# 2. HPS prompt 分配（和 NOVA 对齐）
# =========================

def get_prompt_data_for_rank(hps_prompts, rank, world_size, batch_size, save_root):
    all_items = []
    for prompt_type, prompt_list in hps_prompts.items():
        all_items.extend([
            {"caption": p, "type": prompt_type, "idx": i}
            for i, p in enumerate(prompt_list)
        ])

    # 去掉已经生成过的
    existed_item = set()
    if os.path.exists(save_root):
        for existed_type in os.listdir(save_root):
            subdir = os.path.join(save_root, existed_type)
            if not os.path.isdir(subdir):
                continue
            for name in os.listdir(subdir):
                if name.endswith(".jpg"):
                    idx = int(name.replace(".jpg", ""))
                    existed_item.add((existed_type, idx))

    all_items = [
        item for item in all_items
        if (item["type"], item["idx"]) not in existed_item
    ]

    total_samples = len(all_items)
    if rank == 0:
        print(f"[HPS] Total samples after filtering: {total_samples}")

    samples_per_rank = total_samples // world_size
    start = rank * samples_per_rank
    end = total_samples if rank == world_size - 1 else start + samples_per_rank
    rank_items = all_items[start:end]

    batched = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i + batch_size]
        batched.append({
            "caption": [x["caption"] for x in batch],
            "type": [x["type"] for x in batch],
            "idx": [x["idx"] for x in batch],
        })

    return batched

# =========================
# 3. 主函数
# =========================

def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)

    # 固定随机性（重要）
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)

    device = torch.device(f"cuda:{rank}")

    # =========================
    # 4. 加载 Harmon
    # =========================

    sys.path.insert(0, args.harmon_code_root)

    from harmon.modeling_harmon import HarmonModel
    from harmon.mar import MAR
    from harmon.harmon_generate_image_with_logp import generate_image_with_logp
    from harmon.text2image_hf import GENERATION_TEMPLATE
    from transformers import AutoTokenizer

    process_class = (
        HarmonModel
        .from_pretrained(args.model_path)
        .to(device)
        .bfloat16()
        .eval()
    )

    model: MAR = process_class.mar

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True
    )


    def reload_mar_simple(process_class, ckpt_root, device):
        """
        最简单版本：
        - ckpt_root/model.safetensors
        - 只加载 process_class.mar
        """

        ckpt_path = os.path.join(ckpt_root, "model.safetensors")
        assert os.path.exists(ckpt_path), f"Not found: {ckpt_path}"

        print(f"[reload_mar] loading {ckpt_path}")

        # 1) safetensors 先 load 到 CPU
        state = load_file(ckpt_path, device="cpu")

        mar = process_class.mar
        mar_sd = mar.state_dict()

        # 2) 只取 MAR 相关权重，顺便去前缀
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

        # 3) dtype 对齐
        target_dtype = next(mar.parameters()).dtype
        new_sd = {k: t.to(dtype=target_dtype, device="cpu") for k, t in new_sd.items()}

        # 4) load
        missing, unexpected = mar.load_state_dict(new_sd, strict=False)
        print(f"[reload_mar] loaded={len(new_sd)} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:10])
        if unexpected:
            print("  unexpected:", unexpected[:10])

        # 5) 放回 GPU
        mar.to(device)
        process_class.to(device)

        return process_class


    if args.transformer_path is not None:
        process_class=reload_mar_simple(process_class,args.transformer_path,device)

    neg_prompt = args.neg_prompt

    # =========================
    # 5. HPS prompts
    # =========================

    all_prompts = hpsv2.benchmark_prompts("all")
    batched_data = get_prompt_data_for_rank(
        all_prompts,
        rank,
        world_size,
        args.batch_size,
        args.save_root,
    )

    os.makedirs(args.save_root, exist_ok=True)

    # =========================
    # 6. 推理 + 保存
    # =========================

    for item in batched_data:
        text_prompts = item["caption"]
        types = item["type"]
        indices = item["idx"]

        if rank == 0:
            print(f"[Rank {rank}] Processing batch size = {len(text_prompts)}")

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

        for img, style, idx in zip(images, types, indices):
            save_dir = os.path.join(args.save_root, style)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
            img.save(save_path)

    cleanup()

# =========================
# 7. CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="$CHECKPOINT_ROOT/harmon/Harmon-1_5B",
        help="Path to Harmon model checkpoint",
    )
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument(
        "--harmon_code_root",
        type=str,
        default="$PROJECT_ROOT/src/grpo/src/",
        help="Root dir containing harmon/ (the repo you sys.path.insert)",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./hps_harmon_results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="Generate an image.",
    )

    args = parser.parse_args()
    main(args)
