import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import os
# # if int(os.environ.get("LOCAL_RANK", 0)) == 0:
# 
import json
import time
import torch
import torch.distributed as dist
from PIL import Image
from statistics import pstdev

from utils.reward_geneval import Geneval_score

# ------------------------
# utils
# ------------------------
def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def shard_list(data, rank, world_size):
    return data[rank::world_size]

# ------------------------
# main
# ------------------------
def main():
    rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    root_dir = os.environ.get("GENEVAL_BATCH_ROOT_DIR", "./geneval_inputs")
    out_dir = os.environ.get("GENEVAL_BATCH_OUT_DIR", "./geneval_outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_jsonl = os.path.join(out_dir, f"filtered_rank{rank}.jsonl")

    # 所有 prompt 目录
    prompt_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    # DDP shard
    prompt_dirs = shard_list(prompt_dirs, rank, world_size)

    # Geneval
    geneval = Geneval_score(None)
    geneval.load_to_device(device)

    batch_size = 8   # prompt 级 batch（不是 image）

    buffer = []
    start_time = time.time()

    for idx, prompt_id in enumerate(prompt_dirs):
        prompt_path = os.path.join(root_dir, prompt_id)
        sample_dir = os.path.join(prompt_path, "samples")
        meta_path = os.path.join(prompt_path, "metadata.jsonl")

        if not os.path.exists(sample_dir) or not os.path.exists(meta_path):
            continue

        # ---- load metadata
        with open(meta_path, "r") as f:
            meta_data = [json.loads(line) for line in f]

        prompt = meta_data[0]["prompt"]

        # ---- load images
        images = []
        for img_name in sorted(os.listdir(sample_dir)):
            if img_name.endswith(".png"):
                images.append(Image.open(os.path.join(sample_dir, img_name)))

        if len(images) == 0:
            continue

        buffer.append((prompt_id, prompt, images, meta_data))

        # ------------------------
        # batch eval
        # ------------------------
        if len(buffer) == batch_size or idx == len(prompt_dirs) - 1:
            all_images = []
            all_prompts = []
            all_meta = []
            split_sizes = []

            for (_, p, imgs, m) in buffer:
                all_images.extend(imgs)
                all_prompts.extend([p] * len(imgs))
                all_meta.extend(m * len(imgs))
                split_sizes.append(len(imgs))

            with torch.inference_mode():
                scores,rewards_,strict_rewards,grouped_rewards,grouped_strict_rewards = geneval(all_images, all_prompts, all_meta)

            offset = 0
            for (prompt_id, p, imgs, m), n_img in zip(buffer, split_sizes):
                r = scores[offset: offset + n_img]
                offset += n_img

                std = pstdev(r)

                if std != 0:
                    # record = {
                    #     "id": prompt_id,
                    #     "prompt": p,
                    #     "reward_std": std,
                    #     "metadata": m[0]
                    # }
                    record = m[0].copy()          # ⚠️ 建议 copy，避免原 metadata 被污染
                    record["rewards"] = r         # 加上 reward 列表（scores）

                    with open(out_jsonl, "a") as f:
                        f.write(json.dumps(record) + "\n")
                        f.flush()

            buffer = []

    if rank == 0:
        print(f"[Done] cost {time.time() - start_time:.2f}s")

    cleanup_ddp()

# ------------------------
if __name__ == "__main__":
    main()
