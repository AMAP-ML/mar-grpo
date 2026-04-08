import os
import sys
# $PROJECT_ROOT$PROJECT_ROOT_/src/grpo/src/diffnext/example.py
sys.path.insert(0, "$PROJECT_ROOT$PROJECT_ROOT_/src/grpo/src")

import torch
import random
import time
import argparse
from PIL import Image
import torch.distributed as dist

import hpsv2
from tqdm import tqdm

from safetensors.torch import load_file
from diffnext.pipelines import NOVAPipeline


# ====== 你之前写的 ======
def reload_nova_transformer(pipeline: NOVAPipeline, ckpt_path: str) -> NOVAPipeline:
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.safetensors")

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    state = load_file(ckpt_path, device="cpu")
    transformer = pipeline.transformer
    target_sd = transformer.state_dict()

    load_sd = {}
    for k, v in state.items():
        candidate_keys = [k]
        for prefix in ("module.", "transformer.", "model."):
            if k.startswith(prefix):
                candidate_keys.append(k[len(prefix):])

        for cand in candidate_keys:
            if cand in target_sd:
                load_sd[cand] = v.to(dtype=target_sd[cand].dtype)
                break

    msg = transformer.load_state_dict(load_sd, strict=False)
    print(f"[reload] loaded: {len(load_sd)}")
    print(f"[reload] missing: {len(msg.missing_keys)}")
    print(f"[reload] unexpected: {len(msg.unexpected_keys)}")
    return pipeline


# ====== DDP ======
def setup(rank, world_size):
    # torchrun 模式下已经自动初始化了 NCCL，不需要重复初始化
    if dist.is_initialized():
        print(f"[rank {rank}] DDP already initialized by torchrun")
        torch.cuda.set_device(rank)
        return
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29507'
    print("before init")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("after init")
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


# ====== prompt 切分 ======
def get_prompt_data_for_rank(hps_prompts, rank, world_size, batch_size, save_root):
    all_items = []
    for prompt_type, prompt_list in hps_prompts.items():
        all_items.extend([
            {'caption': p, 'type': prompt_type, 'idx': i}
            for i, p in enumerate(prompt_list)
        ])

    # skip 已存在
    existed = set()
    if os.path.exists(save_root):
        for t in os.listdir(save_root):
            folder = os.path.join(save_root, t)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if name.endswith(".jpg"):
                    idx = int(name.replace(".jpg", ""))
                    existed.add((t, idx))

    all_items = [
        x for x in all_items
        if (x['type'], x['idx']) not in existed
    ]

    if rank == 0:
        print("total remaining:", len(all_items))

    # 切分
    total = len(all_items)
    per_rank = total // world_size

    start = rank * per_rank
    end = total if rank == world_size - 1 else start + per_rank

    rank_items = all_items[start:end]

    # batching
    batches = []
    for i in range(0, len(rank_items), batch_size):
        batch = rank_items[i:i + batch_size]
        batches.append({
            "caption": [x["caption"] for x in batch],
            "type": [x["type"] for x in batch],
            "idx": [x["idx"] for x in batch],
        })

    return batches


# ====== main ======
def main(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)

    torch.manual_seed(20)
    random.seed(20)

    print(f"[rank {rank}] loading model...")

    pipe = NOVAPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")

    if args.ckpt_path is not None:
        pipe = reload_nova_transformer(pipe, args.ckpt_path)

    pipe.set_progress_bar_config(disable=True)

    print(f"[rank {rank}] loading prompts...")

    all_prompts = hpsv2.benchmark_prompts('all')

    batches = get_prompt_data_for_rank(
        all_prompts,
        rank,
        world_size,
        args.batch_size,
        args.save_root
    )

    print(f"[rank {rank}] total batches:", len(batches))

    for item in batches:
        prompts = item["caption"]
        types = item["type"]
        idxs = item["idx"]

        # ====== 推理 ======
        images = pipe(
            prompts,
            guidance_scale=args.cfg,
            num_inference_steps=args.steps,
        ).images

        # ====== 保存 ======
        for i, img in enumerate(images):
            t = types[i]
            idx = idxs[i]

            save_dir = os.path.join(args.save_root, t)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
            img.save(save_path)

        print(f"[rank {rank}] saved batch")

    cleanup()


# ====== args ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="$PROJECT_ROOT/ckpts/BAAI/nova-d48w1024-sd512"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None  # or your ckpt
    )

    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)

    args = parser.parse_args()

    main(args)