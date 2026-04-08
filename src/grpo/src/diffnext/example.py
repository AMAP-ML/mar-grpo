import os
import sys
import math
import torch
from PIL import Image
from safetensors.torch import load_file

# ====== 你的路径（按需修改）======
sys.path.insert(0, "$PROJECT_ROOT$PROJECT_ROOT_/src/grpo/src")

from diffnext.pipelines import NOVAPipeline

model_id = "$PROJECT_ROOT/ckpts/BAAI/nova-d48w1024-sd512"
transformer_ckpt = "$PROJECT_ROOT/my_personal/nova-test-beta0.01_critic4_1_test2_iter500/model.safetensors"
# transformer_ckpt = "$PROJECT_ROOT/amap_upload/nova/test-beta0.01/checkpoint-400/model.safetensors"

# save_root = "out_diversity"   # 输出目录
save_root = "out_diversity_ours"   # 输出目录

# ====== prompts（可以换成你上面那一批 diversity prompts）======
prompts = [
    # "A futuristic city at night",
    # "A surreal landscape",
    # "A colorful abstract painting",

    # "A red apple on a table",
    # "A cup of coffee on a wooden desk",
    # "A car parked on the street",
    # "A house in the countryside",
    # "A bowl of fruit"

    # "A blue cube on top of a red sphere",
    # "A small dog sitting under a large tree",
    # "A yellow car next to a green building",
    # "A person holding a red umbrella in the rain",
    # "A cat sleeping on a pile of books"
    
    # "A person sitting on a chair with a cat under the chair",
    # "A book on a table with a cup on top of the book",
    # "A laptop on a desk with a keyboard in front and a mouse to the right",
    # "A person standing in front of a car with a tree behind them"

    # "A blue cube on top of a red sphere, next to a green cylinder",
    # "A yellow sphere under a blue cube and to the left of a red cylinder",
    # "A green cube stacked on a blue cube, with a red sphere beside them",
    # "A red sphere inside a transparent cube, placed on a yellow cylinder",
    # "A blue cube on top of a red sphere, which is on top of a green cube"

    # "A portrait in oil painting style",
    # "A landscape in watercolor style",
    # "A cyberpunk style city",
    # "A cartoon style character",
    # "A realistic photograph of a mountain"

    # "A room with a sofa, table, and lamp",
    # "A street with cars and pedestrians",
    # "A kitchen with various utensils",
    # "A park with trees and benches",
    # "A beach with people and umbrellas"

    # "A portrait of a young woman, centered, looking at the camera, photorealistic, natural lighting",
    # "A portrait of a young woman, centered, looking at the camera, oil painting style",
    # "A portrait of a young woman, centered, looking at the camera, pencil sketch",
    # "A portrait of a young woman, centered, looking at the camera, watercolor painting",
    # "A portrait of a young woman, centered, looking at the camera, anime style",
    # "A portrait of a young woman, centered, looking at the camera, cinematic lighting, high contrast"

    "A street scene in a busy city",
    "A living room with modern furniture",
    "A market with people and various goods",
    "A park with trees and people walking",
    "A beach with people and umbrellas",
    "A kitchen with cooking utensils and food",
    "A desk with various objects on it",
    "A group of people in a public place"
]

num_samples_per_prompt = 10   # 每个prompt生成多少张
seed_base = 1000              # 随机种子起点


# ====== reload transformer ======
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


# ====== 拼图函数 ======
def make_grid(images, nrow=None, padding=4):
    N = len(images)

    if nrow is None:
        nrow = int(math.ceil(math.sqrt(N)))
    ncol = int(math.ceil(N / nrow))

    w, h = images[0].size

    grid = Image.new(
        "RGB",
        (nrow * w + (nrow - 1) * padding,
         ncol * h + (ncol - 1) * padding),
        (255, 255, 255)
    )

    for i, img in enumerate(images):
        row = i // nrow
        col = i % nrow
        grid.paste(img, (col * (w + padding), row * (h + padding)))

    return grid


# ====== main ======
def main():
    os.makedirs(save_root, exist_ok=True)

    print("Loading NOVA...")
    pipe = NOVAPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")

    print("Reloading transformer...")
    pipe = reload_nova_transformer(pipe, transformer_ckpt)

    pipe.set_progress_bar_config(disable=True)

    print("Start inference...")

    for pid, prompt in enumerate(prompts):
        print(f"\nPrompt {pid}: {prompt}")

        prompt_dir = os.path.join(save_root, f"prompt_{pid}")
        os.makedirs(prompt_dir, exist_ok=True)

        # ====== 一次性 batch 生成 ======
        generators = [
            torch.Generator(device="cuda").manual_seed(seed_base + i)
            for i in range(num_samples_per_prompt)
        ]

        images = pipe(
            [prompt] * num_samples_per_prompt,
            guidance_scale=5.0,
            num_inference_steps=25,
        ).images

        # ====== 保存单张 ======
        all_images = []
        for i, img in enumerate(images):
            save_path = os.path.join(prompt_dir, f"{i:02d}.jpg")
            img.save(save_path)
            all_images.append(img)

        # ====== 拼 grid ======
        grid = make_grid(all_images)
        grid_path = os.path.join(prompt_dir, "grid.jpg")
        grid.save(grid_path)

        print(f"Saved {num_samples_per_prompt} images + grid to {prompt_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()