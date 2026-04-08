import os
import torch
import numpy as np
from PIL import Image

save_root = "$PROJECT_ROOT/src/grpo/src/harmon/sample_seeds"
os.makedirs(save_root, exist_ok=True)

# pred_x0: [N, 1024, 16]
x=sample_all_diff_latents_seed["diff_xt_prev"][0][0,:1024]

T, C = x.shape
assert T == 32 * 32 and C == 16

# ---- 固定投影矩阵 W: [16, 3]（seed 固定，可复现）----
g = torch.Generator(device=x.device).manual_seed(0)
W = torch.randn(16, 3, generator=g, device=x.device) / (16 ** 0.5)
W.requires_grad_(False)


for idx in range(len(sample_all_diff_latents_seed["diff_xt"])):
    x = sample_all_diff_latents_seed["diff_xt"][idx][-1,:1024].detach().to(torch.float32)
    # ---- 全局分位数归一化范围（跨图/跨step可比，且不容易“忽明忽暗”）----
    x3_all = (x.reshape(-1, 16) @ W)               # [N*1024, 3]
    lo = torch.quantile(x3_all, 0.01, dim=0)       # 1% 分位
    hi = torch.quantile(x3_all, 0.99, dim=0)       # 99% 分位
    den = (hi - lo).clamp_min(1e-6)

    x3 = x @ W                            # [1024, 3]
    img = x3.reshape(32, 32, 3)

    # 可选：tanh 让颜色更柔和（不容易刺眼/发花）
    img = torch.tanh(img / 3.0)                # 3.0 可调：越大越“平”
    img = (img + 1) * 0.5                      # [-1,1] -> [0,1]

    # 也可以不用 tanh，改用分位数归一化（更线性、但可能更“硬”）
    # img = (img - lo) / den
    # img = img.clamp(0, 1)

    img_u8 = (img * 255).byte().cpu().numpy()
    Image.fromarray(img_u8).save(os.path.join(save_root, f"diff_latent_step_{idx}.png"))