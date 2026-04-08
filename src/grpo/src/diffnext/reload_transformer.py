"""
重新加载 NOVA Pipeline 的 transformer 模块
用于从训练好的 checkpoint 重新加载 transformer 权重

使用方法:
    python reload_transformer.py
    
或者在代码中调用:
    from reload_transformer import reload_nova_transformer
    pipe = reload_nova_transformer(pipe, "/path/to/checkpoint")
"""

import sys
sys.path.insert(0, '$PROJECT_ROOT/src/grpo/src/nova')

import os
import torch
from safetensors.torch import load_file
from diffnext.pipelines import NOVAPipeline


def reload_nova_transformer(
    pipeline: NOVAPipeline,
    ckpt_path: str,
    device: str = "cuda",
    verbose: bool = True
) -> NOVAPipeline:
    """
    重新加载 NOVA Pipeline 的 transformer 模块
    
    Args:
        pipeline: 原始的 NOVA Pipeline
        ckpt_path: 包含 model.safetensors 的目录路径或直接指向 safetensors 文件的路径
        device: 加载设备
        verbose: 是否打印详细信息
    
    Returns:
        更新后的 Pipeline
    """
    
    # 处理 ckpt_path
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "model.safetensors")
    
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    if verbose:
        print(f"[reload_transformer] Loading from {ckpt_path}")
    
    # 1) 加载 safetensors 到 CPU
    state = load_file(ckpt_path, device="cpu")
    
    # 获取 transformer
    transformer = pipeline.transformer
    transformer_sd = transformer.state_dict()
    
    # 2) 检查 checkpoint 中的顶级前缀
    ckpt_prefixes = set()
    for k in state.keys():
        prefix = k.split('.')[0]
        ckpt_prefixes.add(prefix)
    if verbose:
        print(f"[reload_transformer] Checkpoint prefixes: {sorted(ckpt_prefixes)}")
    
    # 3) 构建新的 state_dict，尝试匹配权重
    new_sd = {}
    matched_keys = []
    unmatched_keys = []
    
    for k, v in state.items():
        matched = False
        
        # 策略1: 尝试直接匹配
        if k in transformer_sd:
            new_sd[k] = v.to(dtype=torch.float32)
            matched_keys.append(k)
            matched = True
        else:
            # 策略2: 尝试去掉常见前缀
            new_k = k
            for prefix in ['module.', 'transformer.', 'model.']:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    if new_k in transformer_sd:
                        new_sd[new_k] = v.to(dtype=torch.float32)
                        matched_keys.append(f"{k} -> {new_k}")
                        matched = True
                        break
            
            # 策略3: 对于 transformer 子模块 (image_decoder, image_encoder 等)
            # 检查 checkpoint 中的顶级前缀是否匹配 transformer 的子模块
            if not matched:
                parts = k.split('.')
                if len(parts) >= 2:
                    # 例如: image_decoder.blocks.0.weight -> transformer.image_decoder.blocks.0.weight
                    prefix = parts[0]
                    if prefix in ['image_decoder', 'image_encoder', 'mask_embed', 'text_embed', 'video_encoder', 'video_pos_embed']:
                        # 构建新的 key: image_decoder.xxx -> image_decoder.xxx
                        new_k = k
                        if new_k in transformer_sd:
                            new_sd[new_k] = v.to(dtype=torch.float32)
                            matched_keys.append(f"{k} -> {new_k}")
                            matched = True
                        else:
                            # 尝试: image_decoder.xxx -> transformer.image_decoder.xxx
                            new_k = f"transformer.{k}"
                            if new_k in transformer_sd:
                                new_sd[new_k] = v.to(dtype=torch.float32)
                                matched_keys.append(f"{k} -> {new_k}")
                                matched = True
        
        if not matched:
            unmatched_keys.append(k)
    
    if verbose:
        print(f"[reload_transformer] Matched keys: {len(matched_keys)}")
        print(f"[reload_transformer] Unmatched keys: {len(unmatched_keys)}")
    
    if unmatched_keys and verbose:
        print(f"[reload_transformer] First 20 unmatched keys:")
        for k in unmatched_keys[:20]:
            print(f"  {k}")
    
    # 4) 加载权重到 transformer
    transformer.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[reload_transformer] Successfully reloaded transformer weights")
    
    return pipeline


def main():
    """主函数: 加载 pipeline，重新加载 transformer，然后生成测试图像"""
    
    # 配置参数 - 根据你的环境修改这些路径
    model_id = "$CHECKPOINT_ROOT/BAAI/nova-d48w1024-sd512"
    # 你的训练好的 transformer checkpoint 路径
    ckpt_path = "$PROJECT_ROOT/my_personal/nova-test-beta0.01_critic4_1_test2_iter500/model.safetensors"
    
    # 加载原始 pipeline
    print("Loading NOVA Pipeline...")
    model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
    pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
    pipe = pipe.to("cuda")
    
    # 重新加载 transformer
    pipe = reload_nova_transformer(pipe, ckpt_path, device="cuda")
    
    # 测试生成
    prompt = ["a beautiful girl"] * 5
    print("Generating images with reloaded transformer...")
    images = pipe(prompt).images
    
    print(f"Generated {len(images)} images")
    for idx, image in enumerate(images):
        image.save(f"girl_reloaded_{idx}.jpg")
        print(f"Saved to girl_reloaded_{idx}.jpg")


if __name__ == "__main__":
    main()
