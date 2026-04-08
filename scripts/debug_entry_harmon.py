import os
import subprocess

os.environ['HF_ENDPOINT']="https://hf-mirror.com"
os.environ['HUGGINGFACE_HUB_BASE_URL']="https://hf-mirror.com"
os.environ['ENABLE_TORCH_COMPILE']='false'
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "72000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key

os.environ["DEBUG_MODE"] = "true"
os.environ["LOG_PATH"] = "./outputs/debug.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

run_name = "t2i-r1-harmon-test-beta0.01_25diff_test6_multistage_critic_ckpt400_absim0.0_lr1e-6"
qwen_path = "$CHECKPOINT_ROOT/harmon/Harmon-1_5B"
hf_dataset = "../../../data/geneval_and_t2i_data_final.json"
output_dir = f"$PROJECT_ROOT/workspace_t2i-r1_harmon/{run_name}"

cmd = [
    "torchrun",
    "--nproc_per_node=3",
    "--nnodes=1",
    "--node_rank=0",
    "--master_addr=127.0.0.1",
    "--master_port=12346",
    "open_r1/grpo_v1_harmon.py",
    "--use_vllm", "False",
    "--deepspeed", "../configs/zero2.json",
    "--output_dir", output_dir,
    "--model_name_or_path", qwen_path,
    "--semantic_cot", "False",
    "--dataset_name", hf_dataset,
    "--max_prompt_length", "512",
    "--max_completion_length", "1024",
    "--temperature", "1.0",
    "--num_generations", "1",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--logging_steps", "1",
    "--bf16",
    "--torch_dtype", "bfloat16",
    "--report_to", "wandb",
    "--gradient_checkpointing", "false",
    "--attn_implementation", "flash_attention_2",
    "--max_steps", "1200",
    "--run_name", run_name,
    "--save_steps", "50",
    "--new_generations_image", "4",#每个cot prompt生成几张图像
    "--image_token_num_per_image", "576",
    "--cfg_weight", "5",
    "--reasoning_prompt_path", "../../../data/prompt/reasoning_prompt.txt",
    "--reward_funcs", "hps",
    "--beta", "0.01",
    "--tf32", "true",
    "--learning_rate", "1e-6",
    "--lr_scheduler_type", "constant",
    "--hps_ckpt_path", "$REWARD_MODEL_ROOT/HPSv2/HPS_v2.1_compressed.pt",
    "--clip_ckpt_path", "$CHECKPOINT_ROOT/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    "--git_ckpt_path", "$CHECKPOINT_ROOT/microsoft/git-large-vqav2",
    "--gdino_ckpt_path", "$REWARD_MODEL_ROOT/groundingdino_swint_ogc/groundingdino_swint_ogc.pth",
    "--gdino_config_path", "utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "--orm_ckpt_path", "$REWARD_MODEL_ROOT/ORM-T2I-R1",
    
    "--reward_smooth", "False",
    "--kl_reweight", "False",
    "--update_ref", "False",
    "--progress_learning", "False",
    "--add_noise", "False",
    "--fix_head", "True",
    "--fix_ar", "False",
    "--use_critic_token", "True",
    "--all_diff_timesteps", "False",
    "--small_lr_for_head", "False",
    "--ema_for_head", "False",
    "--top_mask_percent", "0.3",
    "--critic_for_kl", "False",
    "--use_latent_sim_coef", "True",
    "--transformer_path", "$PROJECT_ROOT/workspace_t2i-r1_harmon/t2i-r1-harmon-test-beta0.01_25diff_test6_fixar/checkpoint-400",
    "--latent_sim_thresh", "0.0",
]

# 设置工作目录
os.chdir("$PROJECT_ROOT/src/grpo/src")

# 添加 PYTHONPATH 环境变量
os.environ['WANDB_PROJECT']="harmon_rl_new"
os.environ["PYTHONPATH"] = f"{os.getcwd()}/..:" + os.environ.get("PYTHONPATH", "")

# 执行命令
subprocess.run(cmd)
