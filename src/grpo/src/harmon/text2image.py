import os
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import sys
sys.path.insert(0,'$PROJECT_ROOT/src/grpo/src/')

from harmon.modeling_harmon import HarmonModel
from transformers import AutoTokenizer
from harmon.mar import MAR
from harmon.harmon_generate_image_with_logp import generate_image_with_logp
from harmon.text2image_hf import PROMPT_TEMPLATE,GENERATION_TEMPLATE

model_name='$CHECKPOINT_ROOT/harmon/Harmon-1_5B'

process_class = HarmonModel.from_pretrained(model_name).cuda().bfloat16().eval()
model:MAR = process_class.mar

neg_prompt='Generate an image.'
image_gen_prompt_list=[GENERATION_TEMPLATE.format(text='a picture of a dog')]

# Harmon 里 LLM 是 Qwen2ForCausalLM
# tokenizer 一般用与 LLM 对应的 tokenizer 路径（你可以在 args 里放 tokenizer_name）
tokenizer_name = model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

inference_variables_to_parse = generate_image_with_logp(
    process_class,model,image_gen_prompt_list,neg_prompt,
    tokenizer,selected_mask_steps=None,
)

images,frame_logs,prompt_embed,attention_mask,sample_all_diff_latents_seed = inference_variables_to_parse

# 指定保存目录
save_dir = '$PROJECT_ROOT/src/grpo/src/harmon'
os.makedirs(save_dir, exist_ok=True)
# 保存每张图像
for idx, img in enumerate(images):
    save_path = os.path.join(save_dir, f"{idx}.png")
    img.save(save_path)