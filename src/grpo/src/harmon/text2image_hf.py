from transformers import AutoTokenizer, AutoModel

harmon_root='$CHECKPOINT_ROOT/harmon/Harmon-1_5B'

import os
# # if int(os.environ.get("LOCAL_RANK", 0)) == 0:
#     
import torch
from transformers import AutoTokenizer, AutoModel
from einops import rearrange
from harmon.modeling_harmon import HarmonModel
from PIL import Image

PROMPT_TEMPLATE = dict(
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'])

GENERATION_TEMPLATE = "Generate an image: {text}"

@torch.no_grad()
def generate_images(prompts,
                    negative_prompt,
                    tokenizer,
                    model:HarmonModel,
                    output,
                    grid_size=2,   # will produce 2 x 2 images per prompt
                    num_steps=64, cfg_scale=3.0, temperature=1.0, image_size=512):
    assert image_size == 512
    m = n = image_size // 16

    prompts = [
                  PROMPT_TEMPLATE['INSTRUCTION'].format(input=prompt)
                  for prompt in prompts
              ] * (grid_size ** 2)

    if cfg_scale != 1.0:
        prompts += [PROMPT_TEMPLATE['INSTRUCTION'].format(input=negative_prompt)] * len(prompts)

    inputs = tokenizer(
        prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    images = model.sample(**inputs, num_iter=num_steps, cfg=cfg_scale, cfg_schedule="constant",
                          temperature=temperature, progress=True, image_shape=(m, n))
    images = rearrange(images, '(m n b) c h w -> b (m h) (n w) c', m=grid_size, n=grid_size)

    images = torch.clamp(
        127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    os.makedirs(output, exist_ok=True)
    for idx, image in enumerate(images):
        Image.fromarray(image).save(f"{output}/{idx:08d}.jpg")

harmon_tokenizer = AutoTokenizer.from_pretrained(harmon_root,
                                                 trust_remote_code=True)
harmon_model = AutoModel.from_pretrained(harmon_root,
                                         trust_remote_code=True).cuda().bfloat16().eval()

texts = ['a dog on the left and a cat on the right.',
         'a photo of a pink stop sign.']
pos_prompts = [GENERATION_TEMPLATE.format(text=text) for text in texts]
neg_prompt = 'Generate an image.'   # for classifier-free guidance

generate_images(prompts=pos_prompts,
                negative_prompt=neg_prompt,
                tokenizer=harmon_tokenizer,
                model=harmon_model,
                output='output',)
