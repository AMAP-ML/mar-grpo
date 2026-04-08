import torch
from diffnext.pipelines import NOVAPipeline
from diffnext.pipelines.pipeline_utils import NOVAPipelineOutput, PipelineMixin
from diffnext.models.transformers.transformer_nova import NOVATransformer3DModel
from diffnext.models.guidance_scaler import GuidanceScaler
import numpy as np
from deepspeed import zero as ds_zero


def encode_prompt(
    transformer,
    guidance_scale,
    prompt,
    num_images_per_prompt=1,
    negative_prompt=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
) -> torch.Tensor:
    """Encode text prompts.

    Args:
        prompt (str or List[str], *optional*):
            The prompt to be encoded.
        num_images_per_prompt (int, *optional*, defaults to 1):
            The number of images that should be generated per prompt.
        negative_prompt (str or List[str], *optional*):
            The prompt or prompts not to guide the image generation.
        prompt_embeds (List[torch.Tensor], *optional*)
            A list of precomputed prompt embeddings.
        negative_prompt_embeds (List[torch.Tensor], *optional*)
            A list of precomputed negative prompt embeddings.

    Returns:
        torch.Tensor: The prompt embedding.
    """

    def select_or_pad(a, b, n=1):
        return [a or b] * n if isinstance(a or b, str) else (a or b)

    # text_enc = transformer.embedder.text_encoder          # e.g. PhiEncoderModel
    tokenizer, text_enc = transformer.text_embed.encoders
    embedder = transformer.text_embed
    
    def zero3_gather_text_encoder():
        if ds_zero is None:
            # 非 ZeRO 环境，直接 no-op
            from contextlib import contextmanager
            @contextmanager
            def _noop(): yield
            return _noop()
        return ds_zero.GatheredParameters(list(text_enc.parameters()), modifier_rank=None)

    # 4) 在聚合上下文里再做 encode/读 weight
    with zero3_gather_text_encoder():
        # 现在再检查就不会是空
        W = text_enc.model.embed_tokens.weight
        V = W.size(0); pad = text_enc.model.embed_tokens.padding_idx
        print(f"[gathered] V={V}, pad={pad}")

        if prompt_embeds is not None:
            prompt_embeds = embedder.encode_prompts(prompt_embeds)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = embedder.encode_prompts(negative_prompt_embeds)
        if prompt_embeds is not None:
            if negative_prompt_embeds is None and guidance_scale > 1:
                bs, seqlen = prompt_embeds.shape[:2]
                negative_prompt_embeds = embedder.weight[:seqlen].expand(bs, -1, -1)
            if guidance_scale > 1:
                c = torch.cat([prompt_embeds, negative_prompt_embeds])
            return c.repeat_interleave(num_images_per_prompt, dim=0)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = select_or_pad(negative_prompt, "", len(prompt))
        prompts = prompt + (negative_prompt if guidance_scale > 1 else [])
        c = embedder.encode_prompts(prompts)
    return c.repeat_interleave(num_images_per_prompt, dim=0)



def generate_image_with_logp(
        process_class:NOVAPipeline,#ok，因为这里输入的参数是会通过inputs传到整个生成步骤的所有函数的，所以一个都不能省
        prompt,
        transformer:NOVATransformer3DModel,
        vae=None,
        generator=None,
        disable_progress_bar=True,
        num_inference_steps=64,
        num_diffusion_steps=25,
        max_latent_length=1,
        guidance_scale=5,
        guidance_trunc=0,
        guidance_renorm=1,
        image_guidance_scale=0,
        spatiotemporal_guidance_scale=0,
        flow_shift=None,
        motion_flow=5,
        negative_prompt=None,
        image=None,
        num_images_per_prompt=1,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        return_latent_with_seed=False,
        output_type="pil"):
    inputs = {"generator": generator, **locals()}
    num_patches = int(np.prod(transformer.config.image_base_size))
    mask_ratios = np.cos(0.5 * np.pi * np.arange(num_inference_steps + 1) / num_inference_steps)
    mask_length = np.round(mask_ratios * num_patches).astype("int64")
    inputs["num_preds"] = mask_length[:-1] - mask_length[1:]
    inputs["tqdm1"] = max_latent_length > 1 and not disable_progress_bar
    inputs["tqdm2"] = max_latent_length == 1 and not disable_progress_bar
    inputs["prompt"] = encode_prompt(transformer, guidance_scale, prompt,1,negative_prompt=None)#实际上就是输入prompt变量
    prompt_embed=inputs["prompt"].clone()
    inputs["latents"] = process_class.prepare_latents(None, 1, generator, None)
    inputs["batch_size"] = len(inputs["prompt"]) // (2 if guidance_scale > 1 else 1)
    inputs["motion_flow"] = [motion_flow] * inputs["batch_size"]
    
    transformer.return_logp=True
    outputs = transformer(inputs,return_latent_with_seed=return_latent_with_seed)
    if return_latent_with_seed:
        outputs,all_ar_latents,sample_all_diff_latents_seed=outputs
    outputs["x"] = process_class.image_processor.decode_latents(vae, outputs["x"])
    output_name = {4: "images", 5: "frames"}[len(outputs["x"].shape)]
    outputs["x"] = process_class.image_processor.postprocess(outputs["x"], "pil")
    if return_latent_with_seed:
        return NOVAPipelineOutput(**{output_name: outputs["x"]}), outputs["frame_logs"],prompt_embed,all_ar_latents,sample_all_diff_latents_seed
    else:
        return NOVAPipelineOutput(**{output_name: outputs["x"]}), outputs["frame_logs"],prompt_embed


def single_forward_without_head(transformer:NOVATransformer3DModel,model_inputs):
    # 此处mask只是创造一个默认的mask，用来提供一个形状，在train的时候不会用这个mask
    # transformer.mask_embed.mask=model_inputs["x"].new_ones(model_inputs["x"].shape[:-1] + (1,))#这个换成从路径里得到的latent x
    # pred_mask, pred_ids = transformer.mask_embed.get_pred_mask(5)#生成随机mask for next step；pred_ids是next step要预测的位置
            
    transformer.pipeline_preprocess(model_inputs)#恒等映射，不进行任何操作
    transformer.preprocess(model_inputs)
    loss=transformer.single_forward_without_head(model_inputs)
    # loss=transformer.train_video(model_inputs)
    return loss


def from_ar_latent_to_img(
        process_class:NOVAPipeline,#ok，因为这里输入的参数是会通过inputs传到整个生成步骤的所有函数的，所以一个都不能省
        prompt,
        transformer:NOVATransformer3DModel,
        vae=None,
        generator=None,
        latent=None,#ar model的latent
        noise=None,
        disable_progress_bar=True,
        num_inference_steps=64,
        num_diffusion_steps=25,
        max_latent_length=1,
        guidance_scale=5,
        guidance_trunc=0,
        guidance_renorm=1,
        image_guidance_scale=0,
        spatiotemporal_guidance_scale=0,
        flow_shift=None,
        motion_flow=5,
        negative_prompt=None,
        image=None,
        num_images_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil"):
    inputs = {"generator": generator, **locals()}
    num_patches = int(np.prod(transformer.config.image_base_size))
    mask_ratios = np.cos(0.5 * np.pi * np.arange(num_inference_steps + 1) / num_inference_steps)
    mask_length = np.round(mask_ratios * num_patches).astype("int64")
    inputs["num_preds"] = mask_length[:-1] - mask_length[1:]
    inputs["tqdm1"] = max_latent_length > 1 and not disable_progress_bar
    inputs["tqdm2"] = max_latent_length == 1 and not disable_progress_bar
    inputs["prompt"] = encode_prompt(transformer, guidance_scale, prompt,1,negative_prompt=None)#实际上就是输入prompt变量
    prompt_embed=inputs["prompt"].clone()
    inputs["latents"] = process_class.prepare_latents(None, 1, generator, None)
    inputs["batch_size"] = len(inputs["prompt"]) // (2 if guidance_scale > 1 else 1)
    inputs["motion_flow"] = [motion_flow] * inputs["batch_size"]
    
    transformer.return_logp=True
    # outputs = transformer(inputs)

    noise.normal_(generator=generator)#b,c,h,w
    guidance_scaler = GuidanceScaler(**inputs)
    guidance_scaler.guidance_scale=1
    sample = transformer.denoise(latent, noise, guidance_scaler.clone(), generator, pred_ids=None)
    sample=transformer.image_encoder.patch_embed.unpatchify(sample)
    
    sample = process_class.image_processor.decode_latents(vae, sample)
    output_name = {4: "images", 5: "frames"}[len(sample.shape)]
    sample = process_class.image_processor.postprocess(sample, "pil")
    return NOVAPipelineOutput(**{output_name: sample})