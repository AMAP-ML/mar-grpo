
"""
Harmon MAR-style generation utilities with per-diffusion-step logp logging.

This mirrors the NOVA utilities you referenced:
- generate_image_with_logp(...)
- single_forward_without_head(...)

Assumptions (match Harmon code you uploaded):
- model is an instance of `HarmonForConditionalGeneration` (or compatible) from `modeling_harmon.py`
- model.diffloss is an instance of `DiffLoss` from `diffloss.py`
- diffusion backend is `GaussianDiffusion` from `gaussian_diffusion.py` (created via misc.create_diffusion)
- diffusion model conditioning key is `c` (see DiffLoss.forward: model_kwargs=dict(c=z))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from harmon.modeling_harmon import HarmonModel
from harmon.mar import MAR
from harmon.diffloss import DiffLoss
from harmon.gaussian_diffusion import GaussianDiffusion
from harmon.text2image_hf import GENERATION_TEMPLATE,PROMPT_TEMPLATE
import numpy as np
from harmon.modeling_harmon import mask_by_order
from einops import rearrange
import random
import torch as th
from PIL import Image

import math
import torch
from transformers.cache_utils import DynamicCache

# -------------------------
# Low-level: single step logp
# -------------------------

def _gaussian_log_prob(prev_x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Elementwise log N(prev_x; mean, var). Keeps the same shape as inputs.
    Matches NOVA's elementwise formulation (no reduction).
    """
    eps = 1e-12  # avoid var==0 -> NaN
    var_safe = torch.where(var > 0, var, torch.full_like(var, eps))
    return -0.5 * (((prev_x - mean) ** 2) / var_safe + torch.log(2.0 * torch.pi * var_safe))


# logp_j,prev_mean_j,std_j=harmon_single_denoise_with_logp(diffloss=model.diffloss,prev_x=prev_latents_flat,cur_x=cur_latents_flat,
#                                                         z=ar_latents_flat,t=t_flat,pred_ids=pid_flat)


def gaussdiffusion_training(
    gaussdiff:GaussianDiffusion,
    model,
    x,
    x_prev,
    t,
    std_dev_t,
    clip_denoised=True,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    temperature=1.0,
    return_logp=False
):
    """
    Sample x_{t-1} from the model at the given timestep.
    :param model: the model to sample from.
    :param x: the current tensor at x_{t-1}.
    :param t: the value of t, starting at 0 for the first diffusion step.
    :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample.
    :param cond_fn: if not None, this is a gradient function that acts
                    similarly to the model.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :param temperature: temperature scaling during Diff Loss sampling.
    :return: a dict containing the following keys:
                - 'sample': a random sample from the model.
                - 'pred_xstart': a prediction of x_0.
    """
    out = gaussdiff.p_mean_variance(
        model,
        x,
        t,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
    )
    noise = th.randn_like(x)
    nonzero_mask = (
        (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
    )  # no noise when t == 0
    if cond_fn is not None:
        out["mean"] = gaussdiff.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
    # scale the noise by temperature

    # std_dev_t = th.exp(0.5 * out["log_variance"].detach()) * temperature
    # sample = out["mean"] + nonzero_mask * noise * std_dev_t
    sample = x_prev

    log_prob = (
        -((sample - out["mean"]) ** 2) / (2 * (std_dev_t**2).clamp_(1e-6))
        - torch.log(std_dev_t.clamp_(1e-6))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob#.mean(dim=tuple(range(1, log_prob.ndim)))
    mean = out["mean"]#.mean(dim=tuple(range(1, out["mean"].ndim)))
    std_dev_t = std_dev_t#.mean(dim=tuple(range(1, std_dev_t.ndim)))

    if return_logp:
        #xt; xt_prev
        return {"sample": sample, "xt": x, "xt_prev":sample, "pred_xstart": out["pred_xstart"], 
                "log_prob": log_prob, 'std_dev_t': std_dev_t,
                "mean": mean}
    else:
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


def harmon_single_denoise_with_logp(
    diffloss:DiffLoss,
    prev_x: torch.Tensor,
    cur_x: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    std_dev_t: torch.Tensor,
    cfg: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute elementwise logp for one diffusion transition:
        log p_theta(x_{t-1}=prev_x | x_t=cur_x, cond=z)

    Returns:
      logp  : same shape as prev_x/cur_x
      mean  : p_theta mean (same shape)
      std   : sqrt(var) (same shape)
    """

    timestep = t
    model_kwargs = dict(c=z)
    sample_fn = diffloss.net.forward

    out=gaussdiffusion_training(diffloss.gen_diffusion,model=sample_fn,x=cur_x,x_prev=prev_x,\
                                        std_dev_t=std_dev_t, t=timestep,\
                                        clip_denoised=False, model_kwargs=model_kwargs,
                                        temperature=1.0,return_logp=True)
    # out["xt"]
    # out["xt_prev"]
    prev_mean=out["mean"]
    std=out["std_dev_t"]   
    logp=out["log_prob"]
    return logp, prev_mean, std


@torch.no_grad()
def harmon_denoise_with_logp(
    diffloss:DiffLoss,
    *,
    z: torch.Tensor,
    noise: torch.Tensor,
    temperature: float = 1.0,
    cfg: float = 1.0,
    progress: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Run the full diffusion sampling loop while logging:
      xs      : list of x_{t-1} (after the step)    [T] each [N, C]
      xs_next : list of x_t (before the step)       [T] each [N, C]
      logps   : list of elementwise logp for step   [T] each [N, C]
      means   : list of p_theta means               [T] each [N, C]
      sample  : final x_0                           [N, C]
    """
    if cfg != 1.0:
        # caller should already have duplicated batch for cfg (like Harmon.sample does)
        model_kwargs = dict(c=z, cfg_scale=cfg)
        sample_fn = diffloss.net.forward_with_cfg
    else:
        model_kwargs = dict(c=z)
        sample_fn = diffloss.net.forward

    xs, xs_next, logps, means = [], [], [], []

    img = noise
    # gen_diffusion.p_sample_loop_progressive iterates i = T-1..0 (see gaussian_diffusion.py)
    for out in diffloss.gen_diffusion.p_sample_loop_progressive(
        sample_fn,
        noise.shape,
        noise=img,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=progress,
        temperature=temperature,
    ):
        cur = img
        nxt = out["sample"]
        mean = out.get("mean", None)
        log_var = out.get("log_variance", None)
        if mean is None or log_var is None:
            # if your GaussianDiffusion.p_sample doesn't expose mean/log_variance,
            # you can recompute them via p_mean_variance here.
            t = out.get("t", None)  # usually absent
            raise RuntimeError("Expected p_sample() to return 'mean' and 'log_variance'.")
        var = torch.exp(log_var)
        logp = _gaussian_log_prob(nxt, mean, var)

        xs_next.append(cur)
        xs.append(nxt)
        logps.append(logp)
        means.append(mean)

        img = nxt

    return xs, xs_next, logps, means, img


# -------------------------
# Mid-level: AR forward without diffusion head
# -------------------------


def prepare_forward_input(process_class:HarmonModel,
                            x,
                            inputs_embeds=None,
                            input_ids=None,
                            attention_mask=None,
                            past_key_values=None):
    b, l, _ = x.shape
    attention_mask = attention_mask.to(device=process_class.device, dtype=torch.bool)
    attention_mask = torch.cat([
        attention_mask, attention_mask.new_ones(b, l)
    ], dim=1)
    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids[position_ids < 0] = 0


    # prepare context
    if past_key_values is not None:
        inputs_embeds = x
        position_ids = position_ids[:, -l:]
    else:
        if inputs_embeds is None:
            input_ids = input_ids.to(process_class.device)
            inputs_embeds = process_class.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

    return dict(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values)

def extract_visual_feature(process_class, mar:MAR, x, mask=None, detach=False):
    b, m, n, _ = x.shape
    x = x.view(b, m*n, -1)
    # x: b mn c
    if mask is None:
        mask = torch.zeros_like(x[..., 0])
    null_embeds = mar.fake_latent.expand(x.shape[0], -1)
    x_enc = mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

    z_enc = process_class.proj_in(x_enc)
    # Move buffers to the end of the image sequence
    z_enc = torch.cat([
        z_enc[:, mar.buffer_size:],
        z_enc[:, :mar.buffer_size]], dim=1)

    if detach:
        x_enc = x_enc.detach()
        z_enc = z_enc.detach()

    return x_enc, z_enc

def forward_mae_encoder_process_class(process_class, mar_model, x, mask, detach=False, **context):
    b, m, n, _ = x.shape
    x_enc, z_enc = extract_visual_feature(process_class, mar_model, x, mask=mask, detach=detach)
    inputs = prepare_forward_input(process_class, x=z_enc, **context)
    output = process_class.llm_model(**inputs, return_dict=True)

    z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]

    # move buffers back to the start of the image sequence
    z_llm = torch.cat([
        z_llm[:, -process_class.mar.buffer_size:],
        z_llm[:, :-process_class.mar.buffer_size]], dim=1)

    # residual learning
    x_enc = x_enc + process_class.proj_out(z_llm)

    return x_enc


def single_forward_without_head(process_class:HarmonModel, mar_model:MAR,model_inputs, attention_mask, image_shape=(32,32)):
            # train_inputs={
            #     "prev_ids":train_inputs_prev_ids,#step i 已知的token
            #     "pred_ids":train_inputs_pred_ids,#step i 新预测的token
            #     "x":train_inputs_x,#[b,4,64,64]#当前maskstep输入的latent
            #     "prompt":train_inputs_class_embed#呃不知道cur和ref prompt embed设置成相同的会不会有问题
            #               }
    m,n=image_shape
    bsz,_,c=model_inputs["x"].shape
    tokens=model_inputs["x"].view(bsz,m,n,-1)#torch.Size([4, 1024, 16])
    mask=model_inputs["prev_ids"]
    past_key_values=model_inputs["prompt"]
    # mae encoder
    x = forward_mae_encoder_process_class(process_class, mar_model, tokens, mask.to(process_class.dtype), past_key_values=past_key_values, attention_mask=attention_mask)
    # mae decoder

    #         z = mar.forward_mae_decoder(x_enc, mask.to(process_class.dtype), image_shape=(m, n), x_con=x_con)
    z = mar_model.forward_mae_decoder(x, mask.to(process_class.dtype), image_shape=(m,n), x_con=None)
    return z


def get_prompt_embed(process_class:HarmonModel,inputs_embeds):
    output = process_class.llm_model(inputs_embeds=inputs_embeds,
                            attention_mask=None,
                            position_ids=None,
                            past_key_values=DynamicCache.from_legacy_cache(),
                            return_dict=True,
                            use_cache=True)
    past_key_values = output.past_key_values

    process_class.curtail_cache(past_key_values, inputs_embeds.shape[1])
    return past_key_values


# -------------------------
# High-level: generate images with logs (outer MAR steps + inner diffusion steps)
# -------------------------

@dataclass
class HarmonFrameLogs:
    xs: List[torch.Tensor]         # length = num_outer_steps; each is [T, B, C] or list-of-steps
    xs_next: List[torch.Tensor]
    out_means: List[torch.Tensor]
    prev_ids: List[torch.Tensor]
    pred_ids: List[torch.Tensor]
    x_latent: List[torch.Tensor]   # tokens grid snapshots at each outer step



@torch.no_grad()
def sample_with_logp(process_class:HarmonModel,
            mar:MAR,
            input_ids=None, inputs_embeds=None,
            attention_mask=None, num_iter=64, cfg=1.0, cfg_schedule="constant", temperature=1.0,
            progress=False, mask=None, past_key_values=None, image_shape=None, x_con=None,
            return_latent_with_seed=False,
            selected_mask_steps=None, **kwargs):
    if inputs_embeds is None and input_ids is not None:
        inputs_embeds = process_class.llm.get_input_embeddings()(input_ids).to()

    bsz = attention_mask.shape[0]
    if cfg != 1.0:
        assert bsz % 2 == 0

    if image_shape is None:
        m = n = int(process_class.gen_seq_len ** 0.5)
    else:
        m, n = image_shape

    if mask is None:
        mask = torch.ones(bsz, m*n, device=process_class.device, dtype=process_class.dtype)
    else:
        mask = mask.view(bsz, m*n)
    tokens = torch.zeros(bsz, m*n, process_class.token_embed_dim,
                            device=process_class.device, dtype=process_class.dtype)
    orders = mar.sample_orders(bsz, seq_len=m*n)
    if cfg != 1.0:
        orders[bsz//2:] = orders[:bsz//2]

    indices = list(range(num_iter))

    # past key values can be prepared outside (usually in multi-turn editing)
    if past_key_values is None:
        output = process_class.llm_model(inputs_embeds=inputs_embeds,
                                attention_mask=None,
                                position_ids=None,
                                past_key_values=DynamicCache.from_legacy_cache(),
                                return_dict=True,
                                use_cache=True)
        past_key_values = output.past_key_values


    if selected_mask_steps is None:
        selected_mask_steps = set()
    else:
        selected_mask_steps = set(selected_mask_steps)


    step_logs = {"tokens": [], "diff_xt": [], "diff_xt_prev": [], "log_prob": [], "out_std_dev_t":[], "out_mean": [], "mask_list": [], "mask_to_pred_list": [], "final_diff_latents": []}

    # generate latents
    all_ar_latents=None
    for step in indices:
        keep_this_step = (step in selected_mask_steps) or (step == num_iter - 1) or (step+1 in selected_mask_steps)

        cur_tokens = tokens.clone()
        x_enc = process_class.forward_mae_encoder(tokens.view(bsz, m, n, -1),
                                            mask.to(process_class.dtype),
                                            past_key_values=past_key_values,
                                            # inputs_embeds=inputs_embeds,
                                            attention_mask=attention_mask)
        process_class.curtail_cache(past_key_values, inputs_embeds.shape[1])

        # z:torch.Size([8, 1024, 1280])
        z = mar.forward_mae_decoder(x_enc, mask.to(process_class.dtype), image_shape=(m, n), x_con=x_con)


        # x_pca = torch.pca_lowrank(z[0].reshape(-1,1280).float(), q=3)[0]
        # img = x_pca.reshape(32, 32, 3);img = (img - img.min()) / (img.max() - img.min());img = (img * 255).byte().cpu().numpy();Image.fromarray(img).save("$PROJECT_ROOT/src/grpo/src/harmon/ar_latent.png")

        # mask ratio for the next round, following MaskGIT and MAGE.
        mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(m*n * mask_ratio)]).to(process_class.device)

        # masks out at least one for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).to(process_class.device),
                                    torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

        # get masking for next iteration and locations to be predicted in this iteration
        mask_next = mask_by_order(mask_len[0], orders, bsz, m*n).to(process_class.device)#下一个mask step的mask
        if cfg != 1.0:
            mask_next[bsz//2:] = mask_next[:bsz//2]
        if step >= num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())#mask=1表示未知的token；
            # mask和mask_next不一样就设置成true，其余false

        if return_latent_with_seed:
            z_=z.clone()
            if all_ar_latents==None:
                all_ar_latents=z_.mul_(mask_to_pred[...,None])
            else:
                all_ar_latents=all_ar_latents.clone()
                all_ar_latents.add_(z_.mul_(mask_to_pred[...,None]))

        # >>> MOD START: record outer-step ids + x_latent snapshot (before update)
        # prev_ids: indices already filled in current mask (mask==0)
        # pred_ids: indices to be predicted this step (mask_to_pred==True)
        prev_ids_step = (~mask[:bsz].bool()).nonzero(as_tuple=False)   # [num_known, 2] (b, idx)
        pred_ids_step = (mask_to_pred).nonzero(as_tuple=False)         # [num_pred, 2] (b, idx)

        # reshape into [B, K] assuming each b has same count (MaskGIT schedule ensures this)
        prev_ids_step = prev_ids_step[:, 1].view(bsz, -1).long()
        pred_ids_step = pred_ids_step[:, 1].view(bsz, -1).long()

        mask = mask_next
        # if not cfg == 1.0:
        #     mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        # sample token latents for this step
        # # z:torch.Size([8, 1024, 1280])
        if keep_this_step:
            z = z.flatten(0,1)#keep的话就保留所有token
        else:
            z = z[mask_to_pred.nonzero(as_tuple=True)]#如果不keep，只decode当前step要预测的token就行了

        # cfg schedule follow Muse
        if cfg_schedule == "linear":
            cfg_iter = 1 + (cfg - 1) * (m*n - mask_len[0]) / (m*n)
        elif cfg_schedule == "constant":
            cfg_iter = cfg
        else:
            raise NotImplementedError
        return_values_to_parse = mar.diffloss.sample(z, temperature, cfg_iter, return_logp=True)#.to(process_class.dtype)
        sampled_token_latent, diff_xt, diff_xt_prev, log_prob, out_mean, out_std_dev_t, pred_x0 = return_values_to_parse
        sampled_token_latent = sampled_token_latent.to(process_class.dtype)
        if keep_this_step:
            sampled_token_latent = sampled_token_latent.reshape(bsz,-1,sampled_token_latent.shape[-1])
        # if not cfg == 1.0:
        #     sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
        #     mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)


        if cfg != 1.0:
            cur_tokens[bsz//2:] = cur_tokens[:bsz//2]
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
            tokens_clone, _ = tokens.chunk(2, dim=0)  # tokens：mask step输入的token
            mask_clone, _ = mask.clone().chunk(2, dim=0)

        if keep_this_step:
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent[mask_to_pred.nonzero(as_tuple=True)]
        else:
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent

            # x_pca = torch.pca_lowrank(cur_tokens[0].reshape(-1,16).float(), q=3)[0]
            # img = x_pca.reshape(32, 32, 3);img = (img - img.min()) / (img.max() - img.min());img = (img * 255).byte().cpu().numpy();Image.fromarray(img).save("$PROJECT_ROOT/src/grpo/src/harmon/diff_latent.png")

        if keep_this_step:
            step_logs["tokens"].append(tokens_clone)#step i 输入的ar tokens
            step_logs["mask_list"].append(mask_clone)#step i 已知的token
            step_logs["mask_to_pred_list"].append(mask_to_pred)#step i 要预测的token

            # log_prob, out_mean, out_std_dev_t
            # 丢掉cfg的部分
            t_num,_,channel=log_prob.shape
            log_prob=log_prob.reshape(t_num,bsz,-1,channel).chunk(2,dim=1)[0]
            out_mean=out_mean.reshape(t_num,bsz,-1,channel).chunk(2,dim=1)[0]
            diff_xt=diff_xt.reshape(t_num,bsz,-1,channel).chunk(2,dim=1)[0]
            diff_xt_prev=diff_xt_prev.reshape(t_num,bsz,-1,channel).chunk(2,dim=1)[0]
            out_std_dev_t=out_std_dev_t.reshape(t_num,bsz,-1,channel).chunk(2,dim=1)[0]

            step_logs["log_prob"].append(log_prob)
            step_logs["out_mean"].append(out_mean)
            step_logs["diff_xt"].append(diff_xt)
            step_logs["diff_xt_prev"].append(diff_xt_prev)
            step_logs["out_std_dev_t"].append(out_std_dev_t)
        else:
            step_logs["tokens"].append(None)
            step_logs["mask_list"].append(None)
            step_logs["mask_to_pred_list"].append(None)

            step_logs["log_prob"].append(None)
            step_logs["out_mean"].append(None)
            step_logs["diff_xt"].append(None)
            step_logs["diff_xt_prev"].append(None)
            step_logs["out_std_dev_t"].append(None)
            del log_prob, out_mean, diff_xt, diff_xt_prev, out_std_dev_t, sampled_token_latent
            
        tokens = cur_tokens.clone()

    num_seed=5
    print('generation seed: ',num_seed)
    seed_list = [random.randint(0, 10000) for _ in range(num_seed)]
    sample_all_diff_latents_seed={"diff_xt": [], "diff_xt_prev": [], "latents_seed":[]}
    # sample_all_diff_latents_seed = {"xs": [], "xs_next": [], "x_latent": [], "log_probs": [], "out_means": [], "prev_ids": [], "pred_ids": []}
    if return_latent_with_seed:
        for seed_ in seed_list:   # seed_list 长度 = 10
            torch.manual_seed(seed_)

            return_values_to_parse = mar.diffloss.sample(all_ar_latents.flatten(0,1), temperature, cfg_iter, return_logp=True)#.to(process_class.dtype)
            _, diff_xt, diff_xt_prev, _, _, _, pred_x0 = return_values_to_parse

            T,_,C=diff_xt.shape
            diff_xt = diff_xt.reshape(T,bsz,-1,C).chunk(2,dim=1)[0]
            diff_xt_prev = diff_xt_prev.reshape(T,bsz,-1,C).chunk(2,dim=1)[0]#T,bsz//2,L,C
            # xs_seed, xs_seed_next, logps_seed, out_means_seed, sample_seed = self.denoise_with_logp(all_ar_latents,inp_noise,guidance_scaler.clone(),generator,pred_ids=None)
            # sample_all_diff_latents = xs_seed[-1]
            sample_all_diff_latents_seed["diff_xt"].append(diff_xt)
            sample_all_diff_latents_seed["diff_xt_prev"].append(diff_xt_prev)
            sample_all_diff_latents_seed["latents_seed"].append(diff_xt_prev[-1])#bsz//2,L,C
            #sampled_token_latent = sampled_token_latent.reshape(bsz,-1,sampled_token_latent.shape[-1])

    pred = process_class.decode(tokens.view(bsz, m, n, -1))
    step_logs["final_diff_latents"].append(tokens[:bsz//2,...].clone())

    if cfg != 1.0:
        pred = pred[:bsz//2]
    return pred,step_logs,inputs_embeds.chunk(2, dim=0)[0],attention_mask.chunk(2, dim=0)[0],sample_all_diff_latents_seed


@torch.inference_mode()
def generate_image_with_logp(
                    process_class:HarmonModel,
                    model:MAR,
                    prompts,
                    negative_prompt,
                    tokenizer,
                    selected_mask_steps=None,
                    grid_size=1,   # will produce 2 x 2 images per prompt
                    num_steps=64, cfg_scale=3.0, temperature=1.0, image_size=512
                    ):

    assert image_size == 512
    m = n = image_size // 16

    prompts = [
                  PROMPT_TEMPLATE['INSTRUCTION'].format(input=prompt)
                  for prompt in prompts
              ]

    if cfg_scale != 1.0:
        prompts += [PROMPT_TEMPLATE['INSTRUCTION'].format(input=negative_prompt)] * len(prompts)

    inputs = tokenizer(
        prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    images,step_logs,prompt_embeds,attention_mask,sample_all_diff_latents_seed = sample_with_logp(process_class,model, **inputs, num_iter=num_steps, 
                        cfg=cfg_scale, cfg_schedule="constant",return_latent_with_seed=True, selected_mask_steps=selected_mask_steps,
                          temperature=temperature, progress=False, image_shape=(m, n))

    images = rearrange(images, '(m n b) c h w -> b (m h) (n w) c', m=grid_size, n=grid_size)

    images = torch.clamp(
        127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    oup_pil_images=[]
    for idx, image in enumerate(images):
        oup_pil_images.append(Image.fromarray(image))

    return oup_pil_images, step_logs, prompt_embeds, attention_mask, sample_all_diff_latents_seed
