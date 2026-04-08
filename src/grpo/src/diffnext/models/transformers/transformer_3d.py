# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Base 3D transformer model for video generation."""

from typing import Dict

import torch
from torch import nn
from tqdm import tqdm
import random

from diffnext.models.guidance_scaler import GuidanceScaler
from diffnext.utils.gather_tensor import gather_tensor
from diffnext.schedulers.scheduling_flow_with_logp import sde_step_with_logprob
# from diffnext.schedulers.scheduling_cfm import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler


# @torch.no_grad()
def ddpm_step_grouped(scheduler, model_pred, t_vec, cur_x, generator=None):
    """
    model_pred: [N, ...]
    t_vec:      [N]  每个样本对应的 timestep（int64 tensor）
    cur_x:      [N, ...]
    返回:
      prev_x_tm1: [N, ...]   # step 输出的 prev_sample
      x0_hat:     [N, ...]   # step 输出的 pred_original_sample
    """
    assert t_vec.dim() == 1 and t_vec.numel() == model_pred.shape[0] == cur_x.shape[0]
    device = model_pred.device
    unique_t, inverse = torch.unique(t_vec.to(device, dtype=torch.long),
                                     sorted=True, return_inverse=True)  # unique_t:[m], inverse:[N]

    prev_out = torch.empty_like(cur_x)
    x0_out   = torch.empty_like(cur_x)

    for gi, t_scalar in enumerate(unique_t.tolist()):
        sel = (inverse == gi)              # 该组的样本布尔索引
        mp  = model_pred[sel]
        cx  = cur_x[sel]
        out = scheduler.step(mp, int(t_scalar), cx, generator=generator)  # 单次t，批量前向
        prev_out[sel] = out.prev_sample
        x0_out[sel]   = out.pred_original_sample

    return prev_out, x0_out

        
def posterior_mean_variance(scheduler, x_t, x0, t_idx):          # [T]
    abar   = scheduler.alphas_cumprod.to(x_t.device)        # [T]#也就是scheduler里的

    t = t_idx.to(device=x_t.device, dtype=torch.long).view(-1)
    abar_t      = abar.gather(0, t).view(-1,1,1,1)
    prev_list = [scheduler.previous_timestep(int(tt)) for tt in t.tolist()]
    prev = torch.tensor(prev_list, device=x_t.device, dtype=torch.long)  # [N]

    safe_prev = prev.clamp_min(0)                             # 先夹到合法范围
    abar_prev_vals = abar.gather(0, safe_prev)                # 安全 gather
    abar_t_prev = torch.where(prev >= 0, abar_prev_vals,      # prev==-1 用 1
                            torch.ones_like(abar_prev_vals)).view(-1,1,1,1)
    # abar_t_prev = torch.where(
    #     prev >= 0,
    #     abar.gather(0, prev),
    #     torch.ones_like(abar.gather(0, t))
    # ).view(-1,1,1,1)
    # abar_t_prev = abar.gather(0, prev) if prev[0]>=0 else torch.ones_like(abar.gather(0, t))
    # abar_t_prev=abar_t_prev.view(-1,1,1,1)
    current_alpha_t=abar_t/abar_t_prev
    current_beta_t = 1 - current_alpha_t

    denom = (1.0 - abar_t).clamp_min(1e-12)  # 数值稳定
    posterior_variance = current_beta_t * (1.0 - abar_t_prev) / denom

    coef1 = torch.sqrt(abar_t_prev) * current_beta_t / denom
    coef2 = torch.sqrt(current_alpha_t) * (1.0 - abar_t_prev) / denom
    posterior_mean = coef1 * x0 + coef2 * x_t

    return posterior_mean, posterior_variance
        


class Transformer3DModel(nn.Module):
    """Base 3D transformer model for video generation."""

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        label_embed=None,
        video_pos_embed=None,
        image_pos_embed=None,
        motion_embed=None,
        noise_scheduler=None,
        sample_scheduler=None,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder = video_encoder
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.mask_embed = mask_embed
        self.text_embed = text_embed
        self.label_embed = label_embed
        self.video_pos_embed = video_pos_embed
        self.image_pos_embed = image_pos_embed
        self.motion_embed = motion_embed
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_scheduler
        self.pipeline_preprocess = lambda inputs: inputs
        self.loss_repeat = 4
        self.return_logp = False

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    def preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        add_guidance = inputs.get("guidance_scale", 1) > 1
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        if inputs.get("x", None) is None:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)#inputs["x"]不知道是啥，但empty并不是随机初始化
        if inputs.get("prompt", None) is not None and self.text_embed:#inputs[prompt]是torch.Size([10, 256, 2560])，输入的text feature
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))#inputs[c]是输入的文本条件
        if inputs.get("motion_flow", None) is not None and self.motion_embed:#self.motion_embed是None，因为是图像生成，这步通常直接跳过
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))#如果是视频还会输入光流
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        if video_shape is not None:
            loss = loss.view((-1,) + video_shape).transpose(0, 1).sum((1, 2))
            i2i = loss[1:].sum().mul_(video_shape[0] / (video_shape[0] - 1))
            return {"loss_t2i": loss[0].mul(video_shape[0]), "loss_i2i": i2i}
        return {"loss": loss.sum()}
    
    
    def get_diff_logps(self, z, x, timestep, x_t) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        # noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        # timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        # x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        return 
    

    @torch.no_grad()
    def denoise(self, z, x, guidance_scaler, generator=None, pred_ids=None) -> torch.Tensor:
        """Run diffusion denoising process."""
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            z, pred_ids = guidance_scaler.maybe_disable(t, z, pred_ids)
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            model_pred = self.image_decoder(guidance_scaler.expand(x), timestep, z, pred_ids)
            model_pred = guidance_scaler.scale(model_pred)
            model_pred = self.image_encoder.patch_embed.unpatchify(model_pred)
            x = self.sample_scheduler.step(model_pred, t, x, generator=generator).prev_sample
        return self.image_encoder.patch_embed.patchify(x)
    
    
    # @torch.no_grad()
    def denoise_with_logp(self, z, x, guidance_scaler, generator=None, pred_ids=None) -> torch.Tensor:
        """Run diffusion denoising process."""
        xs, out_means, log_probs, xs_next = [], [], [], []
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            z, pred_ids = guidance_scaler.maybe_disable(t, z, pred_ids)#没啥用
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            model_pred = self.image_decoder(guidance_scaler.expand(x), timestep, z, pred_ids)#model_pred:[b,1024,16],无论加不加pred_ids都是这样
            model_pred = guidance_scaler.scale(model_pred)#cfg操作，当cfg=1时不需要
            model_pred = self.image_encoder.patch_embed.unpatchify(model_pred)
            

            if isinstance(self.sample_scheduler,FlowMatchEulerDiscreteScheduler):#flow matching scheduler
                # # out = self.sample_scheduler.step(model_pred, t, x, generator=generator)
                timestep_ = t.unsqueeze(0).expand(x.shape[0])
                latents, logp_t, prev_latents_mean, std_dev_t = \
                    sde_step_with_logprob(self.sample_scheduler,model_pred,timestep_,x,generator=generator)
                log_probs.append(logp_t)  # sum over all spatial dims
                out_means.append(prev_latents_mean)

                xs_next.append(x)#当前diff step输入的latent
                x = latents
                xs.append(x)#当前diff step得到的latent（下一个diff step输入的latent）
            # ddpm scheduler:
            else:
                out = self.sample_scheduler.step(model_pred, t, x, generator=generator)
                # mean_x = getattr(out, "pred_original_sample", None)
                # mean_x = ddpm_mean(self.sample_scheduler, t, x, out.pred_original_sample)
                # std_t = torch.sqrt(getattr(out, "variance", torch.zeros_like(x)))  # depends on scheduler
                
                x0 = getattr(out, "pred_original_sample", None)
                assert x0 is not None, "scheduler.step 应该返回 pred_original_sample(x0)"

                # 正确的后验 mean/var
                mean_x, var_t = posterior_mean_variance(self.sample_scheduler, x, x0, timestep[:x.shape[0],...])
                eps = 1e-12#避免当t=1时算出logp=nan（此时var=0，即生成过程是确定的）
                var_safe = torch.where(var_t > 0, var_t, torch.full_like(var_t, eps))

                if mean_x is not None and var_safe is not None:
                    logp_t = -0.5 * (((out.prev_sample - mean_x) ** 2) / var_safe
                                    + torch.log(2 * torch.pi * var_safe))
                    log_probs.append(logp_t)  # sum over all spatial dims
                    out_means.append(mean_x)

                xs_next.append(x)#当前diff step输入的latent
                x = out.prev_sample
                xs.append(x)#当前diff step得到的latent（下一个diff step输入的latent）

        xs = torch.stack(xs, dim=0)  # [T, B, ...]
        xs_next = torch.stack(xs_next, dim=0)
        log_probs = torch.stack(log_probs, dim=0)
        out_means = torch.stack(out_means, dim=0)
        # xs[t]：每个 step 的 latent trajectory
        # out_means[t]：每个 step 的 predicted mean
        # log_probs[t]：每个 step 的 log likelihood
        return xs, xs_next, log_probs, out_means, self.image_encoder.patch_embed.patchify(x)
        # return self.image_encoder.patch_embed.patchify(x)
        
    # @torch.no_grad()
    def single_denoise_with_logp(self, prev_x, cur_x, z, t, generator=None, pred_ids=None, input_pred_ids=False):
        """Compute one denoising step and log-prob."""
        # prev_x/cur_x：latents；z：condition
        # timestep = torch.as_tensor(t, device=cur_x.device).expand(cur_x.shape[0])
        timestep = t

        # predict noise/mean via decoder
        if input_pred_ids==True:
            # 理论上来说这个应该只有推理的时候设成true，train的时候要同时对所有token denoise
            # 推理的时候设成true
            model_pred = self.image_decoder(cur_x, timestep, z, pred_ids)
        else:
            model_pred = self.image_decoder(cur_x, timestep, z)
        model_pred = self.image_encoder.patch_embed.unpatchify(model_pred)#[b,1024,16]->[b,4,64,64]#看起来ar部分和diff head部分都是在32*32 space上做的

        # diffusion step
        # 通过 scheduler.step 拿到 x0（pred_original_sample），不要把它当作后验均值
        # out = scheduler.step(mp, int(t_scalar), cx, generator=generator)  # 单次t，批量前向
        # prev_out[sel] = out.prev_sample
        # x0_out[sel]   = out.pred_original_sample
        if isinstance(self.sample_scheduler,FlowMatchEulerDiscreteScheduler):#flow matching scheduler
            # # out = self.sample_scheduler.step(model_pred, t, x, generator=generator)
            latents, logp, mean, std_dev_t = \
                sde_step_with_logprob(self.sample_scheduler,model_pred,timestep.float().cpu(),cur_x,prev_x,generator=generator)
            std_t = std_dev_t
        # ddpm scheduler:
        else:
            prev_out,x0=ddpm_step_grouped(self.sample_scheduler,model_pred,t,cur_x,generator=generator)
            # out = self.sample_scheduler.step(model_pred, t, cur_x, generator=generator)
            # # mean = getattr(out, "pred_original_sample", None)
            # # var = getattr(out, "variance", torch.zeros_like(cur_x))
            # x0 = getattr(out, "pred_original_sample", None)

            # 正确的后验 mean/var
            mean, var = posterior_mean_variance(self.sample_scheduler, cur_x, x0, timestep)
            # mean = ddpm_mean(self.sample_scheduler, t, cur_x, out.pred_original_sample)
            # std_t = torch.sqrt(getattr(out, "variance", torch.zeros_like(x)))  # depends on scheduler
            # var = self.sample_scheduler._get_variance(t,predicted_variance=None)

            eps = 1e-12#避免当t=1时算出logp=nan（此时var=0，即生成过程是确定的）
            var_safe = torch.where(var > 0, var, torch.full_like(var, eps))
            logp = -0.5 * (((prev_x - mean)**2) / var_safe + torch.log(2 * torch.pi * var_safe))
            std_t = torch.sqrt(var)

        return logp, mean, std_t
        

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict, return_latent_with_seed: bool=False):
        """Generate a batch of frames."""
        guidance_scaler = GuidanceScaler(**inputs)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None
        
        # --- 新增: 初始化本帧日志容器 ---
        if self.return_logp:
            step_logs = {"xs": [], "xs_next": [], "x_latent": [], "log_probs": [], "out_means": [], "prev_ids": [], "pred_ids": [], "final_diff_latents": []}
            
        all_ar_latents=None
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):#mar的64个timestep#总共forward 63次
            guidance_scaler.decay_guidance_scale((i + 1) / len(all_num_preds))
            # self.mask_embed是根据存储的self.mask把patch embed后的x加mask
            z = self.mask_embed(self.image_encoder.patch_embed(x))#patch_embed:torch.Size([5, 4, 64, 64])->torch.Size([5, 1024, 1024])
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)#生成随机mask for next step；pred_ids是next step要预测的位置
            pred_ids = guidance_scaler.expand(pred_ids)
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))#prev_ids是历史预测的tokens，pred_ids是当前预测的toknes，是一个index[b,l',1]，l'是当前step预测的token数
            
            if self.return_logp:
                step_logs["prev_ids"].append(prev_ids)#step i 已知的token
            
            z = self.image_encoder(guidance_scaler.expand(z), c, prev_ids, pos=pos)
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator)#b,c,h,w
            if self.return_logp:
                # return xs, xs_next, log_probs, out_means, self.image_encoder.patch_embed.patchify(x)
                # sample就是patchify后的xs[-1]
                # pred_ids=None确保latent不止包括pred_ids部分的，对生成结果没有影响，只是会更费算力

                if return_latent_with_seed:
                    z_=z.clone()
                    if all_ar_latents==None:
                        all_ar_latents=z_.chunk(2)[0].mul_(pred_mask)
                    else:
                        all_ar_latents=all_ar_latents.clone()
                        all_ar_latents.add_(z_.chunk(2)[0].mul_(pred_mask))

                xs,xs_next,log_probs,out_means,sample = self.denoise_with_logp(z, states["noise"], guidance_scaler.clone(), generator, pred_ids=None)
                step_logs["x_latent"].append(x.clone())#step i 输入的mask latent
                x.add_(self.image_encoder.patch_embed.unpatchify(sample*pred_mask))#pred_mask:[b,l,c]=5,1024,1

                # --- 新增: 记录日志（默认转到 CPU，避免显存占用过大）---
                step_logs["xs"].append(xs)#step i 得到的diff latent；注意denoise_with_logp输出的token是只包括当前pred_id部分的！
                step_logs["xs_next"].append(xs_next)#step i 输入的diff latent
                # step_logs["log_probs"].append(log_probs)
                step_logs["out_means"].append(out_means)#step i 预测的out means
                step_logs["pred_ids"].append(pred_ids)#step i 新预测的token
            else:
                sample = self.denoise(z, states["noise"], guidance_scaler.clone(), generator, pred_ids)
                x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))#pred_mask:[b,l,c]=5,1024,1

        if self.return_logp:
            step_logs["final_diff_latents"].append(x.clone())
            
        num_seed=3
        print('generation seed: ',num_seed)
        seed_list = [random.randint(0, 10000) for _ in range(num_seed)]
        sample_all_diff_latents_seed={"xs": [], "xs_next": [], "latents_seed":[]}
        # sample_all_diff_latents_seed = {"xs": [], "xs_next": [], "x_latent": [], "log_probs": [], "out_means": [], "prev_ids": [], "pred_ids": []}
        if return_latent_with_seed:
            for seed_ in seed_list:   # seed_list 长度 = 10
                torch.manual_seed(seed_)
                inp_noise = torch.randn_like(states["noise"])
                xs_seed, xs_seed_next, logps_seed, out_means_seed, sample_seed = self.denoise_with_logp(all_ar_latents.repeat(2, 1, 1),inp_noise,guidance_scaler.clone(),generator,pred_ids=None)
                sample_all_diff_latents = xs_seed[-1]
                sample_all_diff_latents_seed["latents_seed"].append(sample_all_diff_latents)
                sample_all_diff_latents_seed["xs"].append(xs_seed)
                sample_all_diff_latents_seed["xs_next"].append(xs_seed_next)


        # --- 新增: 把本帧日志挂到 states 上，供 generate_video 在外部读取 ---
        if self.return_logp:
            # 帧级别收集器：每次调用 generate_frame 就 append 一次
            if "frame_logs" not in states:
                states["frame_logs"] = []
            # 也可以把时间步存进去，便于对齐
            step_logs["t"] = states.get("t", None)
            states["frame_logs"].append(step_logs)

        if return_latent_with_seed:
            return all_ar_latents,sample_all_diff_latents_seed
            

    @torch.inference_mode()
    def generate_video(self, inputs: Dict,return_latent_with_seed: bool=False):
        """Generate a batch of videos."""
        guidance_scaler = GuidanceScaler(**inputs)
        max_latent_length = inputs.get("max_latent_length", 1)#max_latent_length=1表示生成的是图像
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        if self.image_pos_embed:#始终false  # RoPE.
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:  # Absolute PE, which will be deprecated in the future.
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length)#time维度上的pos embed，对于只生成一张图的时候=torch.Size([1, 1, 1024])
        inputs["c"] = guidance_scaler.expand_text(inputs["c"])
        self.video_encoder.enable_kvcache(max_latent_length > 1)
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            pos = time_pos[states["t"]] if time_pos else None#对于只生成图像的任务，states["t"]=0,time_pos和pos都是[]
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c
            c = guidance_scaler.expand(c, padding=self.mask_embed.bos_token)
            c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos)#不太懂为啥这里要先过一下video_encoder
            if not isinstance(self.video_encoder.mixer, torch.nn.Identity):
                states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
                states["*"] = states["*"] if states["t"] else states["c"]
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                if return_latent_with_seed:
                    # return all_ar_latents,sample_all_diff_latents_list
                    all_ar_latents,sample_all_diff_latents_seed=self.generate_frame(states, inputs, return_latent_with_seed=return_latent_with_seed)
                else:
                    self.generate_frame(states, inputs,return_latent_with_seed=return_latent_with_seed)
                latents.append(states["x"].clone())
        self.video_encoder.enable_kvcache(False)
        if return_latent_with_seed:
            return states,all_ar_latents,sample_all_diff_latents_seed
        else:
            return states

    def train_video(self, inputs):
        """Train a batch of videos."""
        # 3D temporal autoregressive modeling (TAM).
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1])
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1)
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        z = self.image_encoder(self.mask_embed(z), c.reshape(bs, -1, c.size(-1)), pos=pos)
        # 1D token-wise diffusion modeling (MLP).
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return self.get_losses(z, x, video_shape=video_shape)


    def single_forward_without_head(self, inputs):
        """Train a batch of videos."""
        # 3D temporal autoregressive modeling (TAM).#这部分可能在训练中可以舍弃
        # 可以去看generate_video，也是先整体forward了一次才逐帧生成
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None#从[b,c,h,w]->[b,c,t,h,w]
        prev_ids=inputs["prev_ids"] if "prev_ids" in inputs else None
        pred_ids=inputs["pred_ids"] if "pred_ids" in inputs else None
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)#latent_length=1,图像只有1帧
        c = self.video_encoder.patch_embed(inputs["x"])
        bov = gather_tensor(self.mask_embed.bos_token,inputs["x"].device).expand(bs, 1, c.size(-2), -1)
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None
        if self.image_pos_embed:#好像image_pos_embed是None
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        
        # # mask输入z
        pred_mask = (z.new_ones(z.shape[:-1] + (1,))).scatter_(1, prev_ids, 0)#对每行，在 prev_ids 指定的位置写 0，其他为 1，得到本轮已知的位置掩码。
        z_masked=z.mul(1 - pred_mask).add_(gather_tensor(self.mask_embed.mask_token,pred_mask.device) * pred_mask)#pred_mask为1对应部分会被换成self.mask_embed.mask_token
        
        z = self.image_encoder(z_masked, c.reshape(bs, -1, c.size(-1)), prev_ids=prev_ids,pos=pos)
        # 1D token-wise diffusion modeling (MLP).
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return z#self.get_losses(z, x, video_shape=video_shape)
    
    

    def forward(self, inputs,return_latent_with_seed:bool=False):
        """Define the computation performed at every call."""
        self.pipeline_preprocess(inputs)
        self.preprocess(inputs)
        if self.training:
            return self.train_video(inputs)
        inputs["latents"] = inputs.pop("latents", [])#初始时inputs["latents"]为空
        if self.return_logp:
            if return_latent_with_seed:
                states,all_ar_latents,sample_all_diff_latents_seed=self.generate_video(inputs,return_latent_with_seed=return_latent_with_seed)
                return {"x": torch.stack(inputs["latents"], dim=2),"frame_logs":states["frame_logs"]},all_ar_latents,sample_all_diff_latents_seed
            else:
                states=self.generate_video(inputs,return_latent_with_seed=return_latent_with_seed)
                return {"x": torch.stack(inputs["latents"], dim=2),"frame_logs":states["frame_logs"]}
        else:
            self.generate_video(inputs)
            return {"x": torch.stack(inputs["latents"], dim=2)}
