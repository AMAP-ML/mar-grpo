# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Two Forward Passes
'''

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Tuple, Dict
from PIL import Image
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
# import matplotlib.pyplot as plt
from transformers.cache_utils import Cache, StaticCache,DynamicCache
import random

import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
    is_apex_available,
)
# from accelerate.utils import unwrap_model
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from accelerate.utils import wait_for_everyone
import os, gc, torch

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler
import math
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from diffnext.pipelines import NOVAPipeline
from diffnext.models.text_encoders.phi import PhiEncoderModel
from diffnext.models.transformers.transformer_nova import NOVATransformer3DModel
from diffnext.pipelines.pipeline_utils import NOVAPipelineOutput, PipelineMixin
from diffnext.generate_image_with_logp import generate_image_with_logp,single_forward_without_head

from transformers import CodeGenTokenizerFast
from diffusers import DDPMScheduler, AutoencoderKL
from utils.reward_hps import HPSv2
from utils.reward_geneval import Geneval_score
from diffusers.utils.torch_utils import randn_tensor
from utils.reward_ocr import OcrScorer
from utils.reward_git import GIT
from utils.reward_gdino import GDino
from utils.reward_clip import Clip
# from utils.reward_unifiedreward import UnifiedReward
from utils.reward_orm import ORM
import shutil
import copy
import re
from PIL import Image

from transformers.utils import (
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names


if is_apex_available():
    from apex import amp

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        DistributedType,
    )
    
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DiffEMAUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        if state.global_step > 0:
            trainer.update_diff_ema()


class NOVAT2IR1Trainer_v1(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, NOVATransformer3DModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        # processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
    ):
        # Args
        self.ema_advantage=False#目前来看，advantage是针对特定的question计算的，不存在不同question的影响
        self.reward_smooth=args.reward_smooth#False
        if self.reward_smooth:
            self.sim_cos_thresh=0.2
        self.dynamic_sample=False#sample时使用不同的参数
        self.loss_reweight=False#废弃的
        self.entropy_reward=args.entropy_reward#在生成图像的reward加一个entropy bonus
        self.entropy_reward_v2=False
        self.mask_pos_neg=True#mask掉reward std=0的prompt
        self.kl_reweight=args.kl_reweight#kl权重随样本有差异
        self.kl_clamp=False
        if self.kl_clamp:
            self.kl_clamp_topk=2000
        self.kl_dy_weight=False#废弃的
        self.max_steps=args.max_steps
        if self.kl_dy_weight:
            self.init_learning_rate=args.learning_rate
        self.add_noise=args.add_noise
        self.progress_learning=args.progress_learning#在靠前的step中mask一些质量比较差的样本，避免多样性下降
        self.update_ref=args.update_ref
        self.output_dir=args.output_dir
        self.wokl=False#在reward std=0时是否应该去掉这些samples的kl loss
        self.fix_head=args.fix_head
        self.fix_ar=args.fix_ar
        self.use_critic_token=args.use_critic_token
        self.critic_for_kl=args.critic_for_kl
        self.top_mask_percent=args.top_mask_percent
        self.all_diff_timesteps=args.all_diff_timesteps
        self.select_mask_steps=False
        self.normalize_multi_rewards=False
        self.use_latent_sim_coef=args.use_latent_sim_coef
        self.latent_sim_thresh=args.latent_sim_thresh
        print("latent sim thresh: ", self.latent_sim_thresh)
        self.save_config()
        print('using vanilla grpo...')
        print('fix head: ',self.fix_head, ' using critic token: ',self.use_critic_token, \
            ' all_diff_timesteps: ',self.all_diff_timesteps, ' top_mask_percent: ',self.top_mask_percent)
        
        # # Args baseline
        # self.ema_advantage=False#目前来看，advantage是针对特定的question计算的，不存在不同question的影响
        # self.reward_smooth=False
        # if self.reward_smooth:
        #     self.sim_cos_thresh=0.7
        # self.dynamic_sample=False
        # self.loss_reweight=False
        # self.entropy_reward=True
        # self.kl_reweight=False
        # self.kl_clamp=False
        # if self.kl_clamp:
        #     self.kl_clamp_topk=2000

        print(f'dynamic_sample: {self.dynamic_sample}, reward smooth: {self.reward_smooth}')
        if self.loss_reweight:#存储不同reward funcs的ema mean/std，根据这个把不同的reward归一化
            self.momentum=0.95;self.reward_func_std_ema=None;self.reward_func_mean_ema=None
        
        if self.ema_advantage:
            self.adv_mean = 0.0  # moving average of reward or advantage
            self.adv_std = 1.0   # moving variance
            self.momentum = 0.95  # decay factor (类似 batchnorm 的 momentum)
        self.semantic_cot=args.semantic_cot
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        self.model_id = model
        # self.process_class=NextStepPipeline(model_name_or_path=self.model_id,vae_name_or_path=vae_name_or_path)
        self.process_class = NOVAPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,device_map=None,trust_remote_code=True)
        model = self.process_class.transformer
        
        # prepare text tokenizer
        self.tokenizer = self.process_class.tokenizer
        self.scheduler = self.process_class.scheduler
        
        self.guidance_scale=5
        self.num_inference_steps=64
        self.motion_flow=5
        self.max_latent_length=1 #>1 for video generation
        self.image_gen_temperature = 1
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        mask_ratios = np.cos(0.5 * np.pi * np.arange(self.num_inference_steps + 1) / self.num_inference_steps)
        num_patches = int(np.prod(model.config.image_base_size))
        mask_length = np.round(mask_ratios * num_patches).astype("int64")
        self.num_preds = mask_length[:-1] - mask_length[1:]#mask model每次预测多少token
        # self.train_diff_timesteps = self.scheduler.timesteps#diff model预测多少步

        # # try gradient checkpointing#暂时还不支持gradient checkpoint
        # model.language_model.config.use_cache = False
        # model.language_model.gradient_checkpointing_enable()

        # Reference model
        if is_deepspeed_zero3_enabled() and args.beta != 0:
            self.ref_model = copy.deepcopy(model)
            # model = self.process_class.transformer
            self.ref_model.sample_scheduler = self.scheduler
            if model.text_embed:
                self.ref_model.text_embed.encoders = [self.process_class.tokenizer, self.process_class.text_encoder]
        elif peft_config is None and args.beta != 0:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        
        # prepare vae
        self.vae = self.process_class.vae
        self.vae.eval()
        
        # freeze all vision encoders
        for name, param in model.named_parameters():
            # if ("tokenizer" in name) or ("vae" in name): # choose whatever you like here
                param.requires_grad = True
        for name, param in self.vae.named_parameters():
        # if ("tokenizer" in name) or ("vae" in name): # choose whatever you like here
            param.requires_grad = False
        if self.fix_head:
            for name, param in model.image_decoder.named_parameters():
                param.requires_grad = False
        if self.fix_ar:
            for name, param in model.named_parameters():
                if not "image_decoder" in name: # choose whatever you like here
                    param.requires_grad = False

        self.reward_funcs_name=[item for item in reward_funcs]
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'hps' in reward_func:
                reward_funcs[i] = HPSv2(args)
            elif isinstance(reward_func, str) and 'geneval' in reward_func:
                reward_funcs[i] = Geneval_score(args)
            elif isinstance(reward_func, str) and 'git' in reward_func:
                reward_funcs[i] = GIT(args)
            elif isinstance(reward_func, str) and 'clip' in reward_func:
                reward_funcs[i] = Clip(args)                
            elif isinstance(reward_func, str) and 'gdino' in reward_func:
                reward_funcs[i] = GDino(args)
            elif isinstance(reward_func, str) and 'orm' in reward_func:
                reward_funcs[i] = ORM(args)
            elif isinstance(reward_func, str) and 'ocr' in reward_func:
                reward_funcs[i] = OcrScorer(args)
            else:
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.new_generations_image = args.new_generations_image#num_generations和new_generations_image分别是啥意思
        self.beta = args.beta

        # Initialize the metrics
        self._metrics = defaultdict(list)

        self.small_lr_for_head=args.small_lr_for_head
        self.ema_for_head=args.ema_for_head

        if self.small_lr_for_head:
            self.lr_ar=args.learning_rate
            self.lr_diff=args.learning_rate/2

        if self.ema_for_head:
            self.ema_decay = 0.99
            self.diff_head = model.image_decoder
            self.ema_diff_head = copy.deepcopy(self.diff_head)
            for p in self.ema_diff_head.parameters():
                p.requires_grad_(False)
            self.add_callback(DiffEMAUpdateCallback())


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.beta != 0:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

        self.vae.to(self.accelerator.device)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, HPSv2) or isinstance(reward_func, Clip) or isinstance(reward_func, GDino) \
                or isinstance(reward_func, GIT) or isinstance(reward_func, Geneval_score)\
                    or isinstance(reward_func, OcrScorer):
                reward_func.load_to_device(self.accelerator.device)
            elif isinstance(reward_func, ORM):
                reward_func.load_to_device(self.accelerator.device)
                reward_func.accelerator = self.accelerator
                if self.is_deepspeed_enabled:   
                    reward_func.model = prepare_deepspeed(reward_func.model, self.accelerator)
                else:
                    reward_func.model = self.accelerator.prepare_model(reward_func.model, evaluation_mode=True)
        
        self.image_token_num_per_image = args.image_token_num_per_image
        # self.cfg_weight = args.cfg_weight

        # 不知道为啥，zero1/2的时候，self.process_class.text_encoder加载到cpu上了
        if not is_deepspeed_zero3_enabled():
            self.process_class.to(self.accelerator.device)


    def save_config(self):
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = os.path.join(self.output_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write("=== Configurations ===\n")
            f.write(f"ema_advantage: {self.ema_advantage}\n")
            f.write(f"reward_smooth: {self.reward_smooth}\n")
            if self.reward_smooth:
                f.write(f"  sim_cos_thresh: {self.sim_cos_thresh}\n")
            f.write(f"dynamic_sample: {self.dynamic_sample}\n")
            f.write(f"loss_reweight: {self.loss_reweight}\n")
            f.write(f"entropy_reward: {self.entropy_reward}\n")
            f.write(f"entropy_reward_v2: {self.entropy_reward_v2}\n")
            f.write(f"mask_pos_neg: {self.mask_pos_neg}\n")
            f.write(f"kl_reweight: {self.kl_reweight}\n")
            f.write(f"kl_clamp: {self.kl_clamp}\n")
            if self.kl_clamp:
                f.write(f"  kl_clamp_topk: {self.kl_clamp_topk}\n")
            f.write(f"kl_dy_weight: {self.kl_dy_weight}\n")
            if self.kl_dy_weight:
                f.write(f"  init_learning_rate: {self.init_learning_rate}\n")
            f.write(f"max_steps: {self.max_steps}\n")
            f.write(f"add_noise: {self.add_noise}\n")
            f.write(f"progress_learning: {self.progress_learning}\n")
            f.write(f"update_ref: {self.update_ref}\n")
            f.write(f"output_dir: {self.output_dir}\n")
            f.write(f"wokl: {self.wokl}\n")


    # def create_optimizer(self):
    #     opt_model = self.model_wrapped if hasattr(self, "model_wrapped") and self.model_wrapped is not None else self.model

    #     if self.optimizer is not None:
    #         return self.optimizer

    #     # ---- 你自己的 lr 配置 ----
    #     lr_ar = getattr(self, "lr_ar", self.args.learning_rate)   # 例如 1e-6
    #     lr_diff = getattr(self, "lr_diff", self.args.learning_rate)  # 例如 1e-7

    #     wd = self.args.weight_decay

    #     # ---- 识别哪些参数属于 diff head（你按自己项目改这个规则）----
    #     # 最简单：按 name 关键字
    #     diff_keywords = getattr(self.args, "diff_param_name_keywords", ["image_decoder"])
    #     def is_diff_param(name: str) -> bool:
    #         return any(k in name for k in diff_keywords)

    #     # ---- decay / no-decay 名单（HF 标准写法）----
    #     decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm, nn.Embedding, nn.RMSNorm])
    #     decay_parameters = set(decay_parameters)

    #     # ---- 收集参数 ----
    #     ar_decay, ar_nodecay, diff_decay, diff_nodecay = [], [], [], []

    #     for n, p in opt_model.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         is_decay = (n in decay_parameters) and (p.ndim >= 2)
    #         if is_diff_param(n):
    #             (diff_decay if is_decay else diff_nodecay).append(p)
    #         else:
    #             (ar_decay if is_decay else ar_nodecay).append(p)

    #     optimizer_grouped_parameters = []
    #     if ar_decay:
    #         optimizer_grouped_parameters.append({"params": ar_decay, "lr": lr_ar, "weight_decay": wd})
    #     if ar_nodecay:
    #         optimizer_grouped_parameters.append({"params": ar_nodecay, "lr": lr_ar, "weight_decay": 0.0})
    #     if diff_decay:
    #         optimizer_grouped_parameters.append({"params": diff_decay, "lr": lr_diff, "weight_decay": wd})
    #     if diff_nodecay:
    #         optimizer_grouped_parameters.append({"params": diff_nodecay, "lr": lr_diff, "weight_decay": 0.0})

    #     # ---- 用 HF 内置选择 optimizer class / kwargs（支持 AdamW / fused 等）----
    #     optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

    #     # ⚠️ 关键：不要让 optimizer_kwargs 里的 params/model/optimizer_dict 覆盖你的分组
    #     optimizer_kwargs.pop("params", None)
    #     optimizer_kwargs.pop("model", None)
    #     optimizer_kwargs.pop("optimizer_dict", None)

    #     self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #     return self.optimizer


    def setup_diff_ema(self, ema_decay=0.999):
        self.ema_decay = ema_decay
        self.diff_head = self.model.diff_head
        self.ema_diff_head = copy.deepcopy(self.diff_head)
        for p in self.ema_diff_head.parameters():
            p.requires_grad_(False)


    @torch.no_grad()
    def update_diff_ema(self):
        for p, p_ema in zip(
            self.diff_head.parameters(),
            self.ema_diff_head.parameters()
        ):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # 根据transformers trainer的设置，inputs是来自dataset的，直接输入
        return inputs
    

    def maybe_update_ref_model(self, global_step: int, ref_schedule: dict[int, int] | None = None):
        """
        根据映射在特定 global_step 更新 ref_model。
        ref_schedule 形如 {600: 400, 1000: 800}：
        key  = 当前 global_step
        value= 要加载为 ref_model 的 checkpoint 步数
        """
        if not ref_schedule or global_step == 0:
            return

        ref_step = ref_schedule.get(global_step, None)
        if ref_step is None:
            return  # 本 step 不更新

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{ref_step}")
        # ckpt_dir='$USER_ROOT/workspace_t2i-r1_train_ckpts/outputs_geneval_v3/t2i-r1-my-geneval-lr5e-6-test-datav3/checkpoint-700'
        if not os.path.isdir(ckpt_dir):
            print(f"[Warning] Checkpoint {ckpt_dir} does not exist, skip updating ref_model.")
            return

        # 分布式同步（进入更新前）
        try:
            wait_for_everyone()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(f"[Info] Updating ref_model at global_step={global_step} from {ckpt_dir} ...")

        # 需要每个 rank 都加载，避免死锁
        unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)

        # 与当前训练模型保持一致的 dtype
        target_dtype = next(self.model.parameters()).dtype
        target_device = self.model.device

        # 低内存加载 -> 取 state_dict -> 立刻释放临时模型，避免占峰
        tmp = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            torch_dtype=target_dtype,
            # low_cpu_mem_usage=True
        )
        state_dict = tmp.state_dict()
        del tmp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            missing, unexpected = unwrapped_ref.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Info] ref_model.load_state_dict(strict=False), "
                    f"missing={len(missing)}, unexpected={len(unexpected)}")

        # 移动到训练设备并保持 dtype
        unwrapped_ref.to(device=target_device, dtype=target_dtype)

        # 如使用 DeepSpeed ZeRO-3，需重新封装
        if getattr(self, "is_deepspeed_enabled", False):
            self.ref_model = prepare_deepspeed(unwrapped_ref, self.accelerator)
        else:
            self.ref_model = self.accelerator.prepare_model(unwrapped_ref, evaluation_mode=True)

        # 分布式同步（更新完成后）
        try:
            wait_for_everyone()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        print(f"[Info] ref_model updated (ref_step={ref_step}) at global_step={global_step}")



    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # ❗️如果你在 compute_loss 里已手动 backward，就不要再走 smp_forward_backward
        # 若你确实用到了 SageMaker MP，需要在 compute_loss 里改为 smp-compatible 的 backward。
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # ⬇️ compute_loss 内部自己 backward；这里仅取一个日志值返回
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # 释放 inputs，行为与原版一致
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        # 仅用于日志展示的“无梯度标量”
        # 不要再做 /GAS、不要再做 accelerator.backward
        # 多卡时如需可视化对齐，可以对“已detach”的 loss 做 mean
        log_loss = loss
        if isinstance(log_loss, float):
            log_loss = torch.tensor(log_loss, device=self.args.device, dtype=torch.float32)
        else:
            log_loss = torch.as_tensor(log_loss, device=self.args.device)
        if self.args.n_gpu > 1:
            log_loss = log_loss.detach().mean()
        else:
            log_loss = log_loss.detach()

        return log_loss



    def loss_backward(self, loss: torch.Tensor, *, retain_graph: bool = False,
                    divide_by_gas: bool = True, mean_over_devices: bool = True) -> torch.Tensor:
        """
        在 compute_loss 内部调用，用于手动 backward。
        - retain_graph: 你在同一 forward 内需要多次 backward 时设 True（比如多 diffusion steps）
        - divide_by_gas: 是否按 gradient_accumulation_steps 缩放
        - mean_over_devices: 多卡时是否先对 loss 做 mean 与 Trainer 行为一致
        返回：detach 后的标量（仅用于日志）
        """
        l = loss

        # 多 GPU 时与 Trainer 保持一致的 mean（避免不同卡上标量尺度不同）
        if mean_over_devices and getattr(self.args, "n_gpu", 1) > 1:
            l = l.mean()

        # 与 Trainer 默认缩放保持一致（compute_loss_func=None 时 Trainer 会在 training_step 里 / GAS）
        if divide_by_gas and not getattr(self, "compute_loss_func", None) and not getattr(self, "model_accepts_loss_kwargs", False):
            l = l / max(self.args.gradient_accumulation_steps, 1)

        # DeepSpeed 特殊参数：不要按 GAS 再做一次内部缩放
        kwargs = {"retain_graph": retain_graph}
        if getattr(self.accelerator, "distributed_type", None) == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        # AMP(Apex) 或 Accelerate 统一 backward
        if getattr(self, "use_apex", False):
            with amp.scale_loss(l, self.optimizer) as scaled:
                scaled.backward(retain_graph=retain_graph)
        else:
            self.accelerator.backward(l, **kwargs)

        # 返回日志值（不带图）
        return l.detach()


    def compute_loss(self, model:NOVATransformer3DModel, inputs, return_outputs=False, num_items_in_batch=None):
        # compute_loss的input就是直接来自_prepare_inputs的输出
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        
        prompts = [x["prompt"] for x in inputs]#[{'content': 'You are asked to generate an image based on this prompt: "a tall skyscraper and a sho...sualization directly without explanation: ', 'role': 'User'}, {'content': '', 'role': 'Assistant'}]
        
        loss_dict = {}
        device = self.accelerator.device
        if self.semantic_cot:
            pass
                
        else:
            model.eval();  torch.set_grad_enabled(False)
            loss_dict['semantic-cot']=None
            # prompts = [x["prompt"] for x in inputs]
            image_gen_prompt_list = []#每个prompt有num_generations个扩充
            for i in range(len(inputs)):#ok，前面的semantic-cot看起来像是对prompt的cot扩充，image_gen_prompt_list就是最后扩充完的cot：image_gen_prompt_list# ['User: a tall skyscraper and a short traffic light.  I have generated the following vi...plicitly stated in the prompt.\n\nAssistant:', 'User: a tall skyscraper and a short traffic light.  Below is a visualization of the p...er and a short traffic light."\n\nAssistant:', 'User: a tall skyscraper and a short traffic light.  A tall skyscraper and a short tra...n their heights and proximity.\n\nAssistant:']
                raw_prompt = inputs[i // self.num_generations]['raw_prompt']
                print('rank %s iter %d raw prompt: '%(dist.get_rank(),self.state.global_step),raw_prompt)
                image_gen_prompt_list.extend([raw_prompt]*self.new_generations_image)
        
        # Generate the image tokens
        with torch.no_grad():
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # 5. LLM prefill
                with torch.inference_mode():
                    inference_variables_to_parse = generate_image_with_logp(self.process_class,image_gen_prompt_list,unwrapped_model,self.vae,
                    return_latent_with_seed=self.use_critic_token)
                    if self.use_critic_token:
                        images,frame_logs,prompt_embed,all_ar_latents,sample_all_diff_latents_seed = inference_variables_to_parse
                    else:
                        images,frame_logs,prompt_embed = inference_variables_to_parse
                    images=images["images"]#list
                    frame_logs=frame_logs[0]#长度始终为1
                    # inputs_embeds, attention_mask, _ = prepare_text_embeds(self.process_class, image_gen_prompt_list)
                    # images, sampled_images, log_prob, out_mean, total_latents, timesteps = \
                    #     pipeline_with_logprob(self.process_class, unwrapped_model, attention_mask=attention_mask, vae=self.vae, cfg=self.cfg,
                    #                         cfg_schedule=self.cfg_schedule, inputs_embeds=inputs_embeds, num_images_per_caption=self.num_generations)
            
        dist.barrier()
        # 指定保存目录
        save_dir = os.path.join(os.path.dirname(__file__), 'output_images_debug1')
        os.makedirs(save_dir, exist_ok=True)
        # 保存每张图像
        for idx, img in enumerate(images):
            save_path = os.path.join(save_dir, f"{dist.get_rank()}_{idx}.png")
            img.save(save_path)
        
        # frame_logs["xs"]；frame_logs["out_means"]；frame_logs["pred_ids"]都是list，长度为timestep数
        # frame_logs["xs"][0].shape：torch.Size([25, 5, 4, 64, 64]) #[diff_step,batch,c,h,w];diff_step有25个
        # frame_logs["out_means"][0].shape：torch.Size([25, 5, 4, 64, 64])
        # frame_logs["prev_ids"][0].shape：torch.Size([10, L, 1]):
        samples = {
            "prev_latents": frame_logs["xs"],##step i 得到的latent
            "latents": frame_logs["xs_next"],##step i 输入的latent
            "x_latent": frame_logs["x_latent"],
            "out_mean": frame_logs["out_means"],
            "prev_ids": frame_logs["prev_ids"],
            "pred_ids": frame_logs["pred_ids"],
            "final_diff_latents": frame_logs["final_diff_latents"],
            "pil_images": images,
            "prompt_embed": prompt_embed,

        }

        # 选择部分mask step和diff step：
        steps=[s for s in self.num_preds if s > 0]
        if self.select_mask_steps:
            selected_mask_steps = random.sample(range(5,30),k=min(12, len(steps)))#从5,30中随机选择12个step
        else:
            selected_mask_steps = random.sample(range(len(steps)),k=min(12, len(steps)))

        
        # for idx_ in range(len(samples["latents"])):
        #     # cur_diff_latent=samples["prev_latents"][i][-1]#当前mask step的最后一个diff latent
        #     # final_diff_latent=samples["prev_latents"][-1][-1]#最后mask step的最后一个diff latent
        #     if (idx_ in selected_mask_steps) or (idx_ == len(samples["latents"])-1):
        #         samples["prev_latents"][idx_]=samples["prev_latents"][idx_]
        #         samples["latents"][idx_]=samples["latents"][idx_]
        #     else:
        #         pl = samples["prev_latents"][idx_]
        #         cl = samples["latents"][idx_]
        #         samples["prev_latents"][idx_] = None
        #         samples["latents"][idx_] = None
        #         del pl, cl

        # Compute the rewards
        prompts = [input["raw_prompt"] for input in inputs for _ in range(self.num_generations) for __ in range(self.new_generations_image)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        with torch.no_grad():
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, PreTrainedModel):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    if isinstance(reward_func,Geneval_score):
                        metadatas = [input["metadata"] for input in inputs for _ in range(self.num_generations) for __ in range(self.new_generations_image)]
                        scores,rewards_,strict_rewards,grouped_rewards,grouped_strict_rewards = \
                            reward_func(images=images, prompts=prompts, metadatas=metadatas)
                        rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
                    elif isinstance(reward_func,OcrScorer):
                        metadatas = [input["metadata"] for input in inputs for _ in range(self.num_generations) for __ in range(self.new_generations_image)]
                        scores = \
                            reward_func(images=images, prompts=prompts, metadatas=metadatas)
                        rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
                    else:
                        # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                        reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                        for key in reward_kwargs:
                            for example in inputs:
                                # Repeat each value in the column for `num_generations` times
                                reward_kwargs[key].extend([example[key]] * self.num_generations * self.new_generations_image)#reward_kwargs把输入的list按照key整合成一个dict
                        output_reward_func = reward_func(prompts=prompts, images=images, **reward_kwargs)
                        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)#new_generations_image,num_rewards
        print('rank %s, iter %d'%(dist.get_rank(),self.state.global_step),rewards_per_func.squeeze())
            

        if 'geneval' in self.reward_funcs_name or 'ocr' in self.reward_funcs_name:#要清理掉无效的geneval data
            # rewards_per_func: [b, l]
            # step 1: 找出每个 reward function（列）是否是常数
            std_per_func = rewards_per_func.std(dim=0)  # [l]
            # valid_mask = std_per_func > 1e-6           # [l] -> True 表示该 reward 有效
            if self.mask_pos_neg==True:
                valid_mask = std_per_func > 1e-6           # [l] -> True 表示该 reward 有效
            else:
                valid_mask = torch.ones_like(std_per_func).bool()

            # step 2: 只选取有效 reward functions 进行计算
            valid_rewards = rewards_per_func[:, valid_mask]  # [b, l'] where l' <= l

            # step 3: 按照 grouped 方式计算每个 group 的 reward 总和
            # sum over valid reward functions
            rewards = valid_rewards.sum(dim=1)  # [b]

            # step 4: reshape 成 [num_prompts, num_generations_per_prompt] 再计算 mean/std
            rewards_grouped = rewards.view(-1, self.num_generations * self.new_generations_image)  # [g, n]
            mean_grouped_rewards = rewards_grouped.mean(dim=1)  # [g]
            std_grouped_rewards = rewards_grouped.std(dim=1)    # [g]

            # step 5: 还原回原始维度，方便逐个样本归一化
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)  # [b]
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)    # [b]

            # step 6: advantage 计算
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)#在bf16能容忍的区别内，是0，稍微超过一点点就会被归一化到标准分布

            # step 7: 如果 valid_mask 全是 False，也就是没有任何有效的 reward，则 advantages 全设为 0
            if valid_rewards.shape[1] == 0:
                #这里目前只支持仅有geneval reward的，当有多个reward时，没用的reward默认0，而且后续的计算不同rank的reward也没有去掉这部分为0的reward
                # advantages = torch.zeros_like(rewards)
                rewards=torch.zeros_like(rewards)-1#这个rewards-1主要是为了把值置为-1

                    #ok，我大概懂了，因为我设置的entropy reward在0附近基本没什么差，现在又把除以std这个给去掉了，所以比较难把model给拉回来
        
            
        else:
            if self.normalize_multi_rewards:
                # rewards_per_func: [b, l]
                min_reward = rewards_per_func.min(dim=0, keepdim=True).values
                max_reward = rewards_per_func.max(dim=0, keepdim=True).values

                # range
                reward_range = max_reward - min_reward

                # 0-1 normalization
                rewards_norm = (rewards_per_func - min_reward) / (reward_range + 1e-8)

                # if all rewards in a column are equal, set to 1
                equal_mask = reward_range < 1e-8
                rewards_norm = torch.where(equal_mask, torch.zeros_like(rewards_norm), rewards_norm)
                rewards = rewards_norm.sum(dim=1)#直接把每个sample对应的不同funcs的reward加起来
            else:
                rewards = rewards_per_func.sum(dim=1)#直接把每个sample对应的不同funcs的reward加起来
                
            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).mean(dim=1)#呃看起来是每个prompt产生的图自己算mean和std
            std_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
            
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
            
        
        if self.use_critic_token:
            # list，每个元素是[b,16,64,64]
            sample_all_diff_latents_std=torch.stack(sample_all_diff_latents_seed["latents_seed"], dim=0).std(dim=0, unbiased=False).mean(dim=1,keepdim=True)

            def select_top_tokens_via_std(std: torch.Tensor, top_percent: float, rand_percent: float) -> torch.Tensor:
                B, C, H, W = std.shape
                x = std.view(B, -1)
                N = x.size(1)

                # ---- std-based top tokens ----
                k_std = max(1, int(N * top_percent))
                idx = x.topk(k_std, dim=1, largest=True).indices
                mask_std = torch.zeros_like(x, dtype=torch.bfloat16).scatter_(1, idx, 1.0)

                # ---- random tokens ----
                k_rand = max(1, int(N * rand_percent))
                rand_idx = torch.rand(B, N, device=x.device).topk(k_rand, dim=1).indices
                mask_rand = torch.zeros_like(x, dtype=torch.bfloat16).scatter_(1, rand_idx, 1.0)

                mask = torch.clamp(mask_std + mask_rand, min=0, max=1.0)
                return mask.view(B, C, H, W)

            critic_token_mask=select_top_tokens_via_std(sample_all_diff_latents_std,self.top_mask_percent,0.0)#先选择top 30%token优化
            # critic_token_mask=1-critic_token_mask#ablation:剩下的70%token

            for idx in range(sample_all_diff_latents_std.shape[0]):
                # save_path = os.path.join(save_dir, f"{dist.get_rank()}_{idx}.png")
                save_root=os.path.join(os.path.dirname(__file__), 'output_images_debug1')
                Image.fromarray(((lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) * 255)(sample_all_diff_latents_std[idx][0].float())).byte().cpu().numpy()).save(f"{save_root}/diff_std_rank{dist.get_rank()}_idx{idx}.png")
                Image.fromarray(((lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) * 255)(critic_token_mask[idx][0].float())).byte().cpu().numpy()).save(f"{save_root}/mask_std_rank{dist.get_rank()}_idx{idx}.png")

        
        print('rank %s, iter %d advantages:'%(dist.get_rank(),self.state.global_step),advantages.squeeze())

        model.train();  torch.set_grad_enabled(True)
        # Get the logp for all the generated tokens
        # text+image

        # 思路：在给定的任意timestep下，给定mask和输入（生成过程中每个step的latent，增大下一个latent为positive sample对应latent的概率），
        # 呃不要把每个timestep当作一个分支去看，要把所有timestep的samples当作一组数据，也就是根据给定的输入和mask，对应的输出的reward是正还是负（来自完整图片的reward）
        per_timestep_loss=[];per_timestep_kl=[]
        
        steps=[s for s in self.num_preds if s > 0]
        # if self.select_mask_steps:
        #     selected_mask_steps = random.sample(range(5,30),k=min(12, len(steps)))#从5,30中随机选择12个step
        # else:
        #     selected_mask_steps = random.sample(range(len(steps)),k=min(12, len(steps)))
        for i in selected_mask_steps:#总共63步
            train_inputs_prev_ids=samples["prev_ids"][i].chunk(2)[0].clone()
            train_inputs_pred_ids=samples["pred_ids"][i].chunk(2)[0].clone()
            train_inputs_x=samples["x_latent"][i].clone()
            train_inputs_prompt_embed=samples["prompt_embed"].chunk(2)[0].clone()
            print('rank: ',self.accelerator.device, ' step: ',i)
            train_inputs={
                "prev_ids":train_inputs_prev_ids,#step i 已知的token
                "pred_ids":train_inputs_pred_ids,#step i 新预测的token
                "x":train_inputs_x,#[b,4,64,64]#当前timestep输入的latent
                "prompt":train_inputs_prompt_embed#呃不知道cur和ref prompt embed设置成相同的会不会有问题
                          }
            # denoised_latents_prev=samples["latents"][i-1][-1]#上一个timestep预测的latent
            ar_latents=single_forward_without_head(model,train_inputs)
            with torch.no_grad():
                train_inputs_ref={
                    "prev_ids":train_inputs_prev_ids,#step i 已知的token
                    "pred_ids":train_inputs_pred_ids,#step i 新预测的token
                    "x":train_inputs_x,#[b,4,64,64]#当前timestep输入的latent
                    "prompt":train_inputs_prompt_embed#呃不知道cur和ref prompt embed设置成相同的会不会有问题
                            }
                if self.beta != 0:
                    ref_ar_latents=single_forward_without_head(self.ref_model,train_inputs_ref)
            
                
            # # mask输入z#只算当前token未知区域的loss
            dummy_tensor=model.image_encoder.patch_embed.patchify(frame_logs['out_means'][0][0])
            pred_mask = (dummy_tensor.new_ones(dummy_tensor.shape[:-1] + (16,))).scatter_(1, train_inputs_prev_ids.expand(-1, -1, 16), 0)#对每行，在 prev_ids 指定的位置写 0，其他为 1，得到本轮已知的位置掩码。
            pred_mask = model.image_encoder.patch_embed.unpatchify(pred_mask)
            completion_mask=pred_mask#不算prev_ids对应的区域，只算未知的

            pred_mask_next = (dummy_tensor.new_zeros(dummy_tensor.shape[:-1] + (16,))).scatter_(1, train_inputs_pred_ids.expand(-1, -1, 16), 1)#对每行，在 prev_ids 指定的位置写 0，其他为 1，得到本轮已知的位置掩码。
            pred_mask_next = model.image_encoder.patch_embed.unpatchify(pred_mask_next)
            completion_mask_next=pred_mask_next#不算prev_ids对应的区域，只算未知的
            
            total_policy_loss=0
            total_kl=0
            total_loss=0

            diff_timesteps=self.scheduler.timesteps.to(dummy_tensor.device)
            if self.all_diff_timesteps:
                k=diff_timesteps.numel()#从25个diff step中随机选择10个step
            else:
                k=min(10,diff_timesteps.numel())#从25个diff step中随机选择10个step
            selected_diff_steps_idx = torch.randperm(diff_timesteps.numel(), device=dummy_tensor.device)[:k]
            selected_diff_steps = diff_timesteps.index_select(0,selected_diff_steps_idx)
            
            prev_latents_k=samples["prev_latents"][i].index_select(0,selected_diff_steps_idx)#samples["prev_latents"][i]:[diff_step,b,c,h,w]#step i 得到的latent
            cur_latents_k=samples["latents"][i].index_select(0,selected_diff_steps_idx)#step i 输入的latent
            
            B = cur_latents_k.shape[1]
            
            prev_latents_flat = prev_latents_k.reshape(k*B, *prev_latents_k.shape[2:])#k,b,c,h,w
            cur_latents_flat  = cur_latents_k.reshape(k*B,  *cur_latents_k.shape[2:])
            if self.use_critic_token:
                critic_token_mask_flat = critic_token_mask.unsqueeze(0).expand(k, -1, -1, -1, -1).flatten(0, 1)
            
            ar_latents_flat = ar_latents.unsqueeze(0).expand(k, *ar_latents.shape)           # [k,B,...]
            ar_latents_flat = ar_latents_flat.reshape(k*B, *ar_latents.shape[1:])                     # [k*B,...]
            if self.beta != 0:
                ref_ar_latents_flat = ref_ar_latents.unsqueeze(0).expand(k, *ar_latents.shape)           # [k,B,...]
                ref_ar_latents_flat = ref_ar_latents_flat.reshape(k*B, *ar_latents.shape[1:])                     # [k*B,...]
            
            t_flat = selected_diff_steps.unsqueeze(1).expand(k,B).reshape(k*B)                                # [k*B]
            completion_mask_flat = completion_mask.unsqueeze(0).expand(k, *completion_mask.shape).reshape(k*B, *completion_mask.shape[1:])
            completion_mask_flat_next = completion_mask_next.unsqueeze(0).expand(k, *completion_mask.shape).reshape(k*B, *completion_mask.shape[1:])

            advantages_flat = advantages.view(B,1,1,1).unsqueeze(0).expand(k,B,1,1,1).reshape(k*B,1,1,1)
            
            # pred_ids: [B, K] 或 [B, K, 1] -> [k*B, K{,1}]
            pid = train_inputs_pred_ids
            if pid.dim() == 3 and pid.size(-1) == 1:
                pid = pid.squeeze(-1)
            pid_flat = pid.unsqueeze(0).expand(k, *pid.shape).reshape(k*B, *pid.shape[1:])

            
            logp_j,prev_mean_j,std_j=model.single_denoise_with_logp(prev_x=prev_latents_flat,cur_x=cur_latents_flat,
                                                                    z=ar_latents_flat,t=t_flat,pred_ids=pid_flat)

            per_token_policy_loss_critic=None
            per_token_kl_critic=None
            if self.use_critic_token:
                num_seed=len(sample_all_diff_latents_seed["xs_next"])#把seed拼到最前面
                # seed,k*B,c,h,w
                prev_latents_k_seed=torch.stack(sample_all_diff_latents_seed["xs"],dim=0).index_select(1,selected_diff_steps_idx).reshape(-1,k*B, *prev_latents_k.shape[2:])
                cur_latents_k_seed=torch.stack(sample_all_diff_latents_seed["xs_next"],dim=0).index_select(1,selected_diff_steps_idx).reshape(-1,k*B, *prev_latents_k.shape[2:])#step i 输入的latent

                prev_latents_flat_seed = prev_latents_k_seed.reshape(-1, *prev_latents_k.shape[2:])#seed*k*b,c,h,w
                cur_latents_flat_seed  = cur_latents_k_seed.reshape(-1,  *cur_latents_k.shape[2:])#seed*k*B
                t_flat_seed = t_flat.unsqueeze(0).expand(num_seed,k*B).reshape(-1)#seed,k*B->seed*k*B
                pid_flat_seed = pid_flat.unsqueeze(0).expand(num_seed, *pid_flat.shape).reshape(-1, *pid.shape[1:])
                # completion_mask_flat = completion_mask_flat * critic_token_mask_flat
                # ----------------不带梯度
                # for name, param in model.image_decoder.named_parameters():
                #     param.requires_grad = False

                # sample_all_diff_latents_seed
                # sample_all_diff_latents_seed["xs"]
                # sample_all_diff_latents_seed["xs_next"]
                ar_latents_flat_seed = ar_latents_flat.unsqueeze(0).expand(num_seed, *ar_latents_flat.shape)#seed,k*B
                ar_latents_flat_seed = ar_latents_flat_seed.reshape(-1, *ar_latents_flat.shape[1:])#seed*k*B
                logp_j_critic, prev_mean_j_critic, std_j_critic = model.single_denoise_with_logp(
                    prev_x=prev_latents_flat_seed,
                    cur_x=cur_latents_flat_seed,
                    z=ar_latents_flat_seed,
                    t=t_flat_seed,
                    pred_ids=pid_flat_seed
                )

                # for name, param in model.image_decoder.named_parameters():
                #     param.requires_grad = True
                # ---------------带梯度
                mean_logp_j_critic = logp_j_critic.reshape(num_seed,-1,*logp_j_critic.shape[1:])#seed,k*B,c,h,w
                mean_prev_mean_j_critic = prev_mean_j_critic.reshape(num_seed,-1,*prev_mean_j_critic.shape[1:])#seed,k*B,c,h,w
                per_token_policy_loss_critic=torch.exp(mean_logp_j_critic - mean_logp_j_critic.detach()) * (advantages_flat.view(1,-1,1,1,1))
                per_token_policy_loss_critic=per_token_policy_loss_critic.mean(dim=0)

                if self.beta!=0 and self.critic_for_kl:
                    with torch.no_grad():
                        ref_ar_latents_flat_seed = ref_ar_latents_flat.unsqueeze(0).expand(num_seed, *ar_latents_flat.shape)#seed,k*B
                        ref_ar_latents_flat_seed = ref_ar_latents_flat_seed.reshape(-1, *ar_latents_flat.shape[1:])#seed*k*B
                        ref_logp_j_critic, ref_prev_mean_j_critic, ref_std_j_critic=self.ref_model.single_denoise_with_logp(
                            prev_x=prev_latents_flat_seed,
                            cur_x=cur_latents_flat_seed,
                            z=ref_ar_latents_flat_seed,
                            t=t_flat_seed,
                            pred_ids=pid_flat_seed
                        )
                    mean_ref_prev_mean_j_critic = ref_prev_mean_j_critic.reshape(num_seed,-1,*logp_j_critic.shape[1:])#seed,k*B,c,h,w
                    per_token_kl_critic = torch.exp(mean_ref_prev_mean_j_critic - mean_prev_mean_j_critic) - (mean_ref_prev_mean_j_critic - mean_prev_mean_j_critic) - 1
                    per_token_kl_critic=per_token_kl_critic.mean(dim=0)

            
            if self.beta != 0:
                with torch.no_grad():
                    ref_logp_j,ref_prev_mean_j,ref_std_j=self.ref_model.single_denoise_with_logp(prev_x=prev_latents_flat,cur_x=cur_latents_flat,
                                                                                                 z=ref_ar_latents_flat,t=t_flat,pred_ids=pid_flat)
                    
            per_token_policy_loss=torch.exp(logp_j - logp_j.detach()) * (advantages_flat.view(-1,1,1,1))
            if self.beta != 0:
                per_token_kl = torch.exp(ref_prev_mean_j - prev_mean_j) - (ref_prev_mean_j - prev_mean_j) - 1
            else:
                per_token_kl = torch.zeros_like(per_token_policy_loss)

            # 先算完对所有timestep的loss
            kl_mask=torch.ones_like(per_token_kl)==1
            if self.wokl:
                if 'geneval' in self.reward_funcs_name or 'ocr' in self.reward_funcs_name:
                    if valid_rewards.shape[1] == 0:
                        kl_mask=torch.ones_like(per_token_kl)==0#当当前reward std=0时，设置对应的kl mask为0
                
            per_token_kl = per_token_kl * kl_mask
            print('rank %s, iter %d kl:'%(dist.get_rank(),self.state.global_step),per_token_kl.mean())


            # # v1:根据当前mask step和final step的token的相似度算mask
            # if self.use_latent_sim_coef:
            #     # "ar_latents": frame_logs["ar_latents"]
            #     # cur_ar_latent=samples["ar_latents"][i]
            #     # final_ar_latent=samples["ar_latents"][-1]
            #     print('using use_latent_sim_coef...')
            #     cur_diff_latent=samples["prev_latents"][i][-1]#当前mask step的最后一个diff latent
            #     # final_diff_latent=samples["prev_latents"][-1][-1]#最后mask step的最后一个diff latent #fixbug: 这里的diffstep是错误的
            #     final_diff_latent=samples["final_diff_latents"][-1]

            #     B,_,_,_ = cur_diff_latent.shape  # L=1024
            #     sim_map_diff = F.cosine_similarity(cur_diff_latent, final_diff_latent, dim=1).view(B,-1,64,64)#

            #     sim_map_diff_flat = sim_map_diff.unsqueeze(0).expand(k, *sim_map_diff.shape).reshape(k*B, *sim_map_diff.shape[1:])*0.5+0.5
            #     hard_mask = (sim_map_diff_flat <= 0.9).float().to(sim_map_diff_flat.dtype)  # cos>=0
            #     sim_mask_diff_flat = hard_mask * completion_mask_flat
            #     # sim_mask_diff_flat = hard_mask * completion_mask_flat * (1-critic_token_mask_flat) + completion_mask_flat * critic_token_mask_flat

            # v2:根据当前mask step和prev step的相似度算mask
            if self.use_latent_sim_coef and i>1:
                # "ar_latents": frame_logs["ar_latents"]
                # cur_ar_latent=samples["ar_latents"][i]
                # final_ar_latent=samples["ar_latents"][-1]
                print('using use_latent_sim_coef...')
                cur_diff_latent=samples["prev_latents"][i][-1]#当前mask step的最后一个diff latent
                prev_diff_latent=samples["prev_latents"][i-1][-1]#当前mask step的最后一个diff latent
                final_diff_latent=samples["final_diff_latents"][-1]

                B,_,_,_ = cur_diff_latent.shape  # L=1024
                sim_map_diff_cur = F.cosine_similarity(cur_diff_latent, final_diff_latent, dim=1).view(B,-1,64,64)#
                sim_map_diff_prev = F.cosine_similarity(prev_diff_latent, final_diff_latent, dim=1).view(B,-1,64,64)#
                # sim_map_diff = torch.rand((B, 1, 64, 64),device=cur_diff_latent.device,dtype=cur_diff_latent.dtype)

                sim_map_diff = sim_map_diff_cur - sim_map_diff_prev

                sim_map_diff_flat = sim_map_diff.unsqueeze(0).expand(k, *sim_map_diff.shape).reshape(k*B, *sim_map_diff.shape[1:])#*0.5+0.5
                hard_mask = (sim_map_diff_flat > self.latent_sim_thresh).float().to(sim_map_diff_flat.dtype)  # cos>=0
                # hard_mask_ = (1-critic_token_mask_flat) * hard_mask + critic_token_mask_flat#critic的完全保留，不critic的随机mask
                hard_mask_ = hard_mask
                sim_mask_diff_flat = (hard_mask_ * completion_mask_flat + completion_mask_flat_next).clamp_(0,1)#下一个step要用的完全保留，否则随机mask
                # sim_mask_diff_flat = hard_mask * completion_mask_flat * (1-critic_token_mask_flat) + completion_mask_flat * critic_token_mask_flat

                save_sim_map = False
                if save_sim_map:
                    import matplotlib.pyplot as plt
                    import matplotlib as mpl
                    # from diffnext.image_processor import VaeImageProcessor
                    x = (sim_map_diff[0,0]/2 + 0.5).reshape(64,64).detach().float().cpu().numpy()
                    m = completion_mask[0,0].reshape(64,64).bool().cpu().numpy()
                    m_dim = completion_mask[0,0].reshape(64,64,-1).expand(64,64,3).bool().cpu().numpy()

                    cmap = mpl.cm.get_cmap("RdBu").copy()
                    cmap.set_bad((0.5,0.5,0.5,1.0))  # grey for (1-mask)

                    # x_final = (final_diff_latent[0,:3]).permute(1,2,0).reshape(64,64,3).detach().float()
                    # x_final = (x_final - x_final.min()) / (x_final.max() - x_final.min());x_final = (x_final * 255).byte().cpu().numpy()

                    x_cur = (cur_diff_latent[0,:3]).permute(1,2,0).reshape(64,64,3).detach().float()
                    x_cur = (x_cur - x_cur.min()) / (x_cur.max() - x_cur.min());x_cur = (x_cur * 255).byte().cpu().numpy()

                    x_mask = (sim_mask_diff_flat[0,:3]).permute(1,2,0).reshape(64,64,3).detach().float()
                    x_mask = (x_mask - x_mask.min()) / (x_mask.max() - x_mask.min());x_mask = (x_mask * 255).byte().cpu().numpy()
                    os.makedirs(os.path.join(os.path.dirname(__file__), 'output_latents_sim'),exist_ok=True)
                    plt.imsave(os.path.join(os.path.join(os.path.dirname(__file__), 'output_latents_sim'), 'sim_map_diff_step%d.png' % i), np.ma.array(x, mask=~m), cmap=cmap, vmin=0, vmax=1)
                    # plt.imsave("$PROJECT_ROOT/src/grpo/src/open_r1/trainer/output_latents_sim/diff_final.png", np.ma.array(x_final), cmap=cmap, vmin=0, vmax=1)
                    plt.imsave(os.path.join(os.path.join(os.path.dirname(__file__), 'output_latents_sim'), 'diff_step%d.png' % i), np.ma.array(x_cur, mask=~m_dim), cmap=cmap, vmin=0, vmax=1)
                    plt.imsave(os.path.join(os.path.join(os.path.dirname(__file__), 'output_latents_sim'), 'sim_mask_step%d.png' % i), np.ma.array(x_mask), cmap=cmap, vmin=0, vmax=1)

                    oup_images_inter=self.process_class.image_processor.decode_latents(self.vae, cur_diff_latent.clone())
                    output_name_latent = {4: "images", 5: "frames"}[len(oup_images_inter.shape)]
                    oup_images_inter = self.process_class.image_processor.postprocess(oup_images_inter, "pil")
                    img=NOVAPipelineOutput(**{output_name_latent: oup_images_inter}).images[0]
                    img.save(os.path.join(os.path.join(os.path.dirname(__file__), 'output_latents_sim'), 'oup_img_step%d.png' % i))

                # sim_mask_diff_flat = sim_map_diff_flat * completion_mask_flat
                # os.makedirs("cos_vis", exist_ok=True)
                # for b in range(B):
                #     plt.figure(figsize=(4,4), dpi=200)
                #     plt.imshow(m[b], vmin=-1, vmax=1)   # heatmap
                #     plt.colorbar(fraction=0.046, pad=0.04)
                #     plt.axis("off")
                #     plt.tight_layout(pad=0)
                #     plt.savefig(f"cos_vis/cos_b{b:03d}.png", bbox_inches="tight", pad_inches=0)
                #     plt.close()


            if per_token_kl_critic!=None:
                kl_loss = (per_token_kl * completion_mask_flat * (1-critic_token_mask_flat) + \
                    per_token_kl_critic * completion_mask_flat * critic_token_mask_flat).sum() / completion_mask_flat.sum()
                # kl_loss = (per_token_kl_critic * completion_mask_flat * critic_token_mask_flat).sum() / (completion_mask_flat * critic_token_mask_flat).sum()
            else: 
                kl_loss = (per_token_kl * completion_mask_flat).sum() / completion_mask_flat.sum()

            if per_token_policy_loss_critic!=None:
                if self.use_latent_sim_coef and i>1:
                    completion_mask_flat_ = completion_mask_flat * sim_mask_diff_flat
                else:
                    completion_mask_flat_ = completion_mask_flat
                policy_loss = (per_token_policy_loss * completion_mask_flat_ * (1-critic_token_mask_flat) + \
                    per_token_policy_loss_critic * completion_mask_flat_ * critic_token_mask_flat).sum() / completion_mask_flat_.sum()
                # policy_loss = (per_token_policy_loss * completion_mask_flat * (1-critic_token_mask_flat) + \
                #     per_token_policy_loss_critic * completion_mask_flat * critic_token_mask_flat).sum() / completion_mask_flat.sum()
                # policy_loss = (per_token_policy_loss_critic * completion_mask_flat * critic_token_mask_flat).sum() / (completion_mask_flat * critic_token_mask_flat).sum()
            else:
                policy_loss = (per_token_policy_loss * completion_mask_flat).sum() / completion_mask_flat.sum()
            
            total_kl += kl_loss
            total_policy_loss += policy_loss#需要考虑一下是算per token的，还是per timestep，先保存成per timestep的吧
            cur_step_total_loss = -(policy_loss-self.beta*kl_loss)
            total_loss += cur_step_total_loss
                
            # loss_dict['token-cot']['per_timestep_loss'].append(total_loss)
            # self.accelerator.backward(total_loss)
            self.loss_backward(total_loss, retain_graph=False,
                                  divide_by_gas=True, mean_over_devices=True)
            per_timestep_loss.append(total_policy_loss)
            per_timestep_kl.append(total_kl)
            
        loss_dict['token-cot'] = {
            'per_timestep_policy_loss': per_timestep_loss,
            'per_timestep_kl':per_timestep_kl,
            }
                        
        
        torch.set_grad_enabled(False)
        all_timestep_policy_loss=torch.stack(per_timestep_loss).mean()
        all_timestep_kl=torch.stack(per_timestep_kl).mean()
        loss=-(all_timestep_policy_loss-self.beta*all_timestep_kl)
        
        self._metrics[f"kl"].append(self.accelerator.gather_for_metrics(all_timestep_kl).mean().item())
        self._metrics[f"loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())#?写错了吧，应该是loss不是kl
        

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        total_reward=self.accelerator.gather_for_metrics(rewards)
        self._metrics["reward"].append(total_reward[total_reward!=-1].mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        if 'geneval' in self.reward_funcs_name:
            self._metrics["valid_mask_rate"].append(self.accelerator.gather_for_metrics(valid_mask.float()).mean().item())
        elif 'ocr' in self.reward_funcs_name:
            self._metrics["valid_mask_rate_ocr"].append(self.accelerator.gather_for_metrics(valid_mask.float()).mean().item())
        
        if self.update_ref:
            # {600: 400, 1000: 800}
            self.maybe_update_ref_model(self.state.global_step,ref_schedule = {800:500,1200:900})
        
        if self.state.global_step==0:
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        return torch.tensor(loss,device=self.args.device, dtype=torch.float32)
    

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
