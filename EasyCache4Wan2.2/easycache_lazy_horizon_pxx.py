# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# EasyCache + horizon Lazy-MLP acceleration strategy for Wan2.2
#
# The MLP predicts future EasyCache error increments over a fixed horizon.
# The inference policy takes the longest safe prefix whose cumulative predicted
# error stays below the EasyCache threshold, then reuses the existing residual
# cache for those CFG pairs. The MLP never generates a latent state directly.

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from tqdm import tqdm
from time import time
from collections import deque

from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

EXAMPLE_PROMPT = {
    # "t2v-A14B": {
    #     "prompt":
    #         "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    # },
    # "i2v-A14B": {
    #     "prompt":
    #         "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    #     "image":
    #         "examples/i2v_input.JPG",
    # },
    "ti2v-5B": {
        "prompt":
            # "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
            "A river flows through a deep canyon under bright daylight, birds circling above, majestic natural landscape."
    },
}


class LazyHorizonPredictor(nn.Module):
    """Predict non-negative EasyCache error increments over a future horizon."""

    FEATURE_DIM = 8

    def __init__(
        self,
        hidden_dim=128,
        num_hidden_layers=3,
        horizon=4,
        initial_prediction=0.01,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if initial_prediction <= 0:
            raise ValueError("initial_prediction must be positive")

        self.horizon = horizon
        layers = []
        in_dim = self.FEATURE_DIM
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, horizon))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(math.expm1(initial_prediction)),
        )

    def forward(self, features):
        # [B, horizon], each value is a non-negative future error increment.
        return F.softplus(self.network(features))


def maximum_safe_prefix(predicted_increments, current_error, threshold):
    """Return the longest leading prefix whose cumulative error is safe."""
    if predicted_increments.ndim != 1:
        raise ValueError(
            "predicted_increments must be one-dimensional, got "
            f"{tuple(predicted_increments.shape)}"
        )
    cumulative = current_error + torch.cumsum(predicted_increments, dim=0)
    safe = cumulative < threshold
    prefix = int(safe.to(torch.long).cumprod(dim=0).sum().item())
    return prefix, cumulative


def clear_lazy_plan(self):
    self.lazy_planned_increments.clear()
    self.lazy_force_calc_next = False


def extract_lazy_features(
    t_norm,
    input_change_curr,
    input_change_prev,
    input_mean_curr,
    input_mean_prev,
    output_change_prev,
    residual_norm,
    step_progress,
    device='cpu',
):
    features = torch.tensor([
        t_norm,
        input_change_curr,
        input_change_prev,
        input_mean_curr,
        input_mean_prev,
        output_change_prev,
        residual_norm,
        step_progress,
    ], dtype=torch.float32, device=device)
    return features


def compute_tensor_list_l1_mean(tensor_list_a, tensor_list_b=None):
    if tensor_list_b is not None:
        diff = torch.cat([(u - v).flatten() for u, v in zip(tensor_list_a, tensor_list_b)])
    else:
        diff = torch.cat([u.flatten() for u in tensor_list_a])
    return diff.abs().mean().item()


def compute_tensor_list_l1_norm_mean(tensor_list):
    return torch.cat([u.flatten() for u in tensor_list]).abs().mean().item()


def lazy_easycache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
):
    if self.model_type == 'i2v':
        assert y is not None

    raw_input = [u.clone() for u in x]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_even = (self.cnt % 2 == 0)

    if self.is_even:
        force_full_region = self.cnt < self.ret_steps or self.cnt >= (
            ((getattr(self, "low_start_step", None) is not None
              and getattr(self, "is_high_noise", False))
             and (self.low_start_step - 1) * 2 - 2)
            or
            ((getattr(self, "low_start_step", None) is not None
              and not getattr(self, "is_high_noise", False))
             and (self.num_steps - self.low_start_step) * 2 - 2)
            or (self.num_steps * 2 - 2)
        )

        if force_full_region:
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
            clear_lazy_plan(self)

        elif len(self.lazy_planned_increments) > 0:
            # Continue a previously selected safe prefix without another MLP call.
            planned_increment = float(self.lazy_planned_increments.popleft())
            next_error = self.accumulated_error_even + planned_increment
            effective_threshold = self.thresh * self.lazy_prefix_safety

            if next_error >= effective_threshold:
                # Defensive fallback; this should not occur for a valid plan.
                logging.warning(
                    "[LazyHorizon] planned increment became unsafe: "
                    "cnt=%d next_error=%.6f threshold=%.6f",
                    self.cnt,
                    next_error,
                    effective_threshold,
                )
                self.should_calc_current_pair = True
                self.accumulated_error_even = 0.0
                clear_lazy_plan(self)
            else:
                self.accumulated_error_even = next_error
                self.should_calc_current_pair = False
                self.lazy_planned_skip_cnt += 1
                logging.info(
                    "[LazyHorizon] cnt=%d | follow_plan increment=%.6f | "
                    "accumulated=%.6f | remaining=%d | decision=SKIP",
                    self.cnt,
                    planned_increment,
                    self.accumulated_error_even,
                    len(self.lazy_planned_increments),
                )

        elif self.lazy_force_calc_next:
            # The next offset after the selected prefix was predicted unsafe.
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
            self.lazy_force_calc_next = False
            self.lazy_forced_calc_cnt += 1
            logging.info(
                "[LazyHorizon] cnt=%d | prefix exhausted | decision=CALC",
                self.cnt,
            )

        elif (
            self.previous_raw_input_even is not None
            and self.previous_raw_output_even is not None
        ):
            input_change_curr = compute_tensor_list_l1_mean(
                raw_input,
                self.previous_raw_input_even,
            )
            input_norm_prev = compute_tensor_list_l1_norm_mean(
                self.previous_raw_input_even
            )
            input_change_curr_rel = input_change_curr / (input_norm_prev + 1e-8)

            if self.prev_prev_raw_input_even is not None:
                input_change_prev = compute_tensor_list_l1_mean(
                    self.previous_raw_input_even,
                    self.prev_prev_raw_input_even,
                )
                input_norm_prev_prev = compute_tensor_list_l1_norm_mean(
                    self.prev_prev_raw_input_even
                )
                input_change_prev_rel = (
                    input_change_prev / (input_norm_prev_prev + 1e-8)
                )
            else:
                input_change_prev_rel = 0.0

            input_mean_curr = compute_tensor_list_l1_norm_mean(raw_input)
            input_mean_prev = input_norm_prev

            if self.prev_prev_raw_output_even is not None:
                output_change_prev = compute_tensor_list_l1_mean(
                    self.previous_raw_output_even,
                    self.prev_prev_raw_output_even,
                )
                output_norm_prev_prev = compute_tensor_list_l1_norm_mean(
                    self.prev_prev_raw_output_even
                )
                output_change_prev_rel = (
                    output_change_prev / (output_norm_prev_prev + 1e-8)
                )
            else:
                output_change_prev_rel = 0.0

            if self.cache_even is not None:
                cache_norm = compute_tensor_list_l1_norm_mean(self.cache_even)
                residual_norm = cache_norm / (input_mean_curr + 1e-8)
            else:
                residual_norm = 0.0

            if hasattr(t, 'numel') and t.numel() > 1:
                t_value = t.flatten()[0].item()
            elif hasattr(t, 'item'):
                t_value = t.item()
            else:
                t_value = float(t)

            features = extract_lazy_features(
                t_norm=t_value / 1000.0,
                input_change_curr=input_change_curr_rel,
                input_change_prev=input_change_prev_rel,
                input_mean_curr=input_mean_curr,
                input_mean_prev=input_mean_prev,
                output_change_prev=output_change_prev_rel,
                residual_norm=residual_norm,
                step_progress=self.cnt / max(self.num_steps, 1),
                device=device,
            )

            with torch.no_grad():
                future_pred = self.lazy_predictor(features.unsqueeze(0))[0]

            if future_pred.shape != (self.lazy_horizon,):
                raise RuntimeError(
                    "Lazy horizon predictor returned shape "
                    f"{tuple(future_pred.shape)}, expected ({self.lazy_horizon},)"
                )

            easycache_pred_change = 0.0
            if self.k is not None:
                output_norm = compute_tensor_list_l1_norm_mean(
                    self.previous_raw_output_even
                )
                easycache_pred_change = self.k * (
                    input_change_curr / (output_norm + 1e-8)
                )

            # Optional compatibility blend for offset 0 only. Future offsets are
            # necessarily predicted by the horizon MLP.
            future_pred = future_pred.clone()
            future_pred[0] = (
                self.lazy_weight * future_pred[0]
                + (1.0 - self.lazy_weight) * easycache_pred_change
            )

            effective_threshold = self.thresh * self.lazy_prefix_safety
            safe_prefix, future_cumulative = maximum_safe_prefix(
                future_pred,
                self.accumulated_error_even,
                effective_threshold,
            )

            self.lazy_plan_cnt += 1
            self._last_features = features
            self._last_easycache_pred = easycache_pred_change
            self._last_mlp_pred = future_pred.detach().clone()

            if safe_prefix > 0:
                planned = future_pred[:safe_prefix].detach().cpu().tolist()
                current_increment = float(planned[0])
                self.accumulated_error_even += current_increment
                self.lazy_planned_increments.extend(planned[1:])
                self.lazy_force_calc_next = safe_prefix < self.lazy_horizon
                self.should_calc_current_pair = False
                self.lazy_planned_skip_cnt += 1
            else:
                self.should_calc_current_pair = True
                self.accumulated_error_even = 0.0
                clear_lazy_plan(self)

            logging.info(
                "[LazyHorizon] cnt=%d | easy0=%.6f | pred=%s | "
                "cum=%s | safe_prefix=%d/%d | accumulated=%.6f | "
                "remaining=%d | decision=%s",
                self.cnt,
                easycache_pred_change,
                [round(v, 6) for v in future_pred.detach().cpu().tolist()],
                [round(v, 6) for v in future_cumulative.detach().cpu().tolist()],
                safe_prefix,
                self.lazy_horizon,
                self.accumulated_error_even,
                len(self.lazy_planned_increments),
                "CALC" if self.should_calc_current_pair else "SKIP",
            )

        else:
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
            clear_lazy_plan(self)

        # Preserve the original EasyCache input-history semantics: this updates
        # on every conditional candidate, including skipped candidates.
        self.previous_raw_input_even = [u.clone() for u in raw_input]

        if self.should_calc_current_pair:
            self.calc_pair_cnt += 1
        else:
            self.skip_pair_cnt += 1

    if (
        self.is_even
        and not self.should_calc_current_pair
        and self.previous_raw_output_even is not None
    ):
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_even)]

    if (
        not self.is_even
        and not self.should_calc_current_pair
        and self.previous_raw_output_odd is not None
    ):
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_odd)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([
            u,
            u.new_zeros(1, seq_len - u.size(1), u.size(2)),
        ], dim=1)
        for u in x
    ])

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t)
            .unflatten(0, (bt, seq_len))
            .float()
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([
                u,
                u.new_zeros(self.text_len - u.size(0), u.size(1)),
            ])
            for u in context
        ])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    if self.is_even:
        if self.previous_raw_output_even is not None:
            output_change = compute_tensor_list_l1_mean(
                output,
                self.previous_raw_output_even,
            )

            if self.prev_prev_raw_input_even is not None:
                input_change = compute_tensor_list_l1_mean(
                    self.previous_raw_input_even,
                    self.prev_prev_raw_input_even,
                )
                if input_change > 1e-10:
                    self.k = output_change / input_change

        self.prev_prev_raw_input_even = self.previous_raw_input_even
        self.prev_prev_raw_output_even = self.previous_raw_output_even
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [u - v for u, v in zip(output, raw_input)]
    else:
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [u - v for u, v in zip(output, raw_input)]

    self.cnt += 1
    return [u.float() for u in output]



def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="EasyCache + Lazy Learning hybrid acceleration for Wan2.2 video generation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ti2v-5B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*704",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/Wan2.2-TI2V-5B",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        # default=-1,
        default=123,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    parser.add_argument(
        "--thresh_ti2v",
        type=float,
        default=0.1,
        help="Threshold for skip decision (ti2v task)")
    parser.add_argument(
        "--thresh_t2v",
        type=float,
        default=0.06,
        help="Threshold for skip decision (t2v task)")
    parser.add_argument(
        "--thresh_i2v",
        type=float,
        default=0.08,
        help="Threshold for skip decision (i2v task)")
    parser.add_argument(
        "--ret_steps",
        type=int,
        default=7,
        help="Number of initial steps to always compute (retained steps)")

    parser.add_argument(
        "--lazy_weight",
        type=float,
        default=1.0,
        help=(
            "Blend weight for the current offset only: 1=pure horizon MLP, "
            "0=EasyCache for offset 0. Future offsets always use the MLP."
        ),
    )
    parser.add_argument(
        "--lazy_horizon",
        type=int,
        default=4,
        help="Future error horizon; must match the training checkpoint.",
    )
    parser.add_argument(
        "--lazy_prefix_safety",
        type=float,
        default=1.0,
        help=(
            "Safety multiplier applied to the EasyCache threshold when selecting "
            "the maximum prefix. Values below 1 are more conservative."
        ),
    )
    parser.add_argument(
        "--lazy_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of LazyHorizonPredictor.",
    )
    parser.add_argument(
        "--lazy_num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers in LazyHorizonPredictor.",
    )
    parser.add_argument(
        "--lazy_initial_prediction",
        type=float,
        default=0.01,
        help="Initialization used only before loading a checkpoint.",
    )
    parser.add_argument(
        "--lazy_online_train",
        action="store_true",
        default=False,
        help="Unsupported for horizon inference; kept to reject stale commands.",
    )
    parser.add_argument(
        "--lazy_ckpt",
        type=str,
        default=(
            "/starmap/nas182/h00446189/code/EasyCache-main/"
            "results/easycache_lazy_horizon/lazy_horizon_predictor_full.pt"
        ),
        help="Path to trained LazyHorizonPredictor state_dict.",
    )
    parser.add_argument(
        "--lazy_save_ckpt",
        type=str,
        default=None,
        help="Optional path to save the unchanged horizon predictor state_dict.",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 5))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
                args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
                args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if args.lazy_online_train:
        raise ValueError(
            "--lazy_online_train is incompatible with the multi-step horizon "
            "checkpoint. Retrain offline instead."
        )
    if args.lazy_horizon <= 0:
        raise ValueError("--lazy_horizon must be positive")
    if not 0.0 <= args.lazy_weight <= 1.0:
        raise ValueError("--lazy_weight must be in [0,1]")
    if not 0.0 < args.lazy_prefix_safety <= 1.0:
        raise ValueError("--lazy_prefix_safety must be in (0,1]")

    torch_device = torch.device(f"cuda:{device}")
    lazy_predictor = LazyHorizonPredictor(
        hidden_dim=args.lazy_hidden_dim,
        num_hidden_layers=args.lazy_num_hidden_layers,
        horizon=args.lazy_horizon,
        initial_prediction=args.lazy_initial_prediction,
    ).to(torch_device)

    if args.lazy_ckpt is None:
        raise ValueError("--lazy_ckpt is required for horizon inference")

    try:
        state_dict = torch.load(
            args.lazy_ckpt,
            map_location=torch_device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(args.lazy_ckpt, map_location=torch_device)

    if "raw_input_state_dict" in state_dict:
        state_dict = state_dict["raw_input_state_dict"]

    lazy_predictor.load_state_dict(state_dict)
    lazy_predictor.eval()

    final_layer = lazy_predictor.network[-1]
    if final_layer.out_features != args.lazy_horizon:
        raise RuntimeError(
            "Checkpoint/model horizon mismatch: final layer has "
            f"{final_layer.out_features} outputs, --lazy_horizon={args.lazy_horizon}"
        )

    logging.info("Loaded LazyHorizonPredictor from %s", args.lazy_ckpt)
    logging.info("LazyHorizonPredictor architecture:\n%s", lazy_predictor)
    logging.info(
        "LazyHorizonPredictor parameters=%d horizon=%d current_weight=%.3f "
        "prefix_safety=%.3f",
        sum(p.numel() for p in lazy_predictor.parameters()),
        args.lazy_horizon,
        args.lazy_weight,
        args.lazy_prefix_safety,
    )

    if "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        wan_ti2v.model.__class__.forward = lazy_easycache_forward
        wan_ti2v.model.__class__.cnt = 0
        wan_ti2v.model.__class__.num_steps = args.sample_steps * 2
        wan_ti2v.model.__class__.thresh = args.thresh_ti2v
        wan_ti2v.model.__class__.accumulated_error_even = 0
        wan_ti2v.model.__class__.should_calc_current_pair = True
        wan_ti2v.model.__class__.k = None
        wan_ti2v.model.__class__.previous_raw_input_even = None
        wan_ti2v.model.__class__.previous_raw_output_even = None
        wan_ti2v.model.__class__.previous_raw_output_odd = None
        wan_ti2v.model.__class__.prev_prev_raw_input_even = None
        wan_ti2v.model.__class__.prev_prev_raw_output_even = None
        wan_ti2v.model.__class__.cache_even = None
        wan_ti2v.model.__class__.cache_odd = None
        wan_ti2v.model.__class__.ret_steps = args.ret_steps * 2
        wan_ti2v.model.__class__.lazy_predictor = lazy_predictor
        wan_ti2v.model.__class__.lazy_weight = args.lazy_weight
        wan_ti2v.model.__class__.lazy_horizon = args.lazy_horizon
        wan_ti2v.model.__class__.lazy_prefix_safety = args.lazy_prefix_safety
        wan_ti2v.model.__class__._last_features = None
        wan_ti2v.model.__class__._last_easycache_pred = 0.0
        wan_ti2v.model.__class__._last_mlp_pred = None

        wan_ti2v.model.__class__.lazy_planned_increments = deque()
        wan_ti2v.model.__class__.lazy_force_calc_next = False
        wan_ti2v.model.__class__.lazy_plan_cnt = 0
        wan_ti2v.model.__class__.lazy_planned_skip_cnt = 0
        wan_ti2v.model.__class__.lazy_forced_calc_cnt = 0

        wan_ti2v.model.__class__.skip_pair_cnt = 0
        wan_ti2v.model.__class__.calc_pair_cnt = 0

        logging.info("Generating video with EasyCache + horizon Lazy-MLP strategy ...")
        video = wan_ti2v.generate(
            args.prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

        total_pairs = (
            wan_ti2v.model.calc_pair_cnt
            + wan_ti2v.model.skip_pair_cnt
        )

        logging.info(
            "Lazy horizon statistics: calc_pairs=%d, skip_pairs=%d, "
            "skip_ratio=%.2f%%, plans=%d, planned_skips=%d, "
            "forced_calcs=%d",
            wan_ti2v.model.calc_pair_cnt,
            wan_ti2v.model.skip_pair_cnt,
            100.0 * wan_ti2v.model.skip_pair_cnt / max(total_pairs, 1),
            wan_ti2v.model.lazy_plan_cnt,
            wan_ti2v.model.lazy_planned_skip_cnt,
            wan_ti2v.model.lazy_forced_calc_cnt,
        )

    if args.lazy_save_ckpt is not None:
        torch.save(lazy_predictor.state_dict(), args.lazy_save_ckpt)
        logging.info(f"LazyHorizonPredictor saved to {args.lazy_save_ckpt}")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)