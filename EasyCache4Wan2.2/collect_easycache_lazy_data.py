# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
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
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}

def easycache_forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
):
    """
    EasyCache forward with optional lazy-training data collection.

    When ``self.collect_lazy_data`` is True, every CFG pair is evaluated with
    the full DiT so that the true EasyCache approximation error is available.
    The generated video still follows the original Wan pipeline, including VAE
    decoding and video saving.
    """
    if self.model_type == 'i2v':
        assert y is not None

    # Store original raw input for end-to-end caching.
    raw_input = [u.clone() for u in x]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # even = conditional, odd = unconditional
    self.is_even = (self.cnt % 2 == 0)

    # Preserve the scalar timestep before the embedding code expands it.
    if torch.is_tensor(t):
        t_value = float(t.flatten()[0].item())
    else:
        t_value = float(t)

    pending_features = None

    # Build exactly the 8-D feature used by LazySkipPredictor.  Features are
    # computed before the current full output is available and paired with the
    # true cache error after the forward pass.
    if (
        self.is_even
        and getattr(self, 'collect_lazy_data', False)
        and getattr(self, 'previous_raw_input_even', None) is not None
        and getattr(self, 'previous_raw_output_even', None) is not None
        and getattr(self, 'cache_even', None) is not None
    ):
        input_change_curr = torch.cat([
            (u - v).flatten()
            for u, v in zip(raw_input, self.previous_raw_input_even)
        ]).abs().mean().item()
        input_norm_prev = torch.cat([
            u.flatten() for u in self.previous_raw_input_even
        ]).abs().mean().item()
        input_change_curr_rel = input_change_curr / (input_norm_prev + 1e-8)

        if getattr(self, 'prev_prev_raw_input_even', None) is not None:
            input_change_prev = torch.cat([
                (u - v).flatten()
                for u, v in zip(
                    self.previous_raw_input_even,
                    self.prev_prev_raw_input_even,
                )
            ]).abs().mean().item()
            input_norm_prev_prev = torch.cat([
                u.flatten() for u in self.prev_prev_raw_input_even
            ]).abs().mean().item()
            input_change_prev_rel = (
                input_change_prev / (input_norm_prev_prev + 1e-8)
            )
        else:
            input_change_prev_rel = 0.0

        input_mean_curr = torch.cat([
            u.flatten() for u in raw_input
        ]).abs().mean().item()
        input_mean_prev = input_norm_prev

        if getattr(self, 'prev_prev_raw_output_even', None) is not None:
            output_change_prev = torch.cat([
                (u - v).flatten()
                for u, v in zip(
                    self.previous_raw_output_even,
                    self.prev_prev_raw_output_even,
                )
            ]).abs().mean().item()
            output_norm_prev_prev = torch.cat([
                u.flatten() for u in self.prev_prev_raw_output_even
            ]).abs().mean().item()
            output_change_prev_rel = (
                output_change_prev / (output_norm_prev_prev + 1e-8)
            )
        else:
            output_change_prev_rel = 0.0

        cache_norm = torch.cat([
            u.flatten() for u in self.cache_even
        ]).abs().mean().item()
        residual_norm = cache_norm / (input_mean_curr + 1e-8)

        pending_features = torch.tensor([
            t_value / 1000.0,
            input_change_curr_rel,
            input_change_prev_rel,
            input_mean_curr,
            input_mean_prev,
            output_change_prev_rel,
            residual_norm,
            self.cnt / max(self.num_steps, 1),
        ], dtype=torch.float32)

    # Data collection needs the true DiT output at every timestep. Outside
    # collection mode, retain the original EasyCache skip policy unchanged.
    if self.is_even:
        if getattr(self, 'collect_lazy_data', False):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0
        elif self.cnt < self.ret_steps or self.cnt >= (
                ((getattr(self, "low_start_step", None) is not None and getattr(self, "is_high_noise", False)) and (
                        self.low_start_step - 1) * 2 - 2) or
                ((getattr(self, "low_start_step", None) is not None and not getattr(self, "is_high_noise", False)) and (
                        self.num_steps - self.low_start_step) * 2 - 2) or
                (self.num_steps * 2 - 2)
        ):
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0
        else:
            if (
                getattr(self, 'previous_raw_input_even', None) is not None
                and getattr(self, 'previous_raw_output_even', None) is not None
            ):
                raw_input_change = torch.cat([
                    (u - v).flatten()
                    for u, v in zip(raw_input, self.previous_raw_input_even)
                ]).abs().mean()

                if getattr(self, 'k', None) is not None:
                    output_norm = torch.cat([
                        u.flatten() for u in self.previous_raw_output_even
                    ]).abs().mean()
                    pred_change = self.k * (
                        raw_input_change / (output_norm + 1e-8)
                    )
                    if not hasattr(self, 'accumulated_error_even'):
                        self.accumulated_error_even = 0
                    self.accumulated_error_even += pred_change
                    if self.accumulated_error_even < self.thresh:
                        self.should_calc_current_pair = False
                    else:
                        self.should_calc_current_pair = True
                        self.accumulated_error_even = 0
                else:
                    self.should_calc_current_pair = True
            else:
                self.should_calc_current_pair = True


    if (
        self.is_even
        and not self.should_calc_current_pair
        and getattr(self, 'previous_raw_output_even', None) is not None
    ):
        self.prev_prev_raw_input_even = getattr(
            self, 'previous_raw_input_even', None
        )
        self.previous_raw_input_even = [u.clone() for u in raw_input]
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_even)]

    if (
        not self.is_even
        and not self.should_calc_current_pair
        and getattr(self, 'previous_raw_output_odd', None) is not None
    ):
        self.cnt += 1
        return [(u + v).float() for u, v in zip(raw_input, self.cache_odd)]

    # Original full DiT forward.
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([
        torch.tensor(u.shape[2:], dtype=torch.long) for u in x
    ])
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
            sinusoidal_embedding_1d(
                self.freq_dim,
                t,
            ).unflatten(0, (bt, seq_len)).float()
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
        previous_output = getattr(self, 'previous_raw_output_even', None)
        previous_cache = getattr(self, 'cache_even', None)

        # The target is the relative error that EasyCache would make at this
        # exact timestep by reusing the previous conditional residual.
        if (
            getattr(self, 'collect_lazy_data', False)
            and pending_features is not None
            and previous_cache is not None
        ):
            cached_output = [
                u + cache
                for u, cache in zip(raw_input, previous_cache)
            ]
            cache_error = torch.cat([
                (u - v).flatten()
                for u, v in zip(output, cached_output)
            ]).abs().mean().item()
            output_norm = torch.cat([
                u.flatten() for u in output
            ]).abs().mean().item()
            target = cache_error / (output_norm + 1e-8)

            self.collect_features.append(pending_features.cpu())
            self.collect_targets.append(float(target))
            self.collect_timesteps.append(float(t_value))
            self.collect_step_indices.append(int(self.cnt // 2))

        if previous_output is not None:
            output_change = torch.cat([
                (u - v).flatten()
                for u, v in zip(output, previous_output)
            ]).abs().mean()

            previous_input = getattr(self, 'previous_raw_input_even', None)
            if previous_input is not None:
                input_change = torch.cat([
                    (u - v).flatten()
                    for u, v in zip(raw_input, previous_input)
                ]).abs().mean()
                if input_change > 1e-10:
                    self.k = output_change / input_change

        self.prev_prev_raw_input_even = getattr(
            self, 'previous_raw_input_even', None
        )
        self.previous_raw_input_even = [u.clone() for u in raw_input]
        self.prev_prev_raw_output_even = previous_output
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [
            u - v for u, v in zip(output, raw_input)
        ]
    else:
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [u - v for u, v in zip(output, raw_input)]

    self.cnt += 1
    return [u.float() for u in output]

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt_file is not None:
        assert os.path.isfile(args.prompt_file), \
            f"Prompt file does not exist: {args.prompt_file}"
    elif args.prompt is None:
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
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
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
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/Wan2.2-TI2V-5B",
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
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
        help="The prompt to generate the video from when --prompt_file is not used.")
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/starmap/nas182/h00446189/code/EasyCache-main/EasyCache4Wan2.2/prompt.txt",
        help="UTF-8 text file containing one prompt per non-empty line. "
             "The Wan model is loaded once and reused for all prompts.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory used in prompt-file batch mode. "
             "Defaults to the directory of --save_file or the current directory.")
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
        default=-1,
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
        default=0.05,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--thresh_t2v",
        type=float,
        default=0.06,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--thresh_i2v",
        type=float,
        default=0.05,
        help="Threshold for EasyCache decision making")
    parser.add_argument(
        "--ret_steps",
        type=int,
        default=7,
        help="Number of steps to retain in cache")
    parser.add_argument(
        "--lazy_data_save",
        type=str,
        default=None,
        help="Single-run output .pt path. In --prompt_file mode, a .pt value "
             "is used as a filename prefix; a path without .pt is treated as "
             "the dataset output directory. Defaults next to each video.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
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
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
                args.ulysses_size > 1
        ), "sequence parallel is not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            "The number of ulysses_size should be equal to the world size."
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
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"{cfg.num_heads=} cannot be divided evenly by {args.ulysses_size=}."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Read the prompt list once. Empty lines and lines beginning with '#' are
    # ignored. Under distributed execution, rank 0 reads and broadcasts it so
    # that every rank processes exactly the same sequence.
    if rank == 0:
        if args.prompt_file is not None:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompts = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            if not prompts:
                raise ValueError(
                    f"No valid prompts were found in {args.prompt_file}.")
        else:
            prompts = [args.prompt]
    else:
        prompts = None

    if dist.is_initialized():
        prompt_container = [prompts]
        dist.broadcast_object_list(prompt_container, src=0)
        prompts = prompt_container[0]

    logging.info(f"Loaded {len(prompts)} prompt(s). The Wan model will be loaded once.")

    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    if "ti2v" not in args.task:
        raise NotImplementedError(
            "This minimally modified collector currently follows the original "
            "TI2V-only script and supports ti2v tasks only.")

    logging.info("Creating WanTI2V pipeline once.")
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

    # Preserve the existing forward replacement and static settings. Per-video
    # mutable state is reset inside the loop below.
    wan_ti2v.model.__class__.forward = easycache_forward
    wan_ti2v.model.num_steps = args.sample_steps * 2
    wan_ti2v.model.thresh = args.thresh_ti2v
    wan_ti2v.model.ret_steps = args.ret_steps * 2
    wan_ti2v.model.collect_lazy_data = True

    for prompt_index, raw_prompt in enumerate(prompts):
        current_prompt = raw_prompt
        sample_seed = args.base_seed + prompt_index

        # Prompt extension is performed independently for each line while the
        # prompt-expansion model itself remains loaded.
        if args.use_prompt_extend:
            if rank == 0:
                prompt_output = prompt_expander(
                    current_prompt,
                    image=img,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=sample_seed)
                if prompt_output.status is False:
                    logging.info(
                        f"Prompt {prompt_index}: extension failed: "
                        f"{prompt_output.message}; using the original prompt.")
                else:
                    current_prompt = prompt_output.prompt
                prompt_payload = [current_prompt]
            else:
                prompt_payload = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(prompt_payload, src=0)
            current_prompt = prompt_payload[0]

        logging.info(
            f"[{prompt_index + 1}/{len(prompts)}] Generating with seed "
            f"{sample_seed}: {current_prompt}")

        # Reset all per-trajectory EasyCache and collection state. This prevents
        # any latent/cache history from one prompt contaminating the next.
        wan_ti2v.model.cnt = 0
        wan_ti2v.model.accumulated_error_even = 0
        wan_ti2v.model.should_calc_current_pair = True
        wan_ti2v.model.k = None
        wan_ti2v.model.previous_raw_input_even = None
        wan_ti2v.model.previous_raw_output_even = None
        wan_ti2v.model.previous_raw_output_odd = None
        wan_ti2v.model.prev_prev_raw_input_even = None
        wan_ti2v.model.prev_prev_raw_output_even = None
        wan_ti2v.model.cache_even = None
        wan_ti2v.model.cache_odd = None
        wan_ti2v.model.collect_features = []
        wan_ti2v.model.collect_targets = []
        wan_ti2v.model.collect_timesteps = []
        wan_ti2v.model.collect_step_indices = []

        video = wan_ti2v.generate(
            current_prompt,
            img=img,
            size=SIZE_CONFIGS[args.size],
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=sample_seed,
            offload_model=args.offload_model)

        # Move the decoded video off GPU immediately. Collected feature tensors
        # have already been moved to CPU inside easycache_forward; the explicit
        # conversion below guarantees this remains true before serialization.
        video = video.detach().cpu()
        features_cpu = (
            torch.stack([
                feature.detach().to(device="cpu", dtype=torch.float32)
                for feature in wan_ti2v.model.collect_features
            ])
            if wan_ti2v.model.collect_features
            else torch.empty((0, 8), dtype=torch.float32)
        )
        targets_cpu = torch.tensor(
            wan_ti2v.model.collect_targets,
            dtype=torch.float32,
            device="cpu")
        timesteps_cpu = torch.tensor(
            wan_ti2v.model.collect_timesteps,
            dtype=torch.float32,
            device="cpu")
        step_indices_cpu = torch.tensor(
            wan_ti2v.model.collect_step_indices,
            dtype=torch.long,
            device="cpu")

        formatted_prompt = current_prompt.replace(" ", "_").replace("/", "_")[:50]
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.prompt_file is None:
            if args.save_file is None:
                video_save_path = (
                    f"{args.task}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_"
                    f"{args.ulysses_size}_{formatted_prompt}_{formatted_time}.mp4"
                )
            else:
                video_save_path = args.save_file
        else:
            if args.output_dir is not None:
                output_dir = args.output_dir
            elif args.save_file is not None and os.path.dirname(args.save_file):
                output_dir = os.path.dirname(args.save_file)
            else:
                output_dir = "."
            os.makedirs(output_dir, exist_ok=True)
            if args.save_file is not None:
                file_prefix = os.path.splitext(os.path.basename(args.save_file))[0]
            else:
                file_prefix = (
                    f"{args.task}_{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_"
                    f"{args.ulysses_size}"
                )
            video_save_path = os.path.join(
                output_dir,
                f"{file_prefix}_{prompt_index:05d}_{formatted_prompt}.mp4")

        video_dir = os.path.dirname(video_save_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        if args.prompt_file is None:
            data_save_path = args.lazy_data_save
            if data_save_path is None:
                data_save_path = (
                    os.path.splitext(video_save_path)[0] + "_lazy_data.pt")
        else:
            if args.lazy_data_save is None:
                data_save_path = (
                    os.path.splitext(video_save_path)[0] + "_lazy_data.pt")
            else:
                lazy_root, lazy_ext = os.path.splitext(args.lazy_data_save)
                if lazy_ext.lower() == ".pt":
                    data_save_path = (
                        f"{lazy_root}_{prompt_index:05d}_{formatted_prompt}_lazy_data.pt")
                else:
                    os.makedirs(args.lazy_data_save, exist_ok=True)
                    data_save_path = os.path.join(
                        args.lazy_data_save,
                        f"{prompt_index:05d}_{formatted_prompt}_lazy_data.pt")

        data_dir = os.path.dirname(data_save_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

        # Save the CPU-only training sample first. The video is saved afterwards
        # so the collected data never remains resident on GPU during encoding.
        if rank == 0:
            dataset = {
                "features": features_cpu,
                "targets": targets_cpu,
                "timesteps": timesteps_cpu,
                "step_indices": step_indices_cpu,
                "feature_names": [
                    "t_norm",
                    "input_change_curr",
                    "input_change_prev",
                    "input_mean_curr",
                    "input_mean_prev",
                    "output_change_prev",
                    "residual_norm",
                    "step_progress",
                ],
                "metadata": {
                    "prompt_index": prompt_index,
                    "task": args.task,
                    "size": args.size,
                    "frame_num": args.frame_num,
                    "sampling_steps": args.sample_steps,
                    "sample_solver": args.sample_solver,
                    "sample_shift": args.sample_shift,
                    "guide_scale": args.sample_guide_scale,
                    "seed": sample_seed,
                    "prompt": current_prompt,
                    "source_prompt": raw_prompt,
                    "prompt_file": args.prompt_file,
                    "video_file": video_save_path,
                    "target_definition": (
                        "mean(abs(full_output - (raw_input + previous_cache))) / "
                        "(mean(abs(full_output)) + 1e-8)"
                    ),
                },
            }
            torch.save(dataset, data_save_path)
            logging.info(
                f"Saved {features_cpu.shape[0]} lazy-training samples to "
                f"{data_save_path}")

            logging.info(f"Saving generated video to {video_save_path}")
            save_video(
                tensor=video[None],
                save_file=video_save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

        # Keep all ranks synchronized before destroying this round's mutable
        # state. This matters when Ulysses/FSDP is active.
        if dist.is_initialized():
            dist.barrier()

        del video
        del features_cpu, targets_cpu, timesteps_cpu, step_indices_cpu
        if rank == 0:
            del dataset

        # Explicitly release all per-prompt references. The model itself remains
        # loaded and is reused by the next prompt.
        wan_ti2v.model.collect_features.clear()
        wan_ti2v.model.collect_targets.clear()
        wan_ti2v.model.collect_timesteps.clear()
        wan_ti2v.model.collect_step_indices.clear()
        wan_ti2v.model.previous_raw_input_even = None
        wan_ti2v.model.previous_raw_output_even = None
        wan_ti2v.model.previous_raw_output_odd = None
        wan_ti2v.model.prev_prev_raw_input_even = None
        wan_ti2v.model.prev_prev_raw_output_even = None
        wan_ti2v.model.cache_even = None
        wan_ti2v.model.cache_odd = None
        wan_ti2v.model.k = None
        wan_ti2v.model.cnt = 0
        wan_ti2v.model.accumulated_error_even = 0
        wan_ti2v.model.should_calc_current_pair = True

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.info(
            f"Completed prompt {prompt_index + 1}/{len(prompts)} and cleared "
            "all per-round cache/data state.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished all prompts.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)