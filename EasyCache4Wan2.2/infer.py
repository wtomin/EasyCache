# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan2.2 inference with the Version-2 cumulative-risk predictor.

The predictor is evaluated once at the beginning of a cache segment. It
predicts monotonic cumulative risks for the next ``horizon`` CFG pairs. The
policy selects the longest prefix below one threshold, skips exactly that many
pairs, and then performs a full DiT pair to refresh both CFG caches.

The inference contract is intentionally small:

* ``--lazy_ckpt``: full checkpoint produced by train_lazy_cumulative_risk.py
* ``--lazy_threshold``: optional override of the checkpoint threshold

The network architecture, horizon, folded feature normalization and conformal
offsets are read from the checkpoint. No scalar next-y predictor, scheduler
weight, prediction scale, Lp accumulation, OOD guard or cache-age guard is
used.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import warnings
from datetime import datetime
from typing import Any

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import wan
from wan.configs import (
    MAX_AREA_CONFIGS,
    SIZE_CONFIGS,
    SUPPORTED_SIZES,
    WAN_CONFIGS,
)
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool


EXAMPLE_PROMPT = {
    "ti2v-5B": {
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage."
        ),
    },
}

# Sampling-boundary protection is fixed and is not part of the cache policy.
LAZY_WARMUP_STEPS = 7
FEATURE_DIM = 8
EXPECTED_TARGET_CONTRACT = "cumulative_sum_of_local_cache_errors"
EXPECTED_OUTPUT_CONTRACT = "monotonic_cumulative_risk"


class LazyHorizonPredictor(nn.Module):
    """Predict monotonic cumulative cache risk over a future horizon."""

    FEATURE_DIM = FEATURE_DIM

    def __init__(
        self,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        horizon: int = 4,
        initial_prediction: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if initial_prediction <= 0:
            raise ValueError("initial_prediction must be positive")

        self.horizon = int(horizon)
        layers: list[nn.Module] = []
        in_dim = self.FEATURE_DIM
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.horizon))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(math.expm1(initial_prediction)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        increments = F.softplus(self.network(features))
        return torch.cumsum(increments, dim=-1)


def tensor_list_l1_mean(
    tensor_list_a: list[torch.Tensor],
    tensor_list_b: list[torch.Tensor] | None = None,
) -> float:
    """Mean absolute value without materializing one large concatenation."""
    total = 0.0
    count = 0
    if tensor_list_b is None:
        for tensor in tensor_list_a:
            total += tensor.abs().sum().item()
            count += tensor.numel()
    else:
        if len(tensor_list_a) != len(tensor_list_b):
            raise RuntimeError("tensor-list lengths do not match")
        for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
            total += (tensor_a - tensor_b).abs().sum().item()
            count += tensor_a.numel()
    if count == 0:
        raise RuntimeError("cannot compute a mean over an empty tensor list")
    return total / count


def build_horizon_features(
    self: nn.Module,
    raw_input: list[torch.Tensor],
    t_value: float,
) -> torch.Tensor:
    """Reproduce the exact 8-D feature order used by data collection."""
    input_change_curr = tensor_list_l1_mean(
        raw_input,
        self.previous_raw_input_even,
    )
    input_norm_prev = tensor_list_l1_mean(self.previous_raw_input_even)
    input_change_curr_rel = input_change_curr / (input_norm_prev + 1e-8)

    if self.prev_prev_raw_input_even is None:
        input_change_prev_rel = 0.0
    else:
        input_change_prev = tensor_list_l1_mean(
            self.previous_raw_input_even,
            self.prev_prev_raw_input_even,
        )
        input_norm_prev_prev = tensor_list_l1_mean(
            self.prev_prev_raw_input_even
        )
        input_change_prev_rel = (
            input_change_prev / (input_norm_prev_prev + 1e-8)
        )

    input_mean_curr = tensor_list_l1_mean(raw_input)
    input_mean_prev = input_norm_prev

    if self.prev_prev_raw_output_even is None:
        output_change_prev_rel = 0.0
    else:
        output_change_prev = tensor_list_l1_mean(
            self.previous_raw_output_even,
            self.prev_prev_raw_output_even,
        )
        output_norm_prev_prev = tensor_list_l1_mean(
            self.prev_prev_raw_output_even
        )
        output_change_prev_rel = (
            output_change_prev / (output_norm_prev_prev + 1e-8)
        )

    cache_norm = tensor_list_l1_mean(self.cache_even)
    residual_norm = cache_norm / (input_mean_curr + 1e-8)

    return torch.tensor(
        [
            t_value / 1000.0,
            input_change_curr_rel,
            input_change_prev_rel,
            input_mean_curr,
            input_mean_prev,
            output_change_prev_rel,
            residual_norm,
            self.cnt / max(self.num_steps, 1),
        ],
        dtype=torch.float32,
        device=self.patch_embedding.weight.device,
    )


def longest_safe_prefix(
    cumulative_risk: torch.Tensor,
    threshold: float,
) -> int:
    """Return the number of consecutive horizon entries below threshold."""
    if cumulative_risk.ndim != 1:
        raise ValueError("cumulative_risk must be one-dimensional")
    prefix = 0
    for risk in cumulative_risk:
        if float(risk.item()) >= threshold:
            break
        prefix += 1
    return prefix


def lazy_horizon_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    """Wrap the original WanModel forward with horizon cache decisions."""
    if self.model_type == "i2v":
        assert y is not None

    raw_input = [tensor.clone() for tensor in x]
    self.is_even = self.cnt % 2 == 0

    raw_risk = None
    calibrated_risk = None
    planned_prefix = None
    decision_reason = "follow_conditional"

    if self.is_even:
        if torch.is_tensor(t):
            t_value = float(t.flatten()[0].item())
        else:
            t_value = float(t)

        force_full_region = (
            self.cnt < self.ret_steps
            or self.cnt >= self.num_steps - 2
        )
        history_ready = (
            self.previous_raw_input_even is not None
            and self.previous_raw_output_even is not None
            and self.cache_even is not None
            and self.cache_odd is not None
        )

        if force_full_region:
            self.should_calc_current_pair = True
            self.lazy_skip_remaining = 0
            self.lazy_refresh_pending = False
            decision_reason = "warmup_or_final"
        elif self.lazy_skip_remaining > 0:
            if not history_ready:
                raise RuntimeError(
                    "A planned skip cannot run because the CFG caches are missing"
                )
            self.should_calc_current_pair = False
            self.lazy_skip_remaining -= 1
            decision_reason = "planned_prefix"
        elif self.lazy_refresh_pending:
            # The selected prefix has been consumed. Refresh before making a
            # new prediction so two independent plans cannot join together.
            self.should_calc_current_pair = True
            self.lazy_refresh_pending = False
            decision_reason = "planned_refresh"
        elif history_ready:
            features = build_horizon_features(self, raw_input, t_value)
            with torch.no_grad():
                raw_risk = self.lazy_predictor(features.unsqueeze(0))[0]
                calibrated_risk = torch.cummax(
                    raw_risk + self.lazy_calibration_offsets,
                    dim=0,
                ).values

            planned_prefix = longest_safe_prefix(
                calibrated_risk,
                self.lazy_threshold,
            )
            self.lazy_prediction_cnt += 1
            self.lazy_prefix_histogram[planned_prefix] += 1

            if planned_prefix > 0:
                self.should_calc_current_pair = False
                # The current pair consumes the first planned skip.
                self.lazy_skip_remaining = planned_prefix - 1
                self.lazy_refresh_pending = True
                self.lazy_plan_cnt += 1
                decision_reason = "new_safe_prefix"
            else:
                self.should_calc_current_pair = True
                self.lazy_skip_remaining = 0
                self.lazy_refresh_pending = False
                self.lazy_zero_prefix_cnt += 1
                decision_reason = "zero_safe_prefix"
        else:
            self.should_calc_current_pair = True
            self.lazy_skip_remaining = 0
            self.lazy_refresh_pending = False
            decision_reason = "history_not_ready"

        if self.should_calc_current_pair:
            self.calc_pair_cnt += 1
        else:
            self.skip_pair_cnt += 1

        if calibrated_risk is None:
            risk_text = "n/a"
        else:
            risk_text = "[" + ", ".join(
                f"{value:.6f}" for value in calibrated_risk.tolist()
            ) + "]"

        logging.info(
            "[LazyHorizon] step=%02d cnt=%03d | risk=%s | K=%s | "
            "remaining=%d | %s (%s)",
            self.cnt // 2,
            self.cnt,
            risk_text,
            "-" if planned_prefix is None else str(planned_prefix),
            self.lazy_skip_remaining,
            "CALC" if self.should_calc_current_pair else "SKIP",
            decision_reason,
        )

    if self.is_even and not self.should_calc_current_pair:
        if self.cache_even is None:
            raise RuntimeError("conditional cache is missing on a skip")
        self.skip_forward_cnt += 1
        self.cnt += 1
        return [
            (input_tensor + cache).float()
            for input_tensor, cache in zip(raw_input, self.cache_even)
        ]

    if not self.is_even and not self.should_calc_current_pair:
        if self.cache_odd is None:
            raise RuntimeError("unconditional cache is missing on a skip")
        self.skip_forward_cnt += 1
        self.cnt += 1
        return [
            (input_tensor + cache).float()
            for input_tensor, cache in zip(raw_input, self.cache_odd)
        ]

    self.calc_forward_cnt += 1
    # Wan2.2 releases do not expose exactly the same optional forward
    # arguments. In TI2V both values are None, so passing them explicitly can
    # fail on versions whose WanModel.forward only accepts four arguments.
    optional_kwargs = {}
    if clip_fea is not None:
        optional_kwargs["clip_fea"] = clip_fea
    if y is not None:
        optional_kwargs["y"] = y
    output = self.lazy_original_forward(
        x,
        t,
        context,
        seq_len,
        **optional_kwargs,
    )

    if self.is_even:
        self.prev_prev_raw_input_even = self.previous_raw_input_even
        self.previous_raw_input_even = [tensor.clone() for tensor in raw_input]
        self.prev_prev_raw_output_even = self.previous_raw_output_even
        self.previous_raw_output_even = [tensor.clone() for tensor in output]
        self.cache_even = [
            output_tensor - input_tensor
            for output_tensor, input_tensor in zip(output, raw_input)
        ]
        # Every full conditional step starts a new cache segment.
        self.lazy_skip_remaining = 0
        self.lazy_refresh_pending = False
    else:
        self.cache_odd = [
            output_tensor - input_tensor
            for output_tensor, input_tensor in zip(output, raw_input)
        ]

    self.cnt += 1
    return [tensor.float() for tensor in output]


def safe_torch_load(path: str, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("Lazy checkpoint must contain a dictionary")
    return payload


def load_horizon_predictor(
    checkpoint_path: str,
    device: torch.device,
    threshold_override: float | None,
) -> tuple[LazyHorizonPredictor, torch.Tensor, float, dict[str, Any]]:
    payload = safe_torch_load(checkpoint_path, device)

    required_keys = {
        "raw_input_state_dict",
        "horizon",
        "threshold",
        "calibration_offsets",
        "target_contract",
        "model_output_contract",
        "args",
    }
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise RuntimeError(
            "Use lazy_horizon_predictor_full.pt produced by "
            f"train_lazy_cumulative_risk.py; missing keys: {missing}"
        )

    if payload["target_contract"] != EXPECTED_TARGET_CONTRACT:
        raise RuntimeError(
            "Checkpoint target contract mismatch: "
            f"{payload['target_contract']!r}"
        )
    if payload["model_output_contract"] != EXPECTED_OUTPUT_CONTRACT:
        raise RuntimeError(
            "Checkpoint output contract mismatch: "
            f"{payload['model_output_contract']!r}"
        )
    if int(payload.get("feature_dim", FEATURE_DIM)) != FEATURE_DIM:
        raise RuntimeError("Checkpoint feature dimension must be 8")

    training_args = payload["args"]
    if not isinstance(training_args, dict):
        raise RuntimeError("Checkpoint args metadata must be a dictionary")

    horizon = int(payload["horizon"])
    hidden_dim = int(training_args.get("hidden_dim", 128))
    num_hidden_layers = int(training_args.get("num_hidden_layers", 3))
    initial_prediction = float(training_args.get("initial_prediction", 0.01))

    predictor = LazyHorizonPredictor(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        horizon=horizon,
        initial_prediction=initial_prediction,
    ).to(device)
    predictor.load_state_dict(payload["raw_input_state_dict"], strict=True)
    predictor.eval()
    predictor.requires_grad_(False)

    calibration_offsets = torch.as_tensor(
        payload["calibration_offsets"],
        dtype=torch.float32,
        device=device,
    ).flatten()
    if calibration_offsets.shape != (horizon,):
        raise RuntimeError(
            "calibration_offsets must have shape "
            f"[{horizon}], got {tuple(calibration_offsets.shape)}"
        )
    if not torch.isfinite(calibration_offsets).all():
        raise RuntimeError("calibration_offsets contain NaN or Inf")
    if (calibration_offsets < 0).any():
        raise RuntimeError("calibration_offsets must be non-negative")
    calibration_offsets = torch.cummax(calibration_offsets, dim=0).values

    threshold = (
        float(payload["threshold"])
        if threshold_override is None
        else float(threshold_override)
    )
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("--lazy_threshold must be finite and positive")

    return predictor, calibration_offsets, threshold, payload


def validate_args(args: argparse.Namespace) -> None:
    if args.ckpt_dir is None:
        raise ValueError("--ckpt_dir is required")
    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")
    if args.task not in EXAMPLE_PROMPT:
        raise ValueError(
            "This inference file currently supports the ti2v-5B pipeline only"
        )

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    cfg = WAN_CONFIGS[args.task]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(
            f"Unsupported size {args.size} for {args.task}; supported sizes: "
            + ", ".join(SUPPORTED_SIZES[args.task])
        )
    if not args.lazy_ckpt:
        raise ValueError("--lazy_ckpt is required")
    if args.lazy_threshold is not None and (
        not math.isfinite(args.lazy_threshold) or args.lazy_threshold <= 0
    ):
        raise ValueError("--lazy_threshold must be finite and positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 inference with four-step cumulative-risk caching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    wan_group = parser.add_argument_group("Wan generation")
    wan_group.add_argument(
        "--task",
        type=str,
        default="ti2v-5B",
        choices=list(WAN_CONFIGS.keys()),
    )
    wan_group.add_argument(
        "--size",
        type=str,
        default="1280*704",
        choices=list(SIZE_CONFIGS.keys()),
    )
    wan_group.add_argument("--frame_num", type=int, default=None)
    wan_group.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints/Wan2.2-TI2V-5B",
    )
    wan_group.add_argument("--offload_model", type=str2bool, default=None)
    wan_group.add_argument("--ulysses_size", type=int, default=1)
    wan_group.add_argument("--t5_fsdp", action="store_true", default=False)
    wan_group.add_argument("--t5_cpu", action="store_true", default=False)
    wan_group.add_argument("--dit_fsdp", action="store_true", default=False)
    wan_group.add_argument("--save_file", type=str, default=None)
    wan_group.add_argument("--prompt", type=str, default=None)
    wan_group.add_argument("--image", type=str, default=None)
    wan_group.add_argument("--use_prompt_extend", action="store_true")
    wan_group.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
    )
    wan_group.add_argument("--prompt_extend_model", type=str, default=None)
    wan_group.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
    )
    wan_group.add_argument("--base_seed", type=int, default=123)
    wan_group.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
    )
    wan_group.add_argument("--sample_steps", type=int, default=None)
    wan_group.add_argument("--sample_shift", type=float, default=None)
    wan_group.add_argument("--sample_guide_scale", type=float, default=None)
    wan_group.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
    )

    lazy_group = parser.add_argument_group("Lazy cumulative-risk policy")
    lazy_group.add_argument(
        "--lazy_ckpt",
        type=str,
        default=(
            "results/easycache_lazy_cumulative/"
            "lazy_horizon_predictor_full.pt"
        ),
        help="Full checkpoint produced by train_lazy_cumulative_risk.py",
    )
    lazy_group.add_argument(
        "--lazy_threshold",
        type=float,
        default=None,
        help="Override the training threshold; omitted means checkpoint value",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args: argparse.Namespace) -> None:
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 7))
    device_id = local_rank
    init_logging(rank)

    if args.offload_model is None:
        args.offload_model = world_size == 1
        logging.info(
            "offload_model is not specified, set to %s.",
            args.offload_model,
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError("FSDP requires distributed execution")
        if args.ulysses_size > 1:
            raise ValueError("Ulysses sequence parallelism requires distributed execution")

    if args.ulysses_size > 1:
        if args.ulysses_size != world_size:
            raise ValueError("ulysses_size must equal WORLD_SIZE")
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError("cfg.num_heads must be divisible by ulysses_size")

    logging.info("Generation job args: %s", args)
    logging.info("Generation model config: %s", cfg)

    if dist.is_initialized():
        seed_container = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_container, src=0)
        args.base_seed = seed_container[0]

    logging.info("Input prompt: %s", args.prompt)
    image = None
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        logging.info("Input image: %s", args.image)

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=image is not None,
            )
        else:
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=image is not None,
                device=rank,
            )

        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=image,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed,
            )
            if prompt_output.status is False:
                logging.info(
                    "Prompt extension failed: %s; using original prompt.",
                    prompt_output.message,
                )
                extended_prompt = args.prompt
            else:
                extended_prompt = prompt_output.prompt
            prompt_container = [extended_prompt]
        else:
            prompt_container = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(prompt_container, src=0)
        args.prompt = prompt_container[0]
        logging.info("Extended prompt: %s", args.prompt)

    torch_device = torch.device(f"cuda:{device_id}")
    (
        lazy_predictor,
        calibration_offsets,
        lazy_threshold,
        checkpoint_payload,
    ) = load_horizon_predictor(
        args.lazy_ckpt,
        torch_device,
        args.lazy_threshold,
    )

    horizon = lazy_predictor.horizon
    logging.info("Loaded LazyHorizonPredictor from %s", args.lazy_ckpt)
    logging.info("LazyHorizonPredictor architecture:\n%s", lazy_predictor)
    logging.info(
        "LazyHorizonPredictor parameters=%d horizon=%d threshold=%.6f "
        "calibration_offsets=%s",
        sum(parameter.numel() for parameter in lazy_predictor.parameters()),
        horizon,
        lazy_threshold,
        [round(float(value), 6) for value in calibration_offsets.tolist()],
    )
    validation_metrics = checkpoint_payload.get("val_metrics", {})
    logging.info(
        "Checkpoint validation before conformal calibration: "
        "exact=%.4f unsafe_over_skip=%.4f pred_prefix=%.3f true_prefix=%.3f",
        float(validation_metrics.get("prefix_exact_accuracy", float("nan"))),
        float(validation_metrics.get("unsafe_over_skip_rate", float("nan"))),
        float(validation_metrics.get("mean_predicted_prefix", float("nan"))),
        float(validation_metrics.get("mean_true_prefix", float("nan"))),
    )

    if "ti2v" not in args.task:
        raise NotImplementedError("This file currently supports WanTI2V only")

    logging.info("Creating WanTI2V pipeline.")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=args.ulysses_size > 1,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    model = wan_ti2v.model
    # Retain a bound copy of the original DiT forward before replacing the
    # class method. This keeps the cache wrapper independent of Wan internals.
    model.lazy_original_forward = model.__class__.forward.__get__(
        model,
        model.__class__,
    )
    model.__class__.forward = lazy_horizon_forward

    model.cnt = 0
    model.num_steps = args.sample_steps * 2
    model.ret_steps = LAZY_WARMUP_STEPS * 2
    model.should_calc_current_pair = True

    model.previous_raw_input_even = None
    model.prev_prev_raw_input_even = None
    model.previous_raw_output_even = None
    model.prev_prev_raw_output_even = None
    model.cache_even = None
    model.cache_odd = None

    model.lazy_predictor = lazy_predictor
    model.lazy_calibration_offsets = calibration_offsets
    model.lazy_threshold = lazy_threshold
    model.lazy_skip_remaining = 0
    model.lazy_refresh_pending = False

    model.lazy_prediction_cnt = 0
    model.lazy_plan_cnt = 0
    model.lazy_zero_prefix_cnt = 0
    model.lazy_prefix_histogram = [0 for _ in range(horizon + 1)]
    model.skip_pair_cnt = 0
    model.calc_pair_cnt = 0
    model.skip_forward_cnt = 0
    model.calc_forward_cnt = 0

    logging.info(
        "LazyHorizon policy: one threshold=%.6f, horizon=%d, warmup_steps=%d",
        model.lazy_threshold,
        horizon,
        LAZY_WARMUP_STEPS,
    )
    logging.info("Generating video with cumulative-risk horizon strategy ...")
    video = wan_ti2v.generate(
        args.prompt,
        img=image,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )

    total_pairs = model.calc_pair_cnt + model.skip_pair_cnt
    logging.info(
        "LazyHorizon statistics: calc_pairs=%d, skip_pairs=%d, "
        "skip_ratio=%.2f%%, predictions=%d, plans=%d, zero_prefix=%d, "
        "prefix_histogram=%s, calc_forwards=%d, skip_forwards=%d",
        model.calc_pair_cnt,
        model.skip_pair_cnt,
        100.0 * model.skip_pair_cnt / max(total_pairs, 1),
        model.lazy_prediction_cnt,
        model.lazy_plan_cnt,
        model.lazy_zero_prefix_cnt,
        model.lazy_prefix_histogram,
        model.calc_forward_cnt,
        model.skip_forward_cnt,
    )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = (
                args.prompt.replace(" ", "_").replace("/", "_")[:50]
            )
            formatted_size = (
                args.size.replace("*", "x")
                if sys.platform == "win32"
                else args.size
            )
            args.save_file = (
                f"{args.task}_{formatted_size}_{args.ulysses_size}_"
                f"{formatted_prompt}_{formatted_time}.mp4"
            )

        logging.info("Saving generated video to %s", args.save_file)
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")


if __name__ == "__main__":
    generate(parse_args())