#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a horizon predictor from EasyCache lazy-data trajectory files.

Expected input file format:
{
    "features": Tensor[N, 8],
    "targets": Tensor[N],
    "timesteps": Tensor[N],          # optional
    "step_indices": Tensor[N],       # optional but recommended
    "metadata": dict,                # optional
}

For a sample at row i, the model predicts a vector of EasyCache error
increments beginning at the current candidate step:

    [target[i], target[i+1], ..., target[i+horizon-1]]

Future labels are constructed only inside the same trajectory. Invalid samples
or discontinuous step_indices terminate the valid target prefix, so windows
never cross trajectory boundaries or missing timesteps.

Feature normalization is folded into the first Linear layer before saving.
The exported lazy_horizon_predictor.pth therefore accepts the original raw
8-D features directly. The inference-side model must use the same horizon.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


FEATURE_DIM = 8


class LazyHorizonPredictor(nn.Module):
    """Predict non-negative EasyCache error increments over a future horizon."""

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
            raise ValueError("initial_prediction must be positive for Softplus")

        self.horizon = horizon
        layers: list[nn.Module] = []
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
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(math.expm1(initial_prediction)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Shape: [B, horizon]. Each element is a non-negative error increment.
        return F.softplus(self.network(features))


@dataclass
class Trajectory:
    path: Path
    features: torch.Tensor
    targets: torch.Tensor
    step_indices: torch.Tensor
    valid: torch.Tensor
    step_stride: int
    has_explicit_step_indices: bool


@dataclass
class WindowDataset:
    features: torch.Tensor
    targets: torch.Tensor
    masks: torch.Tensor
    trajectory_ids: torch.Tensor
    start_step_indices: torch.Tensor


@dataclass
class Metrics:
    loss: float
    mae: float
    rmse: float
    bias: float
    underestimation_rate: float
    cumulative_mae: float
    cumulative_rmse: float
    prefix_mae: float
    prefix_exact_accuracy: float
    unsafe_over_skip_rate: float
    conservative_rate: float
    mean_predicted_prefix: float
    mean_true_prefix: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a multi-step EasyCache error predictor and evaluate the "
            "maximum safe prefix induced by cumulative predicted error."
        )
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*_lazy_data.pt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/easycache_lazy_horizon",
    )

    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--initial-prediction", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="EasyCache cumulative-error threshold used for prefix metrics.",
    )
    parser.add_argument(
        "--horizon-decay",
        type=float,
        default=0.8,
        help="Loss weight decay for farther future offsets: decay**offset.",
    )
    parser.add_argument(
        "--cumulative-weight",
        type=float,
        default=1.0,
        help="Weight of cumulative-error regression loss.",
    )
    parser.add_argument(
        "--underestimate-weight",
        type=float,
        default=2.0,
        help="Penalty for underestimating cumulative future error.",
    )
    parser.add_argument(
        "--unsafe-weight",
        type=float,
        default=4.0,
        help="Penalty when true cumulative error is unsafe but prediction is safe.",
    )
    parser.add_argument("--huber-beta", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument(
        "--min-valid-horizon",
        type=int,
        default=1,
        help="Minimum number of contiguous supervised offsets required per window.",
    )
    parser.add_argument(
        "--full-horizon-only",
        action="store_true",
        help="Keep only windows with all horizon targets available.",
    )
    parser.add_argument(
        "--expected-step-stride",
        type=int,
        default=0,
        help=(
            "Expected increment of step_indices. Use 0 to infer the most common "
            "positive increment separately for each trajectory."
        ),
    )
    parser.add_argument(
        "--require-step-indices",
        action="store_true",
        help="Reject files without step_indices instead of assuming row continuity.",
    )
    parser.add_argument(
        "--max-target",
        type=float,
        default=None,
        help=(
            "Mark targets above this value invalid. Invalid rows break future "
            "windows; they are never removed and compressed together."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature z-score normalization.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_dir / "train.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
    )


def safe_load(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("file does not contain a dictionary")
    return obj


def infer_step_stride(step_indices: torch.Tensor) -> int:
    if step_indices.numel() < 2:
        return 1
    differences = step_indices[1:] - step_indices[:-1]
    positive = differences[differences > 0]
    if positive.numel() == 0:
        raise ValueError("step_indices contain no positive increment")
    values, counts = torch.unique(positive, return_counts=True)
    return int(values[counts.argmax()].item())


def load_trajectory(
    path: Path,
    max_target: float | None,
    expected_step_stride: int,
    require_step_indices: bool,
) -> Trajectory:
    obj = safe_load(path)
    if "features" not in obj or "targets" not in obj:
        raise KeyError("missing features or targets")

    features = torch.as_tensor(obj["features"], dtype=torch.float32).cpu()
    targets = torch.as_tensor(obj["targets"], dtype=torch.float32).flatten().cpu()

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features must have shape [N,{FEATURE_DIM}], got {tuple(features.shape)}"
        )
    if targets.shape != (features.shape[0],):
        raise ValueError(
            f"targets must have shape [{features.shape[0]}], got {tuple(targets.shape)}"
        )

    has_explicit_step_indices = "step_indices" in obj
    if has_explicit_step_indices:
        step_indices = torch.as_tensor(obj["step_indices"], dtype=torch.long).flatten().cpu()
        if step_indices.shape != (features.shape[0],):
            raise ValueError(
                "step_indices must have shape "
                f"[{features.shape[0]}], got {tuple(step_indices.shape)}"
            )
    else:
        if require_step_indices:
            raise KeyError("step_indices are required but missing")
        step_indices = torch.arange(features.shape[0], dtype=torch.long)

    valid = torch.isfinite(features).all(dim=1)
    valid &= torch.isfinite(targets)
    valid &= targets >= 0
    if max_target is not None:
        valid &= targets <= max_target

    if not valid.any():
        raise ValueError("no valid samples")

    step_stride = (
        expected_step_stride
        if expected_step_stride > 0
        else infer_step_stride(step_indices)
    )
    if step_stride <= 0:
        raise ValueError("step stride must be positive")

    return Trajectory(
        path=path,
        features=features.contiguous(),
        targets=targets.contiguous(),
        step_indices=step_indices.contiguous(),
        valid=valid.contiguous(),
        step_stride=step_stride,
        has_explicit_step_indices=has_explicit_step_indices,
    )


def load_dataset(
    data_dir: Path,
    pattern: str,
    max_target: float | None,
    expected_step_stride: int,
    require_step_indices: bool,
) -> list[Trajectory]:
    files = sorted(data_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} below {data_dir}")

    trajectories: list[Trajectory] = []
    missing_step_indices = 0
    for path in files:
        try:
            trajectory = load_trajectory(
                path,
                max_target,
                expected_step_stride,
                require_step_indices,
            )
        except Exception as exc:
            logging.warning("Skipping %s: %s", path, exc)
            continue

        trajectories.append(trajectory)
        if not trajectory.has_explicit_step_indices:
            missing_step_indices += 1

        logging.info(
            "Loaded %5d/%5d valid samples from %s | stride=%d | "
            "target mean=%.6f max=%.6f",
            int(trajectory.valid.sum().item()),
            trajectory.targets.numel(),
            path.name,
            trajectory.step_stride,
            trajectory.targets[trajectory.valid].mean().item(),
            trajectory.targets[trajectory.valid].max().item(),
        )

    if not trajectories:
        raise RuntimeError("No valid trajectory files were loaded.")
    if missing_step_indices:
        logging.warning(
            "%d trajectory files have no step_indices; row adjacency is assumed. "
            "Use --require-step-indices for strict continuity checking.",
            missing_step_indices,
        )
    return trajectories


def split_by_trajectory(
    trajectories: list[Trajectory],
    val_ratio: float,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0,1).")

    shuffled = list(trajectories)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1 or val_ratio == 0:
        logging.warning(
            "Validation reuses training data because there is only one trajectory "
            "or val_ratio is zero."
        )
        return shuffled, shuffled

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def build_windows(
    trajectories: list[Trajectory],
    horizon: int,
    min_valid_horizon: int,
    full_horizon_only: bool,
) -> WindowDataset:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    trajectory_ids: list[int] = []
    start_step_indices: list[int] = []

    for trajectory_id, trajectory in enumerate(trajectories):
        n = trajectory.targets.numel()
        for start in range(n):
            if not bool(trajectory.valid[start]):
                continue

            target_window = torch.zeros(horizon, dtype=torch.float32)
            mask_window = torch.zeros(horizon, dtype=torch.bool)

            for offset in range(horizon):
                index = start + offset
                if index >= n or not bool(trajectory.valid[index]):
                    break

                if offset > 0:
                    actual_stride = int(
                        trajectory.step_indices[index].item()
                        - trajectory.step_indices[index - 1].item()
                    )
                    if actual_stride != trajectory.step_stride:
                        break

                target_window[offset] = trajectory.targets[index]
                mask_window[offset] = True

            valid_length = int(mask_window.sum().item())
            required = horizon if full_horizon_only else min_valid_horizon
            if valid_length < required:
                continue

            feature_rows.append(trajectory.features[start])
            target_rows.append(target_window)
            mask_rows.append(mask_window)
            trajectory_ids.append(trajectory_id)
            start_step_indices.append(int(trajectory.step_indices[start].item()))

    if not feature_rows:
        raise RuntimeError(
            "No horizon windows were constructed. Check horizon, continuity, "
            "validity filters, and step_indices."
        )

    return WindowDataset(
        features=torch.stack(feature_rows).contiguous(),
        targets=torch.stack(target_rows).contiguous(),
        masks=torch.stack(mask_rows).contiguous(),
        trajectory_ids=torch.tensor(trajectory_ids, dtype=torch.long),
        start_step_indices=torch.tensor(start_step_indices, dtype=torch.long),
    )


def log_window_summary(name: str, dataset: WindowDataset, horizon: int) -> None:
    valid_lengths = dataset.masks.sum(dim=1).float()
    full_rate = (valid_lengths == horizon).float().mean().item()
    logging.info(
        "%s windows=%d | mean valid horizon=%.3f/%d | full horizon rate=%.4f",
        name,
        dataset.features.shape[0],
        valid_lengths.mean().item(),
        horizon,
        full_rate,
    )


def make_loader(
    dataset: WindowDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    normalized = ((dataset.features - mean) / std).contiguous()
    return DataLoader(
        TensorDataset(normalized, dataset.targets, dataset.masks),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def make_raw_loader(
    dataset: WindowDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(dataset.features, dataset.targets, dataset.masks),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def horizon_weights(
    horizon: int,
    decay: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    offsets = torch.arange(horizon, device=device, dtype=dtype)
    return torch.pow(torch.tensor(decay, device=device, dtype=dtype), offsets)


def masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    effective = mask.to(values.dtype) * weights.unsqueeze(0)
    denominator = effective.sum().clamp_min(1.0)
    return (values * effective).sum() / denominator


def loss_function(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    huber_beta: float,
    horizon_decay: float,
    cumulative_weight: float,
    underestimate_weight: float,
    unsafe_weight: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError(
            "prediction, target and mask must have the same [B,H] shape; got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}"
        )

    weights = horizon_weights(
        prediction.shape[1],
        horizon_decay,
        prediction.device,
        prediction.dtype,
    )

    pointwise_regression = F.smooth_l1_loss(
        prediction,
        target,
        beta=huber_beta,
        reduction="none",
    )
    regression = masked_weighted_mean(pointwise_regression, mask, weights)

    predicted_cumulative = torch.cumsum(prediction, dim=1)
    target_cumulative = torch.cumsum(target, dim=1)
    cumulative_regression = F.smooth_l1_loss(
        predicted_cumulative,
        target_cumulative,
        beta=huber_beta,
        reduction="none",
    )
    cumulative_regression = masked_weighted_mean(
        cumulative_regression,
        mask,
        weights,
    )

    cumulative_underestimation = F.relu(
        target_cumulative - predicted_cumulative
    ).square()
    cumulative_underestimation = masked_weighted_mean(
        cumulative_underestimation,
        mask,
        weights,
    )

    true_unsafe = mask & (target_cumulative >= threshold)
    if true_unsafe.any():
        unsafe_margin = F.relu(
            threshold - predicted_cumulative
        ).square()
        unsafe = masked_weighted_mean(
            unsafe_margin,
            true_unsafe,
            weights,
        )
    else:
        unsafe = prediction.new_zeros(())

    return (
        regression
        + cumulative_weight * cumulative_regression
        + underestimate_weight * cumulative_underestimation
        + unsafe_weight * unsafe
    )


def safe_prefix_length(
    cumulative_error: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return the length of the longest valid prefix strictly below threshold."""
    safe = mask & (cumulative_error < threshold)
    # Once a position is unsafe/invalid, all later positions are excluded.
    return safe.to(torch.long).cumprod(dim=1).sum(dim=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    args: argparse.Namespace,
) -> Metrics:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for feature_batch, target_batch, mask_batch in loader:
        feature_batch = feature_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)

        prediction = model(feature_batch)
        loss = loss_function(
            prediction,
            target_batch,
            mask_batch,
            args.threshold,
            args.huber_beta,
            args.horizon_decay,
            args.cumulative_weight,
            args.underestimate_weight,
            args.unsafe_weight,
        )

        count = target_batch.shape[0]
        total_loss += loss.item() * count
        total_samples += count
        predictions.append(prediction.cpu())
        targets.append(target_batch.cpu())
        masks.append(mask_batch.cpu())

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    valid = mask.to(torch.float32)
    valid_count = valid.sum().clamp_min(1.0)

    error = prediction - target
    mae = (error.abs() * valid).sum() / valid_count
    rmse = ((error.square() * valid).sum() / valid_count).sqrt()
    bias = (error * valid).sum() / valid_count
    underestimation_rate = (
        ((prediction < target) & mask).float().sum() / valid_count
    )

    predicted_cumulative = torch.cumsum(prediction, dim=1)
    target_cumulative = torch.cumsum(target, dim=1)
    cumulative_error = predicted_cumulative - target_cumulative
    cumulative_mae = (cumulative_error.abs() * valid).sum() / valid_count
    cumulative_rmse = (
        (cumulative_error.square() * valid).sum() / valid_count
    ).sqrt()

    predicted_prefix = safe_prefix_length(
        predicted_cumulative,
        mask,
        threshold,
    )
    true_prefix = safe_prefix_length(
        target_cumulative,
        mask,
        threshold,
    )
    prefix_error = predicted_prefix.to(torch.float32) - true_prefix.to(torch.float32)

    return Metrics(
        loss=total_loss / max(total_samples, 1),
        mae=mae.item(),
        rmse=rmse.item(),
        bias=bias.item(),
        underestimation_rate=underestimation_rate.item(),
        cumulative_mae=cumulative_mae.item(),
        cumulative_rmse=cumulative_rmse.item(),
        prefix_mae=prefix_error.abs().mean().item(),
        prefix_exact_accuracy=(predicted_prefix == true_prefix).float().mean().item(),
        unsafe_over_skip_rate=(predicted_prefix > true_prefix).float().mean().item(),
        conservative_rate=(predicted_prefix < true_prefix).float().mean().item(),
        mean_predicted_prefix=predicted_prefix.float().mean().item(),
        mean_true_prefix=true_prefix.float().mean().item(),
    )


@torch.no_grad()
def per_horizon_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list[float]]:
    model.eval()
    absolute_error_sum: torch.Tensor | None = None
    squared_error_sum: torch.Tensor | None = None
    valid_count: torch.Tensor | None = None

    for feature_batch, target_batch, mask_batch in loader:
        prediction = model(feature_batch.to(device, non_blocking=True)).cpu()
        error = prediction - target_batch
        mask_float = mask_batch.to(torch.float32)

        batch_abs = (error.abs() * mask_float).sum(dim=0)
        batch_sq = (error.square() * mask_float).sum(dim=0)
        batch_count = mask_float.sum(dim=0)

        if absolute_error_sum is None:
            absolute_error_sum = batch_abs
            squared_error_sum = batch_sq
            valid_count = batch_count
        else:
            absolute_error_sum += batch_abs
            squared_error_sum += batch_sq
            valid_count += batch_count

    assert absolute_error_sum is not None
    assert squared_error_sum is not None
    assert valid_count is not None
    denominator = valid_count.clamp_min(1.0)
    return {
        "mae": (absolute_error_sum / denominator).tolist(),
        "rmse": (squared_error_sum / denominator).sqrt().tolist(),
        "valid_count": valid_count.to(torch.long).tolist(),
    }



@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect model predictions, targets and masks on CPU."""
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for feature_batch, target_batch, mask_batch in loader:
        prediction = model(feature_batch.to(device, non_blocking=True)).cpu()
        predictions.append(prediction)
        targets.append(target_batch.cpu())
        masks.append(mask_batch.cpu())

    if not predictions:
        raise RuntimeError("Cannot visualize an empty validation loader.")

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
    )


def _load_history_rows(history_path: Path) -> list[dict[str, float]]:
    if not history_path.exists():
        return []

    rows: list[dict[str, float]] = []
    with history_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None or value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def generate_visualizations(
    output_dir: Path,
    history_path: Path,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict[str, str]:
    """Generate post-training diagnostic plots without changing training logic."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logging.warning("Skipping visualization because matplotlib is unavailable: %s", exc)
        return {}

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    history_rows = _load_history_rows(history_path)
    if history_rows:
        epochs = [row["epoch"] for row in history_rows if "epoch" in row]

        if epochs and all("train_loss" in row and "val_loss" in row for row in history_rows):
            path = visualization_dir / "training_loss.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, [row["train_loss"] for row in history_rows], label="Train loss")
            ax.plot(epochs, [row["val_loss"] for row in history_rows], label="Validation loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and validation loss")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated["training_loss"] = str(path)

        metric_keys = [
            ("val_prefix_exact_accuracy", "Exact prefix accuracy"),
            ("val_unsafe_over_skip_rate", "Unsafe over-skip rate"),
            ("val_conservative_rate", "Conservative rate"),
        ]
        if epochs and all(all(key in row for key, _ in metric_keys) for row in history_rows):
            path = visualization_dir / "validation_prefix_metrics.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            for key, label in metric_keys:
                ax.plot(epochs, [row[key] for row in history_rows], label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rate")
            ax.set_ylim(bottom=0.0)
            ax.set_title("Validation prefix-decision metrics")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated["validation_prefix_metrics"] = str(path)

    predicted_cumulative = torch.cumsum(prediction, dim=1)
    target_cumulative = torch.cumsum(target, dim=1)
    predicted_prefix = safe_prefix_length(predicted_cumulative, mask, threshold)
    true_prefix = safe_prefix_length(target_cumulative, mask, threshold)

    max_prefix = int(max(true_prefix.max().item(), predicted_prefix.max().item(), 1))

    confusion_size = max_prefix + 1
    confusion = torch.zeros((confusion_size, confusion_size), dtype=torch.long)
    for true_value, predicted_value in zip(true_prefix.tolist(), predicted_prefix.tolist()):
        confusion[true_value, predicted_value] += 1

    path = visualization_dir / "prefix_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(confusion.numpy(), aspect="auto")
    ax.set_xticks(range(confusion_size))
    ax.set_yticks(range(confusion_size))
    ax.set_xlabel("Predicted safe-prefix length")
    ax.set_ylabel("True safe-prefix length")
    ax.set_title("Safe-prefix confusion matrix")
    fig.colorbar(image, ax=ax, label="Sample count")
    for row in range(confusion_size):
        for column in range(confusion_size):
            value = int(confusion[row, column].item())
            if value > 0:
                ax.text(column, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated["prefix_confusion_matrix"] = str(path)

    logging.info("Generated %d visualization files in %s", len(generated), visualization_dir)
    return generated

def fold_normalization(
    model: LazyHorizonPredictor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    weight = state["network.0.weight"]
    bias = state["network.0.bias"]
    mean = mean.to(weight)
    std = std.to(weight)

    state["network.0.weight"] = weight / std.unsqueeze(0)
    state["network.0.bias"] = (
        bias - (weight * (mean / std).unsqueeze(0)).sum(dim=1)
    )
    return state


@torch.no_grad()
def verify_fold(
    model: LazyHorizonPredictor,
    folded_state: dict[str, torch.Tensor],
    raw_features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    device = next(model.parameters()).device
    reference = model(((raw_features - mean) / std).to(device)).cpu()

    folded = LazyHorizonPredictor(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        horizon=args.horizon,
        initial_prediction=args.initial_prediction,
    )
    folded.load_state_dict(folded_state)
    folded.eval()
    test = folded(raw_features.cpu())
    return (reference - test).abs().max().item()


def append_history(
    path: Path,
    epoch: int,
    lr: float,
    train_metrics: Metrics,
    val_metrics: Metrics,
) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(
                ["epoch", "lr"]
                + [f"train_{key}" for key in asdict(train_metrics)]
                + [f"val_{key}" for key in asdict(val_metrics)]
            )
        writer.writerow(
            [epoch, lr]
            + list(asdict(train_metrics).values())
            + list(asdict(val_metrics).values())
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError("epochs, batch-size and lr must be positive")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 1 <= args.min_valid_horizon <= args.horizon:
        raise ValueError("min-valid-horizon must be in [1, horizon]")
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")
    if not 0 < args.horizon_decay <= 1:
        raise ValueError("horizon-decay must be in (0,1]")
    if args.expected_step_stride < 0:
        raise ValueError("expected-step-stride must be non-negative")
    if args.huber_beta <= 0:
        raise ValueError("huber-beta must be positive")
    if min(
        args.cumulative_weight,
        args.underestimate_weight,
        args.unsafe_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    logging.info("Using device: %s", device)
    logging.info(
        "Horizon configuration: horizon=%d, min_valid_horizon=%d, "
        "full_horizon_only=%s",
        args.horizon,
        args.min_valid_horizon,
        args.full_horizon_only,
    )

    trajectories = load_dataset(
        Path(args.data_dir),
        args.pattern,
        args.max_target,
        args.expected_step_stride,
        args.require_step_indices,
    )
    train_trajectories, val_trajectories = split_by_trajectory(
        trajectories,
        args.val_ratio,
        args.seed,
    )

    train_dataset = build_windows(
        train_trajectories,
        args.horizon,
        args.min_valid_horizon,
        args.full_horizon_only,
    )
    val_dataset = build_windows(
        val_trajectories,
        args.horizon,
        args.min_valid_horizon,
        args.full_horizon_only,
    )

    logging.info(
        "Split: train=%d trajectories, val=%d trajectories",
        len(train_trajectories),
        len(val_trajectories),
    )
    log_window_summary("Train", train_dataset, args.horizon)
    log_window_summary("Val", val_dataset, args.horizon)

    train_valid_targets = train_dataset.targets[train_dataset.masks]
    val_valid_targets = val_dataset.targets[val_dataset.masks]
    logging.info(
        "Target increments: train mean=%.6f std=%.6f min=%.6f max=%.6f | "
        "val mean=%.6f max=%.6f",
        train_valid_targets.mean().item(),
        train_valid_targets.std(unbiased=False).item(),
        train_valid_targets.min().item(),
        train_valid_targets.max().item(),
        val_valid_targets.mean().item(),
        val_valid_targets.max().item(),
    )

    if args.no_normalize:
        feature_mean = torch.zeros(FEATURE_DIM)
        feature_std = torch.ones(FEATURE_DIM)
    else:
        feature_mean = train_dataset.features.mean(dim=0)
        feature_std = train_dataset.features.std(dim=0, unbiased=False)
        feature_std = torch.where(
            feature_std < 1e-6,
            torch.ones_like(feature_std),
            feature_std,
        )

    train_loader = make_loader(
        train_dataset,
        feature_mean,
        feature_std,
        args.batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        feature_mean,
        feature_std,
        args.batch_size,
        False,
        args.num_workers,
    )

    model = LazyHorizonPredictor(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        horizon=args.horizon,
        initial_prediction=args.initial_prediction,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, args.patience // 4) if args.patience > 0 else 10,
        min_lr=1e-6,
    )

    logging.info(
        "Trainable parameters: %d",
        sum(parameter.numel() for parameter in model.parameters()),
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history_path = output_dir / "history.csv"
    checkpoint_path = output_dir / "lazy_horizon_predictor.pth"
    full_checkpoint_path = output_dir / "lazy_horizon_predictor_full.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for feature_batch, target_batch, mask_batch in train_loader:
            feature_batch = feature_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)

            prediction = model(feature_batch)
            loss = loss_function(
                prediction,
                target_batch,
                mask_batch,
                args.threshold,
                args.huber_beta,
                args.horizon_decay,
                args.cumulative_weight,
                args.underestimate_weight,
                args.unsafe_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            optimizer.step()

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            args.threshold,
            args,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.threshold,
            args,
        )
        scheduler.step(val_metrics.loss)
        current_lr = optimizer.param_groups[0]["lr"]

        append_history(
            history_path,
            epoch,
            current_lr,
            train_metrics,
            val_metrics,
        )

        if epoch == 1 or epoch % 10 == 0:
            logging.info(
                "Epoch %04d | lr=%.2e | train loss=%.6f mae=%.6f "
                "prefix_mae=%.4f | val loss=%.6f mae=%.6f cum_mae=%.6f "
                "prefix_mae=%.4f exact=%.4f unsafe_over_skip=%.4f "
                "pred_prefix=%.3f true_prefix=%.3f",
                epoch,
                current_lr,
                train_metrics.loss,
                train_metrics.mae,
                train_metrics.prefix_mae,
                val_metrics.loss,
                val_metrics.mae,
                val_metrics.cumulative_mae,
                val_metrics.prefix_mae,
                val_metrics.prefix_exact_accuracy,
                val_metrics.unsafe_over_skip_rate,
                val_metrics.mean_predicted_prefix,
                val_metrics.mean_true_prefix,
            )

        if val_metrics.loss < best_val_loss - 1e-8:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            stale_epochs = 0

            raw_state = fold_normalization(
                model,
                feature_mean,
                feature_std,
            )
            verify_count = min(32, val_dataset.features.shape[0])
            fold_error = verify_fold(
                model,
                raw_state,
                val_dataset.features[:verify_count],
                feature_mean,
                feature_std,
                args,
            )
            if fold_error > 1e-5:
                raise RuntimeError(
                    f"Feature normalization folding failed: {fold_error}"
                )

            # Pure state_dict for direct inference-side load_state_dict(...).
            torch.save(raw_state, checkpoint_path)

            torch.save(
                {
                    "raw_input_state_dict": raw_state,
                    "normalized_input_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "feature_dim": FEATURE_DIM,
                    "horizon": args.horizon,
                    "threshold": args.threshold,
                    "train_metrics": asdict(train_metrics),
                    "val_metrics": asdict(val_metrics),
                    "args": vars(args),
                    "train_files": [str(item.path) for item in train_trajectories],
                    "val_files": [str(item.path) for item in val_trajectories],
                },
                full_checkpoint_path,
            )
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            logging.info(
                "Early stopping at epoch %d; best epoch=%d val_loss=%.6f",
                epoch,
                best_epoch,
                best_val_loss,
            )
            break

    try:
        best_state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        best_state = torch.load(checkpoint_path, map_location="cpu")

    final_model = LazyHorizonPredictor(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        horizon=args.horizon,
        initial_prediction=args.initial_prediction,
    ).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    raw_val_loader = make_raw_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
    )
    final_metrics = evaluate(
        final_model,
        raw_val_loader,
        device,
        args.threshold,
        args,
    )
    final_per_horizon = per_horizon_metrics(
        final_model,
        raw_val_loader,
        device,
    )

    final_prediction, final_target, final_mask = collect_predictions(
        final_model,
        raw_val_loader,
        device,
    )
    visualization_paths = generate_visualizations(
        output_dir=output_dir,
        history_path=history_path,
        prediction=final_prediction,
        target=final_target,
        mask=final_mask,
        threshold=args.threshold,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_raw_input_val_metrics": asdict(final_metrics),
        "per_horizon_val_metrics": final_per_horizon,
        "trajectory_files": len(trajectories),
        "train_trajectories": len(train_trajectories),
        "val_trajectories": len(val_trajectories),
        "train_windows": int(train_dataset.features.shape[0]),
        "val_windows": int(val_dataset.features.shape[0]),
        "horizon": args.horizon,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "threshold": args.threshold,
        "checkpoint": str(checkpoint_path),
        "full_checkpoint": str(full_checkpoint_path),
        "visualizations": visualization_paths,
    }
    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logging.info(
        "Finished | best epoch=%d | val MAE=%.6f cumulative_MAE=%.6f "
        "prefix_MAE=%.4f exact=%.4f unsafe_over_skip=%.4f",
        best_epoch,
        final_metrics.mae,
        final_metrics.cumulative_mae,
        final_metrics.prefix_mae,
        final_metrics.prefix_exact_accuracy,
        final_metrics.unsafe_over_skip_rate,
    )
    logging.info("Inference checkpoint: %s", checkpoint_path)
    logging.info("Full checkpoint: %s", full_checkpoint_path)
    logging.info(
        "Inference model must use horizon=%d; the old scalar predictor class is "
        "not shape-compatible with this checkpoint.",
        args.horizon,
    )


if __name__ == "__main__":
    main()