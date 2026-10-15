"""
Checkpoint utilities
---------------------
Save and load SentriTransformer checkpoints with full metadata.

Checkpoint format (single .pt file):
    {
        "model_state_dict":     OrderedDict,
        "optimizer_state_dict": OrderedDict   (optional),
        "config":               dict,          (TransformerConfig.to_dict())
        "epoch":                int,
        "global_step":          int,
        "val_loss":             float,
        "model_version":        str,
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: "torch.nn.Module",
    config,
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    val_loss: float,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    tag: str = "best",
) -> Path:
    """Save a checkpoint to ``checkpoint_dir/<tag>.pt``.

    Args:
        model:          SentriTransformer instance.
        config:         TransformerConfig.
        checkpoint_dir: Directory path (created if absent).
        epoch:          Current epoch index (0-based).
        global_step:    Total training steps completed.
        val_loss:       Validation loss at this checkpoint.
        optimizer:      Optionally include optimizer state for resuming.
        tag:            File name stem ('best', 'latest', etc.).

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pt"

    payload = {
        "model_state_dict":  model.state_dict(),
        "config":            config.to_dict(),
        "epoch":             epoch,
        "global_step":       global_step,
        "val_loss":          val_loss,
        "model_version":     config.model_version,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, ckpt_path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d  step=%d  val_loss=%.4f)",
        ckpt_path, epoch, global_step, val_loss,
    )
    return ckpt_path


def load_checkpoint(
    checkpoint_path: str,
    model: "torch.nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, float]:
    """Load weights (and optionally optimizer state) from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt file.
        model:           SentriTransformer instance with the correct architecture.
        optimizer:       If provided, optimizer state is restored.
        device:          Map location; defaults to CPU.

    Returns:
        (epoch, global_step, val_loss) from the checkpoint metadata.

    Raises:
        FileNotFoundError: if the checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    epoch       = payload.get("epoch", 0)
    global_step = payload.get("global_step", 0)
    val_loss    = payload.get("val_loss", float("inf"))
    version     = payload.get("model_version", "unknown")

    logger.info(
        "Checkpoint loaded ← %s  (epoch=%d  step=%d  val_loss=%.4f  version=%s)",
        path, epoch, global_step, val_loss, version,
    )
    return epoch, global_step, val_loss


def checkpoint_exists(checkpoint_dir: str, tag: str = "best") -> bool:
    """Return True if a checkpoint file exists for the given tag."""
    return (Path(checkpoint_dir) / f"{tag}.pt").exists()


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Return the most recently modified .pt file in checkpoint_dir, or None."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None
