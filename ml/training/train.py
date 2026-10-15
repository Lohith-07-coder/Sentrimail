"""
Training loop for SentriTransformer
-------------------------------------
Usage:
    python -m ml.training.train [--config ml/config.yaml] [--epochs 10] ...

The training loop supports:
- AdamW optimizer with linear warmup + cosine decay
- Gradient clipping
- Multi-task loss (classification + generation)
- Early stopping on validation loss
- Checkpoint saving (best and latest)
- CPU and CUDA training
- Reproducible random seeds
- Per-epoch training/validation loss logging

It does NOT automatically run during application startup.
Call it from the command line or a dedicated training script.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ml.models.transformer_config import TransformerConfig
from ml.models.multitask_transformer import SentriTransformer
from ml.models.model_utils import get_device, log_model_info
from ml.data.dataset import ComplaintDataset
from ml.data.collator import SentriCollator
from ml.data.preprocessing import preprocess_samples, split_samples
from ml.training.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── LR scheduler: linear warmup then cosine decay ───────────────────────────

def get_linear_warmup_cosine_schedule(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── Single epoch helpers ─────────────────────────────────────────────────────

def _train_epoch(
    model: SentriTransformer,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0

    for step, batch in enumerate(loader):
        # Move all tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            src_input_ids=batch["src_input_ids"],
            src_attention_mask=batch["src_attention_mask"],
            tgt_input_ids=batch.get("tgt_input_ids"),
            tgt_attention_mask=batch.get("tgt_attention_mask"),
            tgt_labels=batch.get("tgt_labels"),
            sentiment_labels=batch["sentiment_labels"],
            emotion_labels=batch["emotion_labels"],
            category_labels=batch["category_labels"],
            issue_labels=batch["issue_labels"],
        )

        loss = outputs.get("total_loss")
        if loss is None or not loss.requires_grad:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_steps += 1

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info("  step %d  loss=%.4f  lr=%.2e", step, loss.item(), lr)

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def _val_epoch(
    model: SentriTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            src_input_ids=batch["src_input_ids"],
            src_attention_mask=batch["src_attention_mask"],
            tgt_input_ids=batch.get("tgt_input_ids"),
            tgt_attention_mask=batch.get("tgt_attention_mask"),
            tgt_labels=batch.get("tgt_labels"),
            sentiment_labels=batch["sentiment_labels"],
            emotion_labels=batch["emotion_labels"],
            category_labels=batch["category_labels"],
            issue_labels=batch["issue_labels"],
        )

        loss = outputs.get("total_loss")
        if loss is not None:
            total_loss += loss.item()
            n_steps += 1

    return total_loss / max(n_steps, 1)


# ── Main training entry point ─────────────────────────────────────────────────

def train(
    config: Optional[TransformerConfig] = None,
    data_dir: str = "data/raw",
    checkpoint_dir: str = "ml/checkpoints",
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    patience: int = 3,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> None:
    """Train SentriTransformer end-to-end.

    Args:
        config:         TransformerConfig (defaults to small CPU config).
        data_dir:       Directory containing raw dataset CSVs.
        checkpoint_dir: Where to save checkpoints.
        num_epochs:     Maximum epochs.
        batch_size:     Training batch size.
        learning_rate:  Peak learning rate for AdamW.
        weight_decay:   AdamW weight decay.
        grad_clip:      Max gradient norm.
        warmup_ratio:   Fraction of total steps used for warmup.
        patience:       Early stopping patience (epochs without improvement).
        seed:           Random seed.
        max_samples:    Cap dataset size for quick experiments (None = use all).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    set_seed(seed)
    device = get_device()

    if config is None:
        config = TransformerConfig()
    config.checkpoint_dir = checkpoint_dir

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading datasets from %s ...", data_dir)
    from ml.dataset.loaders import load_all_datasets

    raw_dir = Path(data_dir)
    flags = {
        "use_local_complaint_dataset": True,
        "use_cfpb":       False,   # 5.5 GB — disable by default for dev runs
        "use_bitext":     True,
        "use_banking77":  True,
        "use_goemotions": True,
        "use_sms_jigsaw": False,
        "use_massive":    False,
    }
    samples = load_all_datasets(raw_dir, flags)

    if not samples:
        logger.error("No samples loaded. Check your data_dir path and dataset files.")
        sys.exit(1)

    if max_samples:
        random.seed(seed)
        samples = random.sample(samples, min(max_samples, len(samples)))
        logger.info("Capped to %d samples.", len(samples))

    # ── Preprocess & split ─────────────────────────────────────────────────
    processed = preprocess_samples(samples, config)
    train_data, val_data, _ = split_samples(processed, seed=seed)   # type: ignore[arg-type]

    # ── Datasets & loaders ─────────────────────────────────────────────────
    collator = SentriCollator(pad_id=config.pad_token_id)
    train_loader = DataLoader(
        ComplaintDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=0,   # keep 0 for Windows compatibility
    )
    val_loader = DataLoader(
        ComplaintDataset(val_data),
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SentriTransformer(config).to(device)
    log_model_info(model, config)

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_warmup_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logger.info("═══ Epoch %d / %d ═══", epoch + 1, num_epochs)
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip)
        val_loss   = _val_epoch(model, val_loader, device)

        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f",
            epoch + 1, train_loss, val_loss,
        )

        # Always save latest checkpoint
        global_step = (epoch + 1) * len(train_loader)
        save_checkpoint(
            model, config, checkpoint_dir,
            epoch=epoch, global_step=global_step, val_loss=val_loss,
            optimizer=optimizer, tag="latest",
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                model, config, checkpoint_dir,
                epoch=epoch, global_step=global_step, val_loss=val_loss,
                tag="best",
            )
            logger.info("  ↑ New best val_loss=%.4f", best_val_loss)
        else:
            epochs_no_improve += 1
            logger.info(
                "  No improvement (%d/%d patience).", epochs_no_improve, patience
            )
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train SentriTransformer")
    parser.add_argument("--data-dir",    default="data/raw",      help="Raw dataset directory")
    parser.add_argument("--ckpt-dir",    default="ml/checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--max-samples", type=int,   default=None,
                        help="Cap dataset size for quick smoke tests")
    parser.add_argument("--d-model",     type=int,   default=256)
    parser.add_argument("--nhead",       type=int,   default=4)
    parser.add_argument("--enc-layers",  type=int,   default=4)
    parser.add_argument("--dec-layers",  type=int,   default=4)
    args = parser.parse_args()

    config = TransformerConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
    )

    train(
        config=config,
        data_dir=args.data_dir,
        checkpoint_dir=args.ckpt_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    _cli()
