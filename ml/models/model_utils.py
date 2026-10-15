"""
Model utilities — parameter counting, device selection, memory profiling.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_info(model: torch.nn.Module, config=None) -> None:
    """Log parameter count and config summary."""
    n = count_parameters(model)
    logger.info("SentriTransformer trainable parameters: %s (%.1fM)", f"{n:,}", n / 1e6)
    if config is not None:
        logger.info(
            "Config: d_model=%d nhead=%d enc_layers=%d dec_layers=%d vocab=%d",
            config.d_model,
            config.nhead,
            config.num_encoder_layers,
            config.num_decoder_layers,
            config.vocab_size,
        )


def measure_inference_latency(
    model: torch.nn.Module,
    src_input_ids: torch.Tensor,
    src_attention_mask: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> float:
    """Return average inference latency in milliseconds over n_runs.

    Runs in eval + no_grad mode.  Excludes warmup iterations from timing.
    """
    model.eval()
    device = next(model.parameters()).device
    src_input_ids = src_input_ids.to(device)
    src_attention_mask = src_attention_mask.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model.classify(src_input_ids, src_attention_mask)

        start = time.perf_counter()
        for _ in range(n_runs):
            model.classify(src_input_ids, src_attention_mask)
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    logger.info("Average classify latency: %.2f ms over %d runs", avg_ms, n_runs)
    return avg_ms


def peak_memory_mb(device: Optional[torch.device] = None) -> float:
    """Return CUDA peak memory usage in MB, or 0.0 on CPU."""
    if device is None or device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
