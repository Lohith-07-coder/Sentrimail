"""
ComplaintDataset — torch.utils.data.Dataset wrapper
----------------------------------------------------
Wraps preprocessed feature dicts for use with PyTorch DataLoader.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ComplaintDataset(Dataset):
    """PyTorch Dataset over preprocessed SentriMail samples.

    Args:
        features: List of dicts produced by :func:`~ml.data.preprocessing.preprocess_samples`.
    """

    def __init__(self, features: List[Dict]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict:
        """Return a single feature dict.  Tensors are created in the collator
        to avoid large tensor allocations during dataset construction."""
        return self.features[idx]
