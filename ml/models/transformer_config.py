"""
TransformerConfig
-----------------
Dataclass holding every architectural hyper-parameter for the SentriMail
custom Encoder–Decoder Transformer.  All values have safe defaults that
fit comfortably on a CPU-only machine (Railway / local dev).

Constraint enforced: d_model must be divisible by nhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TransformerConfig:
    # ── Vocabulary / tokenizer ───────────────────────────────────────────────
    tokenizer_name: str = "distilbert-base-uncased"
    """HuggingFace tokenizer used for token-id lookup only; no pretrained
    weights are loaded into the neural network layers."""

    vocab_size: int = 30_522          # DistilBERT uncased vocab size
    pad_token_id: int = 0
    bos_token_id: int = 101           # [CLS] in DistilBERT vocab
    eos_token_id: int = 102           # [SEP] in DistilBERT vocab

    # ── Sequence lengths ─────────────────────────────────────────────────────
    max_src_len: int = 128            # encoder (complaint) max tokens
    max_tgt_len: int = 64            # decoder (response / root-cause) max tokens

    # ── Core architecture ────────────────────────────────────────────────────
    d_model: int = 256                # must be divisible by nhead
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # ── Classification head label counts ─────────────────────────────────────
    num_sentiment_classes: int = 3    # POSITIVE, NEUTRAL, NEGATIVE
    num_emotion_classes: int = 7      # anger, fear, sadness, disgust, surprise, joy, neutral
    num_category_classes: int = 7     # billing, technical, delivery, customer_service, product, refund, other
    num_issue_classes: int = 6        # billing, auth, technical, delivery, support, general

    # ── Label mappings ───────────────────────────────────────────────────────
    sentiment_labels: List[str] = field(
        default_factory=lambda: ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    )
    emotion_labels: List[str] = field(
        default_factory=lambda: ["anger", "fear", "sadness", "disgust", "surprise", "joy", "neutral"]
    )
    category_labels: List[str] = field(
        default_factory=lambda: ["billing", "technical", "delivery", "customer_service", "product", "refund", "other"]
    )
    issue_labels: List[str] = field(
        default_factory=lambda: ["billing", "auth", "technical", "delivery", "support", "general"]
    )

    # ── Loss weights ─────────────────────────────────────────────────────────
    sentiment_loss_weight: float = 1.0
    emotion_loss_weight: float = 1.0
    category_loss_weight: float = 1.0
    issue_loss_weight: float = 1.0
    generation_loss_weight: float = 2.0  # decoder CE over non-pad tokens

    # ── Checkpoint ───────────────────────────────────────────────────────────
    checkpoint_dir: str = "ml/checkpoints"
    model_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})."
            )

    # ── Convenience helpers ───────────────────────────────────────────────────
    @property
    def sentiment_label_map(self) -> Dict[int, str]:
        return dict(enumerate(self.sentiment_labels))

    @property
    def emotion_label_map(self) -> Dict[int, str]:
        return dict(enumerate(self.emotion_labels))

    @property
    def category_label_map(self) -> Dict[int, str]:
        return dict(enumerate(self.category_labels))

    @property
    def issue_label_map(self) -> Dict[int, str]:
        return dict(enumerate(self.issue_labels))

    @property
    def inv_sentiment_map(self) -> Dict[str, int]:
        return {v: k for k, v in self.sentiment_label_map.items()}

    @property
    def inv_emotion_map(self) -> Dict[str, int]:
        return {v: k for k, v in self.emotion_label_map.items()}

    @property
    def inv_category_map(self) -> Dict[str, int]:
        return {v: k for k, v in self.category_label_map.items()}

    @property
    def inv_issue_map(self) -> Dict[str, int]:
        return {v: k for k, v in self.issue_label_map.items()}

    def to_dict(self) -> Dict:
        """Serialise to plain dict for checkpoint metadata."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TransformerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
