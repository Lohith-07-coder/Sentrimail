"""
SentriPredictor — singleton inference interface
-----------------------------------------------
Loads SentriTransformer once at startup and reuses it for every API request.

Output schema (compatible with the existing ai_service.py output):

    {
        "sentiment":           {"label": str, "confidence": float},
        "emotion":             {"label": str, "confidence": float},
        "category":            {"label": str, "confidence": float},
        "issue":               {"label": str, "confidence": float},
        "root_cause_suggestion": str | None,
        "response_draft":      str | None,
        "model_used":          "custom_transformer",
        "model_version":       str,
    }

When no checkpoint is available, ``load()`` raises ``FileNotFoundError``
and the caller should fall back to the existing HuggingFace pipeline path.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# ── Thread-safe singleton lock ─────────────────────────────────────────────────
_lock = threading.Lock()
_predictor: Optional["SentriPredictor"] = None


class SentriPredictor:
    """Stateful inference wrapper around SentriTransformer.

    Args:
        checkpoint_path: Path to a .pt checkpoint file produced by training.
        device:          Override device selection; defaults to CPU.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        from ml.models.transformer_config import TransformerConfig
        from ml.models.multitask_transformer import SentriTransformer
        from ml.models.model_utils import get_device
        from ml.training.checkpoint import load_checkpoint
        from ml.data.preprocessing import get_tokenizer

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Custom model checkpoint not found: {ckpt}\n"
                "Run 'python -m ml.training.train' to create a checkpoint."
            )

        self.device = device or get_device()
        payload = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        self.config = TransformerConfig.from_dict(payload["config"])
        self.model = SentriTransformer(self.config).to(self.device)
        load_checkpoint(str(ckpt), self.model, device=self.device)
        self.model.eval()

        # Load tokenizer (cached globally)
        self.tokenizer = get_tokenizer(self.config.tokenizer_name)
        self.model_version = payload.get("model_version", self.config.model_version)

        logger.info(
            "SentriPredictor ready — checkpoint=%s  version=%s  device=%s",
            ckpt.name, self.model_version, self.device,
        )

    # ── Tokenisation ─────────────────────────────────────────────────────────

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise text and return tensors on the correct device."""
        encoded = self.tokenizer(
            text,
            max_length=self.config.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _decode_ids(self, token_ids: torch.Tensor) -> str:
        """Convert generated token ids to a string, stripping special tokens."""
        ids = token_ids[0].tolist()
        # Remove PAD / BOS / EOS tokens
        specials = {
            self.config.pad_token_id,
            self.config.bos_token_id,
            self.config.eos_token_id,
        }
        ids = [i for i in ids if i not in specials]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    # ── Core prediction ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        text: str,
        generate_text: bool = True,
    ) -> Dict[str, Any]:
        """Run the model and return a structured prediction dict.

        Args:
            text:          Raw complaint text.
            generate_text: If True, also run the decoder to produce a
                           response draft and root cause suggestion.

        Returns:
            Prediction dict matching the SentriMail AI service schema.
        """
        if not text or not text.strip():
            return self._empty_result("Empty input text.")

        try:
            encoded = self._encode(text)
            src_ids  = encoded["input_ids"]
            src_mask = encoded["attention_mask"]
        except Exception as exc:
            logger.warning("Tokenization failed for input: %s", exc)
            return self._empty_result(f"Tokenization error: {exc}")

        try:
            cls_out = self.model.classify(src_ids, src_mask)
        except Exception as exc:
            logger.error("Classification forward pass failed: %s", exc)
            return self._empty_result(f"Model error: {exc}")

        # ── Extract classification predictions ────────────────────────────
        def _top(probs_key: str, label_map: Dict[int, str]) -> Dict[str, Any]:
            probs = cls_out[probs_key][0]            # (num_classes,)
            idx   = probs.argmax().item()
            conf  = round(probs[idx].item(), 4)
            label = label_map.get(idx, "unknown")
            return {"label": label, "confidence": conf}

        sentiment = _top("sentiment_probs", self.config.sentiment_label_map)
        emotion   = _top("emotion_probs",   self.config.emotion_label_map)
        category  = _top("category_probs",  self.config.category_label_map)
        issue     = _top("issue_probs",     self.config.issue_label_map)

        # ── Decoder generation ─────────────────────────────────────────────
        response_draft: Optional[str] = None
        root_cause_suggestion: Optional[str] = None

        if generate_text:
            try:
                gen_ids = self.model.generate_text(src_ids, src_mask)
                generated = self._decode_ids(gen_ids)
                if generated:
                    response_draft = f"[Draft] {generated}"
                    root_cause_suggestion = f"[Model suggestion] {generated}"
            except Exception as exc:
                logger.warning("Text generation failed: %s", exc)

        return {
            "sentiment":               sentiment,
            "emotion":                 emotion,
            "category":                category,
            "issue":                   issue,
            "root_cause_suggestion":   root_cause_suggestion,
            "response_draft":          response_draft,
            "model_used":              "custom_transformer",
            "model_version":           self.model_version,
        }

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        """Return an empty result structure when prediction cannot proceed."""
        logger.warning("SentriPredictor returning empty result: %s", reason)
        return {
            "sentiment":             {"label": None, "confidence": None},
            "emotion":               {"label": None, "confidence": None},
            "category":              {"label": None, "confidence": None},
            "issue":                 {"label": None, "confidence": None},
            "root_cause_suggestion": None,
            "response_draft":        None,
            "model_used":            "custom_transformer",
            "model_version":         None,
        }


# ── Singleton helpers (for application integration) ───────────────────────────

def get_predictor(checkpoint_path: str) -> SentriPredictor:
    """Return or initialise the global singleton SentriPredictor.

    Thread-safe via a module-level lock.  Raises ``FileNotFoundError``
    when the checkpoint does not exist — callers must handle this.
    """
    global _predictor
    with _lock:
        if _predictor is None:
            _predictor = SentriPredictor(checkpoint_path)
    return _predictor


def reset_predictor() -> None:
    """Release the singleton (used in tests)."""
    global _predictor
    with _lock:
        _predictor = None
