"""
Model evaluation — classification metrics + generation metrics
--------------------------------------------------------------
Usage:
    python -m ml.training.evaluate --ckpt ml/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Classification metrics ────────────────────────────────────────────────────

def classification_report(
    all_preds: List[int],
    all_labels: List[int],
    label_names: List[str],
    task_name: str,
) -> Dict:
    """Compute per-class precision, recall, F1, and macro-F1.

    Only samples with labels >= 0 are evaluated.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report as sk_report,
            confusion_matrix,
        )
    except ImportError:
        logger.warning("scikit-learn not installed; skipping classification metrics.")
        return {}

    valid = [(p, l) for p, l in zip(all_preds, all_labels) if l >= 0]
    if not valid:
        logger.warning("No valid labels for task '%s'.", task_name)
        return {}

    preds, labels = zip(*valid)
    acc = accuracy_score(labels, preds)
    report = sk_report(labels, preds, target_names=label_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    logger.info(
        "%s — accuracy=%.4f  macro_f1=%.4f",
        task_name, acc, report["macro avg"]["f1-score"],
    )
    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


# ── Generation metrics ────────────────────────────────────────────────────────

def generation_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict:
    """Compute ROUGE-L and token-level validation perplexity proxy."""
    results: Dict = {}

    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        for key in ["rouge1", "rouge2", "rougeL"]:
            f1_scores = [s[key].fmeasure for s in scores]
            results[key] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        logger.info(
            "ROUGE — R1=%.4f  R2=%.4f  RL=%.4f",
            results.get("rouge1", 0), results.get("rouge2", 0), results.get("rougeL", 0),
        )
    except ImportError:
        logger.warning("rouge_score not installed; skipping ROUGE metrics.")

    return results


# ── Full evaluation run ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_dir: str = "data/raw",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """Run full evaluation on the test split.

    Loads the checkpoint, reconstructs the model, runs classification and
    optionally generation over the held-out test set.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from ml.models.transformer_config import TransformerConfig
    from ml.models.multitask_transformer import SentriTransformer
    from ml.models.model_utils import get_device
    from ml.data.preprocessing import preprocess_samples, split_samples
    from ml.data.dataset import ComplaintDataset
    from ml.data.collator import SentriCollator
    from ml.training.checkpoint import load_checkpoint
    from ml.dataset.loaders import load_all_datasets

    import random, torch
    random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    # Load checkpoint and reconstruct config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = SentriTransformer(config).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # Load test set
    raw_dir = Path(data_dir)
    flags = {
        "use_local_complaint_dataset": True,
        "use_cfpb": False, "use_bitext": True, "use_banking77": True,
        "use_goemotions": True, "use_sms_jigsaw": False, "use_massive": False,
    }
    samples = load_all_datasets(raw_dir, flags)
    if max_samples:
        samples = random.sample(samples, min(max_samples, len(samples)))

    processed = preprocess_samples(samples, config)
    _, _, test_data = split_samples(processed, seed=seed)   # type: ignore[arg-type]

    if not test_data:
        logger.warning("Test set is empty.")
        return {}

    collator = SentriCollator(pad_id=config.pad_token_id)
    test_loader = DataLoader(ComplaintDataset(test_data), batch_size=batch_size,
                             shuffle=False, collate_fn=collator)

    # Collect predictions
    all_sent_preds,  all_sent_lbls  = [], []
    all_emot_preds,  all_emot_lbls  = [], []
    all_cat_preds,   all_cat_lbls   = [], []
    all_iss_preds,   all_iss_lbls   = [], []
    total_gen_loss, gen_steps = 0.0, 0

    for batch in test_loader:
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

        for logits_key, preds_list, lbls_list, lbl_tensor in [
            ("sentiment_logits", all_sent_preds, all_sent_lbls, batch["sentiment_labels"]),
            ("emotion_logits",   all_emot_preds, all_emot_lbls, batch["emotion_labels"]),
            ("category_logits",  all_cat_preds,  all_cat_lbls,  batch["category_labels"]),
            ("issue_logits",     all_iss_preds,  all_iss_lbls,  batch["issue_labels"]),
        ]:
            preds_list.extend(outputs[logits_key].argmax(dim=-1).cpu().tolist())
            lbls_list.extend(lbl_tensor.cpu().tolist())

        if "generation_loss" in outputs:
            total_gen_loss += outputs["generation_loss"].item()
            gen_steps += 1

    results: Dict = {}
    results["sentiment"]  = classification_report(all_sent_preds, all_sent_lbls,  config.sentiment_labels,  "Sentiment")
    results["emotion"]    = classification_report(all_emot_preds, all_emot_lbls,  config.emotion_labels,    "Emotion")
    results["category"]   = classification_report(all_cat_preds,  all_cat_lbls,   config.category_labels,   "Category")
    results["issue"]      = classification_report(all_iss_preds,  all_iss_lbls,   config.issue_labels,      "Issue")
    if gen_steps:
        results["avg_generation_loss"] = total_gen_loss / gen_steps
        import math
        results["perplexity"] = math.exp(min(total_gen_loss / gen_steps, 100))
        logger.info("Generation avg_loss=%.4f  perplexity=%.2f",
                    results["avg_generation_loss"], results["perplexity"])

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Evaluate SentriTransformer")
    parser.add_argument("--ckpt",        required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data-dir",    default="data/raw")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    results = evaluate(
        checkpoint_path=args.ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
