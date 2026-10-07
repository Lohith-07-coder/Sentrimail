"""
scripts/generate_silver_labels.py
----------------------------------
Fully local, offline silver root-cause label generator for SentriMail.
No external API keys or network calls are permitted.

Features
--------
1. Configurable template map  (category, issue) → root-cause strings.
   Each bucket carries several paraphrase variants; one is chosen at
   random per sample to maximise training diversity.

2. --use-local-llm flag: loads a previously downloaded local Hugging Face
   text2text / causal instruct model exactly once.  It always uses
   ``local_files_only=True`` and therefore never downloads a model.

3. Exports exactly 200 random samples by default.  It fails clearly when the
   selected source has fewer samples instead of silently creating a smaller
   review set.

4. Template and LLM labels use separate method-aware caches so re-runs are
   incremental without mixing generation methods.

Usage
-----
  # Template-only (zero dependencies beyond pandas):
  python scripts/generate_silver_labels.py

  # With a model already present in the local Hugging Face cache:
  python scripts/generate_silver_labels.py --use-local-llm \
      --local-model /path/to/local-model

  # With a locally-downloaded Qwen instruct model:
  python scripts/generate_silver_labels.py --use-local-llm \\
      --local-model Qwen/Qwen2.5-1.5B-Instruct

  # Override number of review samples and output CSV:
  python scripts/generate_silver_labels.py --review-samples 100 \\
      --output-csv data/my_review.csv

Raw datasets
------------
Place any of the supported CSV/JSON files in  data/raw/  (downloaded
manually from Kaggle or the source website — no API key needed):

  data/raw/cfpb_complaints.csv          CFPB Financial Complaints
  data/raw/bitext_customer_support.csv  Bitext Customer Support
  data/raw/banking77.csv                Banking77 Intent Dataset
  data/raw/goemotions.csv               GoEmotions Emotion Dataset
  data/raw/spam_abuse.csv               SMS Spam / Jigsaw Toxic
  data/raw/massive_multilingual.json    Amazon MASSIVE Multilingual

If no files are found the script generates labels for an internal
bootstrap set that covers all (category, issue) buckets.
"""

import sys
import json
import random
import argparse
import logging
import textwrap
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TEMPLATE_CACHE_FILE = PROJECT_ROOT / "data" / "silver_labels_template_cache.json"
LLM_CACHE_FILE = PROJECT_ROOT / "data" / "silver_labels_llm_cache.json"
DEFAULT_CSV = PROJECT_ROOT / "data" / "silver_labels_sample_review.csv"

# ── Template map: (category, issue) → [paraphrase variants] ──────────────────
# Each key is a (category, issue) tuple matching the UnifiedSample schema.
# Add or extend any bucket freely — more variants = better training diversity.

TEMPLATE_VARIANTS: Dict[Tuple[str, str], List[str]] = {

    # ── BILLING ──────────────────────────────────────────────────────────────
    ("billing", "billing"): [
        "Likely caused by an unexpected subscription auto-renewal that was not clearly communicated in advance.",
        "Attributable to an invoicing calculation error in the monthly billing cycle where discounts were not applied.",
        "Caused by a payment-gateway processing delay that resulted in a duplicate charge being recorded.",
        "Root cause appears to be a mismatch between the quoted promotional rate and the rate applied at renewal.",
        "Likely triggered by a back-end billing-service race condition that posted two transactions for a single authorisation.",
    ],
    ("billing", "refund"): [
        "Likely caused by the bank's clearing-house hold exceeding the merchant's standard reversal window.",
        "Root cause appears to be the refund request being routed to a decommissioned payment processor endpoint.",
        "Caused by missing refund-authorisation confirmation, leaving the transaction in a perpetual 'pending' state.",
        "Attributable to a currency-conversion rounding error that blocked automated refund reconciliation.",
        "Likely due to the customer's card-issuer declining the credit posting because the original charge has aged out.",
    ],
    ("billing", "general"): [
        "Likely caused by an unresolved billing-profile discrepancy between the CRM and the payment system.",
        "Root cause appears to be unclear billing-schedule notifications that led to an unexpected charge.",
        "Attributable to an unsupported payment method silently failing at the point-of-sale without an error message.",
    ],

    # ── TECHNICAL ─────────────────────────────────────────────────────────────
    ("technical", "auth"): [
        "Likely caused by a multi-factor authentication token expiring before the user completed the verification step.",
        "Root cause appears to be an OAuth 2.0 state-token mismatch triggered by a browser cookie-partition policy update.",
        "Caused by a password-reset link that expired prematurely due to a UTC timezone parsing inconsistency in the token service.",
        "Attributable to a session-cookie SameSite attribute mismatch after a recent platform deployment.",
        "Likely triggered by an account-lockout policy activating after three failed login attempts within 60 seconds.",
        "Root cause appears to be a stale LDAP directory cache entry that prevented the updated credential from being accepted.",
    ],
    ("technical", "technical"): [
        "Likely caused by a backend microservice experiencing connection-pool exhaustion under peak concurrency.",
        "Root cause appears to be an unhandled null-pointer exception in the order-processing API causing HTTP 500 responses.",
        "Caused by a database cluster failover that left read replicas returning stale data for 8–12 minutes.",
        "Attributable to memory-leak accumulation in the long-running worker process eventually triggering an OOM restart.",
        "Likely triggered by a misconfigured CDN cache TTL serving a stale payload to the client.",
        "Root cause appears to be a dependency-version conflict introduced in the latest deployment that broke a core library.",
        "Caused by a network-latency spike between availability zones degrading API response times above acceptable thresholds.",
    ],
    ("technical", "general"): [
        "Likely caused by an incompatibility between the client application version and the latest server API contract.",
        "Root cause appears to be an edge-case input that triggered an unhandled exception path in the business-logic layer.",
        "Attributable to a third-party integration returning an unexpected response schema after a silent upstream change.",
    ],

    # ── DELIVERY ──────────────────────────────────────────────────────────────
    ("delivery", "delivery"): [
        "Likely caused by a logistics-carrier transit delay due to adverse weather conditions affecting regional routing.",
        "Root cause appears to be a warehouse dispatch bottleneck from peak-season volume overload.",
        "Caused by an address-verification mismatch that routed the parcel to an incorrect distribution facility.",
        "Attributable to a package scan missed at the sortation centre, producing a 'label created but not dispatched' status.",
        "Likely triggered by a courier app GPS error that marked the delivery as completed before the item was handed over.",
    ],
    ("delivery", "general"): [
        "Likely caused by a fulfillment-centre inventory discrepancy between the stock-management system and physical stock.",
        "Root cause appears to be a return-to-sender rule triggered by an undeliverable-address flag set in error.",
        "Attributable to a third-party drop-shipper failing to confirm dispatch within the agreed SLA window.",
    ],

    # ── CUSTOMER SERVICE ──────────────────────────────────────────────────────
    ("customer_service", "support"): [
        "Likely caused by support-queue capacity saturation during a concurrent incident, leading to delayed agent assignment.",
        "Root cause appears to be a missing specialist-escalation path in the ticket-routing configuration.",
        "Caused by agent knowledge-base information being out of date after a recent product change, producing incorrect guidance.",
        "Attributable to a multi-channel ticket-duplication issue that split the conversation context across two separate threads.",
        "Likely triggered by an auto-responder misclassifying the ticket as resolved before the customer confirmed satisfaction.",
    ],
    ("customer_service", "general"): [
        "Root cause appears to be inadequate agent onboarding for the newly released product feature, resulting in incorrect advice.",
        "Likely caused by a workflow gap where handoff from first-line to second-line support was not tracked in the CRM.",
        "Attributable to an SLA-timer configuration error that deprioritised an urgent ticket incorrectly.",
    ],

    # ── PRODUCT ───────────────────────────────────────────────────────────────
    ("product", "general"): [
        "Likely caused by a manufacturing quality-control gap that permitted a defective unit to pass final inspection.",
        "Root cause appears to be a firmware regression introduced in the v2.4 update that removed a previously working feature.",
        "Caused by a documented product specification that diverges from the actual shipped hardware configuration.",
        "Attributable to an edge-case interaction between two product features that was not covered in QA test scenarios.",
        "Likely triggered by user environmental conditions (voltage, humidity) outside the product's rated operating range.",
    ],

    # ── REFUND (top-level category) ───────────────────────────────────────────
    ("refund", "billing"): [
        "Likely caused by a refund request being opened after the merchant's 30-day policy window had elapsed.",
        "Root cause appears to be a payment-processor webhook failure that left the refund event unacknowledged.",
        "Attributable to a currency-mismatch in the refund record preventing the automated reconciliation job from completing.",
    ],
    ("refund", "general"): [
        "Likely caused by the refund being initiated on a closed or expired payment instrument.",
        "Root cause appears to be a manual override in the CRM that incorrectly marked the refund as 'paid' prematurely.",
        "Caused by an integration bug between the order-management system and the payment-gateway refund endpoint.",
    ],

    # ── OTHER / GENERIC ───────────────────────────────────────────────────────
    ("other", "general"): [
        "Likely caused by a cross-functional process gap that was not covered by existing operational runbooks.",
        "Root cause requires further triage — the complaint spans multiple service domains with unclear ownership.",
        "Attributable to an undocumented edge case in the service workflow that has not previously been encountered.",
    ],
}

# Catch-all when no specific (category, issue) key matches
DEFAULT_VARIANTS: List[str] = [
    "Likely caused by a service-reliability gap requiring cross-team operational investigation.",
    "Root cause appears to be an unhandled edge-case scenario in an integrated system component.",
    "Attributable to a workflow exception that fell outside the current support-team runbook coverage.",
    "Caused by a systemic process discrepancy between two interdependent service teams.",
    "Likely triggered by a configuration drift that was not detected during the last change-management review.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Template-based label selection
# ─────────────────────────────────────────────────────────────────────────────

def _variants_for(category: str, issue: str) -> List[str]:
    """Return the paraphrase pool for a (category, issue) pair, with fallbacks."""
    cat = category.lower().strip()
    iss = issue.lower().strip()

    # Exact match
    variants = TEMPLATE_VARIANTS.get((cat, iss))
    if variants:
        return variants

    # Category-only match (merge all issue-variants for this category)
    cat_variants = [
        v for (c, _i), vs in TEMPLATE_VARIANTS.items()
        if c == cat for v in vs
    ]
    if cat_variants:
        return cat_variants

    return DEFAULT_VARIANTS


def generate_template_label(category: str, issue: str) -> str:
    """Pick one paraphrase variant at random from the matching pool."""
    return random.choice(_variants_for(category, issue))


# ─────────────────────────────────────────────────────────────────────────────
# Optional: local Hugging Face instruct model
# ─────────────────────────────────────────────────────────────────────────────

def load_local_llm_pipeline(model_name: str):
    """Load one already-local Hugging Face pipeline without network access."""
    try:
        from transformers import (  # noqa: WPS433
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:
        raise RuntimeError("transformers is required for --use-local-llm") from exc

    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Local model '{model_name}' is unavailable. Download it separately or "
            "provide a local path; this command will not download models."
        ) from exc

    model_type = getattr(cfg, "model_type", "").lower()
    is_causal = any(
        kind in model_type
        for kind in ("qwen", "llama", "gpt", "bloom", "mistral", "falcon")
    )
    task = "text-generation" if is_causal else "text2text-generation"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_class = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM
        model = model_class.from_pretrained(model_name, local_files_only=True)
        return pipeline(task, model=model, tokenizer=tokenizer), is_causal
    except Exception as exc:
        raise RuntimeError(f"Unable to load local model '{model_name}': {exc}") from exc


def generate_llm_label(
    text: str,
    category: str,
    issue: str,
    pipeline_instance,
    is_causal: bool,
) -> str:
    """
    Generate one root-cause sentence from a preloaded local model.

    Callers decide whether errors should fail the run or explicitly fall back
    to templates.  This function must never hide an LLM failure as a template
    result because that would corrupt label provenance.
    """
    # Instruction prompt shared by both model families
    instruction = (
        f"You are a customer-support root-cause analyst. "
        f"Write one concise sentence explaining the most likely root cause of the "
        f"following {category} complaint (issue sub-type: {issue}).\n\n"
        f"Complaint: {text[:400]}\n\nRoot cause:"
    )

    try:
        if is_causal:
            raw = pipeline_instance(
                instruction, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=80,
            )[0]["generated_text"]
            result = raw.split("Root cause:")[-1].strip()
        else:
            result = pipeline_instance(
                instruction, do_sample=True, temperature=0.7, top_p=0.9,
                max_new_tokens=80,
            )[0]["generated_text"].strip()
    except Exception as exc:
        raise RuntimeError(f"Local LLM generation failed: {exc}") from exc

    result = result.split("\n")[0].strip()
    if result and result[-1] not in ".!?":
        result = result.rsplit(" ", 1)[0] + "."
    if not result:
        raise RuntimeError("Local LLM returned an empty root-cause label")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Unable to read cache %s; starting with an empty cache.", cache_file)
            return {}
    return {}


def save_cache(cache_file: Path, cache: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_datasets(data_dir: Path):
    """
    Load all available raw datasets from data/raw/.
    Returns a list of UnifiedSample objects.
    Gracefully skips any file that is absent or fails to parse.
    """
    try:
        from ml.dataset.loaders import load_all_datasets  # noqa: WPS433
        from ml.dataset.schema import UnifiedSample  # noqa: WPS433
    except ImportError as exc:
        logger.warning("ml.dataset not importable (%s). No local samples loaded.", exc)
        return [], None

    flags = {
        "use_local_complaint_dataset": True,
        "use_cfpb": True,
        "use_bitext": True,
        "use_banking77": True,
        "use_goemotions": True,
        "use_sms_jigsaw": True,
        "use_massive": True,
    }

    # Load from data/raw/ only. No dataset-service client is used.
    samples = load_all_datasets(data_dir, flags=flags)
    return samples, UnifiedSample


def _bootstrap_samples(UnifiedSample):
    """
    Internal seed set covering all (category, issue) pairs so the script
    always produces a meaningful CSV even with no external datasets.
    """
    seed_data = [
        # text, category, issue, sentiment
        ("I was double-charged on my monthly subscription invoice #8912.", "billing", "billing", "NEGATIVE"),
        ("Received a $499 charge I never authorised — card ending 4821.", "billing", "billing", "NEGATIVE"),
        ("My refund from 14 days ago still shows 'pending' in my account.", "billing", "refund", "NEGATIVE"),
        ("Refund was approved but money never arrived back on my card.", "billing", "refund", "NEGATIVE"),
        ("Invoice shows the wrong VAT rate for my country.", "billing", "general", "NEGATIVE"),
        ("Can't log in after password reset — the link gives a 404 error.", "technical", "auth", "NEGATIVE"),
        ("Two-factor authentication loop: entering the code just refreshes the page.", "technical", "auth", "NEGATIVE"),
        ("Authenticator app TOTP code is rejected immediately on every attempt.", "technical", "auth", "NEGATIVE"),
        ("App crashes every time I open the settings menu on iOS 17.", "technical", "technical", "NEGATIVE"),
        ("API returning 504 Gateway Timeout during nightly batch export.", "technical", "technical", "NEGATIVE"),
        ("Dashboard shows blank screen after the v2.4 update rolled out.", "technical", "technical", "NEGATIVE"),
        ("Webhook events stopped firing after we rotated the signing secret.", "technical", "general", "NEGATIVE"),
        ("Package shows 'delivered' but nothing arrived at the correct address.", "delivery", "delivery", "NEGATIVE"),
        ("Order dispatched 10 days ago, tracking stuck on 'in transit' since day 3.", "delivery", "delivery", "NEGATIVE"),
        ("Wrong item delivered — I ordered size L, received size S.", "delivery", "general", "NEGATIVE"),
        ("Agent hung up when I asked for a refund escalation — very unprofessional.", "customer_service", "support", "NEGATIVE"),
        ("Ticket open for 8 days with no agent response beyond the auto-reply.", "customer_service", "support", "NEGATIVE"),
        ("Support chat gave me incorrect instructions that made the issue worse.", "customer_service", "general", "NEGATIVE"),
        ("Screen on my device flickers whenever brightness goes above 60%.", "product", "general", "NEGATIVE"),
        ("Battery drains from 100% to 0% in under 90 minutes after firmware update.", "product", "general", "NEGATIVE"),
        ("Refund policy says 30 days but system won't accept my request on day 28.", "refund", "billing", "NEGATIVE"),
        ("Refund reference number you sent doesn't match anything on my statement.", "refund", "general", "NEGATIVE"),
        ("Not sure where to categorise this — the issue spans billing and delivery.", "other", "general", "NEUTRAL"),
        ("Generally satisfied but this one small thing bothered me.", "other", "general", "NEUTRAL"),
        # Slightly positive / neutral variants for diversity
        ("The issue resolved itself but I wanted to flag the experience.", "technical", "general", "NEUTRAL"),
        ("Billing looks correct now — just confirming the dispute is closed.", "billing", "general", "POSITIVE"),
        ("Package arrived today after the delay — thank you for following up.", "delivery", "general", "POSITIVE"),
        ("Agent eventually solved the problem but it took three calls.", "customer_service", "support", "NEUTRAL"),
        ("Happy with the product overall but the pairing process was confusing.", "product", "general", "NEUTRAL"),
        ("Refund received — I'm satisfied with the resolution.", "refund", "general", "POSITIVE"),
    ]

    samples = []
    for i, (txt, cat, iss, sent) in enumerate(seed_data):
        samples.append(UnifiedSample(
            id=f"bootstrap_{i:04d}",
            text=txt,
            language="en",
            message_type="complaint",
            category=cat,
            issue=iss,
            sentiment=sent,
            emotion="anger" if sent == "NEGATIVE" else "neutral",
        ))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(sample, generation_method: str, model_name: str, source: str) -> str:
    """Make a cache key that cannot mix sources, methods, models, or changed text."""
    payload = "\x1f".join(
        [generation_method, model_name, source, sample.id, sample.category,
         sample.issue, sample.text]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_entry(label: str, generation_method: str, model_name: str, source: str) -> dict:
    return {
        "label": label,
        "generation_method": generation_method,
        "model_name": model_name,
        "source": source,
    }


def run(
    use_local_llm: bool = False,
    local_model: str = "",
    review_samples: int = 200,
    output_csv: Path = DEFAULT_CSV,
    seed: int = 42,
    raw_data_dir: Path = RAW_DATA_DIR,
    template_cache_file: Path = TEMPLATE_CACHE_FILE,
    llm_cache_file: Path = LLM_CACHE_FILE,
    allow_bootstrap: bool = False,
    fallback_to_template: bool = False,
) -> Path:
    """Generate auditable labels and a review CSV from one explicit source.

    Bootstrap records are only allowed when explicitly requested. They are
    marked as synthetic and never silently mixed into real local-data reviews.
    """
    if review_samples <= 0:
        raise ValueError("review_samples must be greater than zero")
    if use_local_llm and not local_model:
        raise ValueError("--local-model is required with --use-local-llm")

    random.seed(seed)
    generation_method = "local_hf" if use_local_llm else "template"
    model_name = local_model if use_local_llm else "template-variants-v1"
    cache_file = llm_cache_file if use_local_llm else template_cache_file
    cache = load_cache(cache_file)
    logger.info("Loaded %d pre-cached silver labels.", len(cache))

    samples, UnifiedSample = _load_raw_datasets(Path(raw_data_dir))
    source = "local_dataset"

    if not samples:
        if not allow_bootstrap:
            raise RuntimeError(
                f"No local datasets found in {raw_data_dir}. Add manually downloaded "
                "data or pass --allow-bootstrap for explicitly synthetic demo data."
            )
        try:
            from ml.dataset.schema import UnifiedSample as US  # noqa: WPS433
        except ImportError:
            raise RuntimeError("Cannot import UnifiedSample for bootstrap data")
        samples = _bootstrap_samples(US)
        source = "synthetic_bootstrap"

    logger.info("Total samples to label: %d", len(samples))
    if len(samples) < review_samples:
        raise RuntimeError(
            f"Requested {review_samples} review samples but only {len(samples)} "
            f"valid {source} samples are available. No review CSV was written."
        )

    pipeline_instance = None
    is_causal = False
    if use_local_llm:
        pipeline_instance, is_causal = load_local_llm_pipeline(local_model)

    new_count = 0
    sample_generation_methods: dict[str, str] = {}
    for sample in samples:
        key = _cache_key(sample, generation_method, model_name, source)
        cached = cache.get(key)
        if isinstance(cached, dict) and cached.get("label"):
            sample.root_cause = cached["label"]
            sample_generation_methods[sample.id] = cached.get(
                "generation_method", generation_method
            )
            continue

        if use_local_llm:
            try:
                label = generate_llm_label(
                    text=sample.text, category=sample.category, issue=sample.issue,
                    pipeline_instance=pipeline_instance, is_causal=is_causal,
                )
            except RuntimeError:
                if not fallback_to_template:
                    raise
                logger.warning("LLM label failed for %s; explicit template fallback used.", sample.id)
                label = generate_template_label(sample.category, sample.issue)
                generation_for_entry = "template_fallback"
            else:
                generation_for_entry = "local_hf"
        else:
            label = generate_template_label(sample.category, sample.issue)
            generation_for_entry = "template"

        cache[key] = _cache_entry(label, generation_for_entry, model_name, source)
        sample.root_cause = label
        sample_generation_methods[sample.id] = generation_for_entry
        new_count += 1

    save_cache(cache_file, cache)
    logger.info(
        "Labels generated: %d new, %d from cache.  Total cached: %d.",
        new_count, len(samples) - new_count, len(cache),
    )

    review_pool = random.sample(samples, review_samples)

    rows = []
    for s in review_pool:
        rows.append({
            "sample_id":            s.id,
            "category":             s.category,
            "issue":                s.issue,
            "sentiment":            s.sentiment,
            "language":             getattr(s, "language", "en"),
            "text":                 textwrap.shorten(s.text, width=300, placeholder="…"),
            "silver_root_cause":    s.root_cause,
            "manual_approved":      "",
            "manual_correction":    "",
            "generation_method":    sample_generation_methods[s.id],
            "model_name":           model_name,
            "source":               source,
        })

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Exported %d samples for review → %s", len(df), output_csv)

    # ── 5. Print summary stats ───────────────────────────────────────────────
    cat_counts = df["category"].value_counts().to_dict()
    logger.info("Category distribution in exported CSV:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-20s %d", cat, cnt)
    return output_csv


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fully local silver root-cause label generator (no API keys).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Pure template mode (fastest, zero GPU required):
            python scripts/generate_silver_labels.py

            # Local model, with no downloads:
            python scripts/generate_silver_labels.py --use-local-llm \
                --local-model /path/to/local-model

            # Qwen2.5-1.5B-Instruct (GGUF / safetensors in a local folder):
            python scripts/generate_silver_labels.py --use-local-llm \\
                --local-model /path/to/Qwen2.5-1.5B-Instruct

            # Limit export to 50 rows:
            python scripts/generate_silver_labels.py --review-samples 50
        """),
    )
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        default=False,
        help="Use a local Hugging Face instruct model to generate additional label variants.",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="",
        help=(
            "Previously downloaded local Hugging Face model path or cache id. "
            "Required with --use-local-llm; downloads are disabled."
        ),
    )
    parser.add_argument(
        "--review-samples",
        type=int,
        default=200,
        help="Number of random samples to export to the review CSV (default: 200).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Destination path for the manual-review CSV file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Delete the selected method-specific silver-label cache before running.",
    )
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Use clearly labelled synthetic bootstrap data only when no local dataset exists.",
    )
    parser.add_argument(
        "--fallback-to-template",
        action="store_true",
        help="Allow explicit template fallback if local LLM generation fails.",
    )

    args = parser.parse_args()

    selected_cache = LLM_CACHE_FILE if args.use_local_llm else TEMPLATE_CACHE_FILE
    if args.clear_cache and selected_cache.exists():
        selected_cache.unlink()
        logger.info("Cache cleared: %s", selected_cache)

    run(
        use_local_llm=args.use_local_llm,
        local_model=args.local_model,
        review_samples=args.review_samples,
        output_csv=Path(args.output_csv),
        seed=args.seed,
        allow_bootstrap=args.allow_bootstrap,
        fallback_to_template=args.fallback_to_template,
    )
