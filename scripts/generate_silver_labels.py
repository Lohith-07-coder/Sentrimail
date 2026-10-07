"""
scripts/generate_silver_labels.py
----------------------------------
Fully local, offline silver root-cause label generator for SentriMail.
No external API keys or network calls required.

Features
--------
1. Configurable template map  (category, issue) → root-cause strings.
   Each bucket carries several paraphrase variants; one is chosen at
   random per sample to maximise training diversity.

2. --use-local-llm  flag:  loads a local Hugging Face text2text / causal
   instruct model (e.g. Qwen/Qwen2.5-1.5B-Instruct or google/flan-t5-large)
   fully offline and appends LLM-generated variants to the pool.
   Pass --local-model <name-or-path> to override the default.

3. Exports exactly 200 random samples (or all, if fewer exist) to
   data/silver_labels_sample_review.csv for manual spot-check.

4. Results are cached to data/silver_labels_cache.json so re-runs are
   incremental and idempotent.

Usage
-----
  # Template-only (zero dependencies beyond pandas):
  python scripts/generate_silver_labels.py

  # With local LLM (flan-t5-large, downloads once to HF cache):
  python scripts/generate_silver_labels.py --use-local-llm

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
RAW_DATA_DIR   = PROJECT_ROOT / "data" / "raw"
CACHE_FILE     = PROJECT_ROOT / "data" / "silver_labels_cache.json"
DEFAULT_CSV    = PROJECT_ROOT / "data" / "silver_labels_sample_review.csv"

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

def generate_llm_label(
    text: str,
    category: str,
    issue: str,
    model_name: str = "google/flan-t5-large",
) -> str:
    """
    Generate a root-cause sentence using a local HF model (no network call
    after first download).  Falls back to the template pool on any error.

    For text2text models (T5, BART, etc.) the pipeline type is
    'text2text-generation'.  For causal / instruct models (Qwen, Llama, etc.)
    use 'text-generation' and format an instruction prompt.
    """
    try:
        from transformers import pipeline, AutoConfig  # noqa: WPS433
    except ImportError:
        logger.warning("transformers not installed — falling back to templates.")
        return generate_template_label(category, issue)

    try:
        cfg = AutoConfig.from_pretrained(model_name)
        model_type = getattr(cfg, "model_type", "").lower()
    except Exception:
        model_type = ""

    # Instruction prompt shared by both model families
    instruction = (
        f"You are a customer-support root-cause analyst. "
        f"Write one concise sentence explaining the most likely root cause of the "
        f"following {category} complaint (issue sub-type: {issue}).\n\n"
        f"Complaint: {text[:400]}\n\nRoot cause:"
    )

    try:
        logger.info("Loading local HF model '%s' …", model_name)

        is_causal = any(k in model_type for k in ("qwen", "llama", "gpt", "bloom", "mistral", "falcon"))

        if is_causal:
            pipe = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                max_new_tokens=80,
            )
            raw = pipe(instruction, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
            # Strip the echoed prompt
            result = raw.split("Root cause:")[-1].strip()
        else:
            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                device_map="auto",
                max_new_tokens=80,
            )
            result = pipe(instruction, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"].strip()

        # Sanitise: keep only the first sentence and strip incomplete tails
        result = result.split("\n")[0].strip()
        if result and result[-1] not in ".!?":
            result = result.rsplit(" ", 1)[0] + "."
        return result if result else generate_template_label(category, issue)

    except Exception as exc:
        logger.warning("Local LLM inference failed (%s). Using template fallback.", exc)
        return generate_template_label(category, issue)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_datasets():
    """
    Load all available raw datasets from data/raw/.
    Returns a list of UnifiedSample objects.
    Gracefully skips any file that is absent or fails to parse.
    """
    try:
        from ml.dataset.loaders import load_all_datasets  # noqa: WPS433
        from ml.dataset.schema import UnifiedSample  # noqa: WPS433
    except ImportError as exc:
        logger.warning("ml.dataset not importable (%s). Using bootstrap samples only.", exc)
        return [], None

    flags = {
        "use_kaggle": True,
        "use_cfpb": True,
        "use_bitext": True,
        "use_banking77": True,
        "use_goemotions": True,
        "use_sms_jigsaw": True,
        "use_massive": True,
    }

    # Load from data/raw/ (no Kaggle API needed — manual downloads only)
    samples = load_all_datasets(RAW_DATA_DIR, flags=flags)
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

def run(
    use_local_llm: bool = False,
    local_model: str = "google/flan-t5-large",
    review_samples: int = 200,
    output_csv: Path = DEFAULT_CSV,
    seed: int = 42,
) -> None:
    random.seed(seed)

    # ── 1. Load cache ────────────────────────────────────────────────────────
    cache = load_cache()
    logger.info("Loaded %d pre-cached silver labels.", len(cache))

    # ── 2. Load datasets ─────────────────────────────────────────────────────
    samples, UnifiedSample = _load_raw_datasets()

    if not samples:
        logger.info(
            "No datasets found in %s — using internal bootstrap set "
            "(%d samples covering all (category, issue) buckets).",
            RAW_DATA_DIR, 30,
        )
        # Re-import UnifiedSample for bootstrap
        try:
            from ml.dataset.schema import UnifiedSample as US  # noqa: WPS433
        except ImportError:
            logger.error("Cannot import UnifiedSample. Aborting.")
            return
        samples = _bootstrap_samples(US)
    else:
        # Always add bootstrap samples to guarantee full bucket coverage
        try:
            from ml.dataset.schema import UnifiedSample as US  # noqa: WPS433
            samples = samples + _bootstrap_samples(US)
        except ImportError:
            pass

    logger.info("Total samples to label: %d", len(samples))

    # ── 3. Generate labels ───────────────────────────────────────────────────
    new_count = 0
    for sample in samples:
        sid = sample.id

        if sid in cache:
            sample.root_cause = cache[sid]
            continue

        if use_local_llm:
            label = generate_llm_label(
                text=sample.text,
                category=sample.category,
                issue=sample.issue,
                model_name=local_model,
            )
        else:
            label = generate_template_label(sample.category, sample.issue)

        cache[sid] = label
        sample.root_cause = label
        new_count += 1

    save_cache(cache)
    logger.info(
        "Labels generated: %d new, %d from cache.  Total cached: %d.",
        new_count, len(samples) - new_count, len(cache),
    )

    # ── 4. Export 200 random samples to CSV ─────────────────────────────────
    n_export = min(review_samples, len(samples))
    review_pool = random.sample(samples, n_export)

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
            # Reviewer fills in these two columns during manual spot-check
            "manual_approved":      "",   # Y / N / EDIT
            "manual_correction":    "",
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

            # Local flan-t5-large  (downloads ~3 GB once, then offline):
            python scripts/generate_silver_labels.py --use-local-llm

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
        default="google/flan-t5-large",
        help=(
            "HF model name or local path for instruct inference. "
            "Default: google/flan-t5-large"
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
        help="Delete the existing silver-label cache before running.",
    )

    args = parser.parse_args()

    if args.clear_cache and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info("Cache cleared: %s", CACHE_FILE)

    run(
        use_local_llm=args.use_local_llm,
        local_model=args.local_model,
        review_samples=args.review_samples,
        output_csv=Path(args.output_csv),
        seed=args.seed,
    )
