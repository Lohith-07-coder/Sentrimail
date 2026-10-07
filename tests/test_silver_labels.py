"""Phase 1 tests for the strictly local silver-label workflow."""

import csv
import random
from pathlib import Path

import pytest

from scripts import generate_silver_labels as silver


def _write_local_dataset(path: Path, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "category", "response"])
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "text": f"My order has a billing issue number {index}.",
                    "category": "billing",
                    "response": "We are investigating the billing issue.",
                }
            )


def _run_template_generation(tmp_path: Path, count: int = 200) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    _write_local_dataset(raw_dir / "complaints.csv", count)
    output = tmp_path / "review.csv"
    return silver.run(
        review_samples=count,
        output_csv=output,
        seed=42,
        raw_data_dir=raw_dir,
        template_cache_file=tmp_path / "template-cache.json",
        llm_cache_file=tmp_path / "llm-cache.json",
    )


def test_template_selection_is_reproducible_with_seed():
    random.seed(42)
    first = [silver.generate_template_label("billing", "billing") for _ in range(10)]
    random.seed(42)
    second = [silver.generate_template_label("billing", "billing") for _ in range(10)]
    assert first == second


def test_review_export_requires_exact_requested_sample_count(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Requested 200 review samples"):
        silver.run(
            review_samples=200,
            output_csv=tmp_path / "review.csv",
            raw_data_dir=tmp_path / "missing-raw",
            template_cache_file=tmp_path / "template-cache.json",
            llm_cache_file=tmp_path / "llm-cache.json",
            allow_bootstrap=True,
        )


def test_template_generation_exports_auditable_200_row_review_set(tmp_path: Path):
    output = _run_template_generation(tmp_path)
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 200
    assert set(rows[0]) == {
        "sample_id", "category", "issue", "sentiment", "language", "text",
        "silver_root_cause", "manual_approved", "manual_correction",
        "generation_method", "model_name", "source",
    }
    assert {row["manual_approved"] for row in rows} == {""}
    assert {row["manual_correction"] for row in rows} == {""}
    assert {row["generation_method"] for row in rows} == {"template"}
    assert {row["source"] for row in rows} == {"local_dataset"}


def test_template_cache_is_method_aware_and_reusable(tmp_path: Path):
    _run_template_generation(tmp_path, count=5)
    cache_path = tmp_path / "template-cache.json"
    cached_before = cache_path.read_text(encoding="utf-8")
    _run_template_generation(tmp_path, count=5)

    assert cache_path.read_text(encoding="utf-8") == cached_before
    assert '"generation_method": "template"' in cached_before


def test_local_hf_loader_is_called_once_and_never_downloads(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_local_dataset(raw_dir / "complaints.csv", 3)
    loads = []

    def fake_loader(model_name):
        loads.append(model_name)

        def fake_pipeline(*_args, **_kwargs):
            return [{"generated_text": "A local model root cause."}]

        return fake_pipeline, False

    loader_source = Path(silver.__file__).read_text(encoding="utf-8")
    monkeypatch.setattr(silver, "load_local_llm_pipeline", fake_loader)
    silver.run(
        use_local_llm=True,
        local_model="already-local-model",
        review_samples=3,
        output_csv=tmp_path / "review.csv",
        raw_data_dir=raw_dir,
        template_cache_file=tmp_path / "template-cache.json",
        llm_cache_file=tmp_path / "llm-cache.json",
    )

    assert loads == ["already-local-model"]
    assert "local_files_only=True" in loader_source


def test_active_python_code_has_no_gemini_or_kaggle_api_dependency():
    root = Path(__file__).resolve().parents[1]
    active_paths = [
        *root.joinpath("app").rglob("*.py"),
        *root.joinpath("scripts").rglob("*.py"),
        *root.joinpath("ml").rglob("*.py"),
    ]
    active_python = "\n".join(
        path.read_text(encoding="utf-8") for path in active_paths
    ).lower()
    assert "google.generativeai" not in active_python
    assert "gemini_api_key" not in active_python
    assert "import kaggle" not in active_python
    assert "kaggle.api" not in active_python
