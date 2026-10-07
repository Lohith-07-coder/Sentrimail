import csv
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.response_intelligence import _retrieve_candidates

def main():
    csv_path = PROJECT_ROOT / "data" / "silver_labels_sample_review.csv"
    if not csv_path.exists():
        print(f"Data not found at {csv_path}")
        return

    total = 0
    tfidf_wins = 0
    semantic_wins = 0

    print("--- Evaluating Retrieval Approaches ---")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            category = row.get("category", "")
            
            candidates = _retrieve_candidates(text, category, "LOW")
            if candidates:
                best = candidates[0]
                total += 1
                source = best.get("source")
                if source == "tfidf":
                    tfidf_wins += 1
                else:
                    semantic_wins += 1
                
                print(f"Text: {text[:50]}...")
                print(f"Top Source: {source}, Score: {best['score']:.2f}")
                print("-" * 20)

    print("\n--- Summary ---")
    print(f"Total samples evaluated: {total}")
    print(f"TF-IDF top hits: {tfidf_wins}")
    print(f"Semantic top hits: {semantic_wins}")

if __name__ == "__main__":
    main()
