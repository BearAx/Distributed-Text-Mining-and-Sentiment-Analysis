from pathlib import Path
import json
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths for lexicon-scoring stage inputs and outputs.
TOKENS_FILE = BASE_DIR / "output_data" / "tokens.json"
LEXICON_FILE = BASE_DIR / "data" / "sentiment_lexicon.json"
OUTPUT_DIR = BASE_DIR / "output_lexicon"

SCORED_OUTPUT_FILE = OUTPUT_DIR / "scored_documents.json"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "sentiment_summary.json"


def load_tokens(tokens_path: Path) -> List[Dict]:
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Tokens file not found: {tokens_path}. Run data_preprocessing.py first."
        )
    with open(tokens_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tokens.json must contain a list of records.")
    return data


def load_lexicon(lexicon_path: Path) -> Dict[str, int]:
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    with open(lexicon_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Lexicon file must contain a JSON object.")

    flattened: Dict[str, int] = {}

    # Accept either a flat dictionary or grouped positive/negative blocks.
    has_nested_blocks = any(isinstance(v, dict) for v in raw.values())
    if has_nested_blocks:
        for block_name, block in raw.items():
            if not isinstance(block, dict):
                raise ValueError(
                    f"Block '{block_name}' must be an object of word->score pairs."
                )
            for word, score in block.items():
                if not isinstance(score, int):
                    raise ValueError(
                        f"Lexicon score for '{word}' must be integer, got {type(score)}."
                    )
                flattened[word.lower()] = score
    else:
        for word, score in raw.items():
            if not isinstance(score, int):
                raise ValueError(
                    f"Lexicon score for '{word}' must be integer, got {type(score)}."
                )
            flattened[word.lower()] = score

    return flattened


def score_tokens(tokens: List[str], lexicon: Dict[str, int]) -> Tuple[int, int, int]:
    # Aggregate document score and track contributing positive/negative token hits.
    score = 0
    positive_hits = 0
    negative_hits = 0

    for token in tokens:
        token_score = lexicon.get(str(token).lower(), 0)
        score += token_score
        if token_score > 0:
            positive_hits += 1
        elif token_score < 0:
            negative_hits += 1

    return score, positive_hits, negative_hits


def score_to_label(score: int) -> str:
    # Convert scalar score into required sentiment class.
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenized documents from {TOKENS_FILE}")
    token_records = load_tokens(TOKENS_FILE)
    print(f"Loaded {len(token_records)} documents")

    print(f"Loading sentiment lexicon from {LEXICON_FILE}")
    lexicon = load_lexicon(LEXICON_FILE)
    print(f"Loaded {len(lexicon)} lexicon terms")

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    evaluated = 0
    correct = 0
    scored_records = []

    print("Scoring documents")
    for record in token_records:
        doc_id = int(record.get("doc_id", -1))
        tokens = record.get("tokens", [])

        if not isinstance(tokens, list):
            tokens = []

        score, pos_hits, neg_hits = score_tokens(tokens, lexicon)
        predicted = score_to_label(score)
        counts[predicted] += 1

        out = {
            "doc_id": doc_id,
            "tokens": tokens,
            "score": score,
            "positive_hits": pos_hits,
            "negative_hits": neg_hits,
            "predicted_sentiment": predicted
        }

        true_label = record.get("label")
        if isinstance(true_label, str):
            normalized_label = true_label.strip().lower()
            out["true_label"] = normalized_label
            if normalized_label in counts:
                evaluated += 1
                if normalized_label == predicted:
                    correct += 1

        scored_records.append(out)

    accuracy = round(correct / evaluated, 4) if evaluated > 0 else None
    summary = {
        "documents_count": len(scored_records),
        "lexicon_terms": len(lexicon),
        "positive_documents": counts["positive"],
        "negative_documents": counts["negative"],
        "neutral_documents": counts["neutral"],
        "evaluated_documents": evaluated,
        "accuracy": accuracy
    }

    with open(SCORED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scored_records, f, ensure_ascii=False, indent=2)
    print(f"Saved document-level predictions to {SCORED_OUTPUT_FILE}")

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregate summary to {SUMMARY_OUTPUT_FILE}")

    print("\nSummary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nLexicon stage completed.")


if __name__ == "__main__":
    main()
