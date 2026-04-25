import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent


def choose_default_predictions_file(base_dir: Path) -> Path:
    candidates = [
        base_dir / "output_parallel" / "parallel_scored_documents.json",
        base_dir / "output_lexicon" / "scored_documents.json",
        base_dir / "output_mapreduce" / "mapper_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Predictions file must contain a list of document records.")
    return data


def normalize_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if label in {"positive", "negative", "neutral"}:
        return label
    return None


def extract_predicted_label(record: Dict[str, Any]) -> Optional[str]:
    for key in ("predicted_sentiment", "predicted"):
        label = normalize_label(record.get(key))
        if label:
            return label
    return None


def extract_true_label(record: Dict[str, Any]) -> Optional[str]:
    for key in ("true_label", "label"):
        label = normalize_label(record.get(key))
        if label:
            return label
    return None


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentiment_order = ["positive", "negative", "neutral"]

    counts = Counter()
    evaluated = 0
    correct = 0

    confusion = {
        true_label: {pred_label: 0 for pred_label in sentiment_order}
        for true_label in sentiment_order
    }

    validated_records: List[Dict[str, Any]] = []
    token_lengths: List[int] = []
    score_values: List[int] = []

    for record in records:
        predicted = extract_predicted_label(record)
        true_label = extract_true_label(record)

        if predicted:
            counts[predicted] += 1

        tokens = record.get("tokens", [])
        if isinstance(tokens, list):
            token_lengths.append(len(tokens))

        score = record.get("score")
        if isinstance(score, int):
            score_values.append(score)

        out_record: Dict[str, Any] = {
            "doc_id": record.get("doc_id"),
            "predicted_sentiment": predicted,
        }

        if "score" in record:
            out_record["score"] = record.get("score")
        if "positive_hits" in record:
            out_record["positive_hits"] = record.get("positive_hits")
        if "negative_hits" in record:
            out_record["negative_hits"] = record.get("negative_hits")
        if isinstance(tokens, list):
            out_record["token_count"] = len(tokens)

        if true_label:
            evaluated += 1
            is_correct = predicted == true_label
            if is_correct:
                correct += 1
            if predicted in confusion[true_label]:
                confusion[true_label][predicted] += 1

            out_record["true_label"] = true_label
            out_record["is_correct"] = is_correct

        validated_records.append(out_record)

    accuracy = round(correct / evaluated, 4) if evaluated > 0 else None
    total_documents = len(records)
    avg_tokens = round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0
    avg_score = round(sum(score_values) / len(score_values), 4) if score_values else None

    class_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for label in sentiment_order:
        tp = confusion[label][label]
        predicted_total = sum(confusion[true][label] for true in sentiment_order)
        actual_total = sum(confusion[label][pred] for pred in sentiment_order)

        precision = round(tp / predicted_total, 4) if predicted_total > 0 else None
        recall = round(tp / actual_total, 4) if actual_total > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 4)
        else:
            f1 = None

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
        }

    summary = {
        "documents_count": total_documents,
        "positive_documents": counts["positive"],
        "negative_documents": counts["negative"],
        "neutral_documents": counts["neutral"],
        "evaluated_documents": evaluated,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "avg_tokens_per_document": avg_tokens,
        "avg_score": avg_score,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion,
    }

    return {
        "validated_records": validated_records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Validation & Metrics for sentiment analysis pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=choose_default_predictions_file(BASE_DIR),
        help=(
            "Predictions file to validate. Defaults to the first existing file among "
            "output_parallel/parallel_scored_documents.json, "
            "output_lexicon/scored_documents.json, "
            "output_mapreduce/mapper_results.json."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE_DIR / "output_validation",
        help="Output directory (default: output_validation)",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.input}")
    records = load_json_list(args.input)
    print(f"Loaded {len(records)} records")

    result = compute_metrics(records)
    validated_records = result["validated_records"]
    summary = result["summary"]

    validated_path = args.outdir / "validated_predictions.json"
    summary_path = args.outdir / "metrics_summary.json"

    with open(validated_path, "w", encoding="utf-8") as f:
        json.dump(validated_records, f, ensure_ascii=False, indent=2)
    print(f"Saved validated document results to {validated_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics summary to {summary_path}")

    print("\nValidation summary")
    for key, value in summary.items():
        if key not in {"class_metrics", "confusion_matrix"}:
            print(f"{key}: {value}")

    if summary["accuracy"] is None:
        print("\nNo labels were found, so accuracy was not computed.")
    else:
        print("\nClass metrics:")
        for label, metrics in summary["class_metrics"].items():
            print(f"{label}: {metrics}")

    print("\nValidation & Metrics stage completed.")


if __name__ == "__main__":
    main()
