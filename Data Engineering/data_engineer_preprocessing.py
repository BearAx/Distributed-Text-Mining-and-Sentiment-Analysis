#!/usr/bin/env python3
"""
Responsibilities:
- Load dataset from TXT / CSV / JSON / JSONL
- Clean text: lowercase + remove punctuation + normalize spaces
- Tokenize text into a tokens list
- Output per-document tokens for the next pipeline stage

Example:
    python data_engineer_preprocessing.py reviews.csv --text-column review --output tokens.json
    python data_engineer_preprocessing.py tweets.jsonl --output tokens.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional


DEFAULT_TEXT_COLUMNS = [
    "text",
    "review",
    "tweet",
    "headline",
    "content",
    "body",
    "message",
    "sentence",
    "document",
]

PUNCT_TRANSLATION_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""


class TextColumnError(Exception):
    """Raised when no text column can be found."""



def detect_text_column(columns: Iterable[str], user_choice: Optional[str] = None) -> str:
    columns = list(columns)
    if not columns:
        raise TextColumnError("The dataset has no columns.")

    if user_choice:
        if user_choice in columns:
            return user_choice
        raise TextColumnError(
            f"Specified text column '{user_choice}' was not found. Available columns: {columns}"
        )

    lowered = {col.lower(): col for col in columns}
    for candidate in DEFAULT_TEXT_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]

    # fallback: first string-like looking column name if typical names are absent
    for col in columns:
        if "text" in col.lower() or "review" in col.lower() or "tweet" in col.lower():
            return col

    raise TextColumnError(
        "Could not automatically detect the text column. "
        f"Please pass --text-column. Available columns: {columns}"
    )



def clean_text(text: Any) -> str:
    """Lowercase text, remove punctuation, normalize whitespace."""
    if text is None:
        return ""

    text = str(text).lower()

    # replace punctuation with spaces so words do not merge together
    text = text.translate(PUNCT_TRANSLATION_TABLE)

    # remove common unicode punctuation that is not covered by string.punctuation
    text = re.sub(r"[“”‘’—–…•]", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text



def tokenize(text: str) -> List[str]:
    """Whitespace tokenization after cleaning."""
    if not text:
        return []
    return text.split()



def load_txt(path: Path) -> List[Dict[str, Any]]:
    """Assume one document per non-empty line."""
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                docs.append({"doc_id": idx, "text": line})
    return docs



def load_csv(path: Path, text_column: Optional[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise DatasetLoadError("CSV file has no header row.")

        actual_text_column = detect_text_column(reader.fieldnames, text_column)
        docs: List[Dict[str, Any]] = []

        for idx, row in enumerate(reader, start=1):
            docs.append(
                {
                    "doc_id": row.get("doc_id", idx),
                    "text": row.get(actual_text_column, ""),
                }
            )
        return docs



def load_json(path: Path, text_column: Optional[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Allow dict with documents list inside common keys
        for key in ("documents", "data", "items", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # single JSON document object
            data = [data]

    if not isinstance(data, list):
        raise DatasetLoadError("JSON file must contain a list of documents or an object with a documents list.")

    if not data:
        return []

    if not isinstance(data[0], dict):
        # list of strings
        return [{"doc_id": idx, "text": item} for idx, item in enumerate(data, start=1)]

    actual_text_column = detect_text_column(data[0].keys(), text_column)

    docs: List[Dict[str, Any]] = []
    for idx, row in enumerate(data, start=1):
        docs.append(
            {
                "doc_id": row.get("doc_id", idx),
                "text": row.get(actual_text_column, ""),
            }
        )
    return docs



def load_jsonl(path: Path, text_column: Optional[str]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    actual_text_column: Optional[str] = text_column

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if isinstance(row, dict):
                if actual_text_column is None:
                    actual_text_column = detect_text_column(row.keys(), text_column)
                docs.append(
                    {
                        "doc_id": row.get("doc_id", idx),
                        "text": row.get(actual_text_column, ""),
                    }
                )
            else:
                docs.append({"doc_id": idx, "text": row})

    return docs



def load_dataset(path: str, text_column: Optional[str] = None) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return load_txt(file_path)
    if suffix == ".csv":
        return load_csv(file_path, text_column)
    if suffix == ".json":
        return load_json(file_path, text_column)
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(file_path, text_column)

    raise DatasetLoadError(
        f"Unsupported file type: {suffix}. Supported types: .txt, .csv, .json, .jsonl, .ndjson"
    )



def preprocess_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []

    for doc in documents:
        original_text = doc.get("text", "")
        cleaned = clean_text(original_text)
        tokens = tokenize(cleaned)
        processed.append(
            {
                "doc_id": doc.get("doc_id"),
                "original_text": original_text,
                "cleaned_text": cleaned,
                "tokens": tokens,
            }
        )

    return processed



def save_output(processed_docs: List[Dict[str, Any]], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)



def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a text dataset, clean it, tokenize it, and export tokens for the sentiment pipeline."
    )
    parser.add_argument("input", help="Path to input dataset (.txt, .csv, .json, .jsonl)")
    parser.add_argument(
        "--text-column",
        default=None,
        help="Name of the column/key containing text. If omitted, the script tries to detect it automatically.",
    )
    parser.add_argument(
        "--output",
        default="tokens_output.json",
        help="Path to output JSON file. Default: tokens_output.json",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print first 3 processed documents to the console.",
    )
    return parser



def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    documents = load_dataset(args.input, args.text_column)
    processed = preprocess_documents(documents)
    save_output(processed, args.output)

    print(f"Loaded documents: {len(documents)}")
    print(f"Saved output to: {args.output}")

    if args.preview:
        preview_count = min(3, len(processed))
        print(f"\nPreviewing first {preview_count} processed documents:")
        for item in processed[:preview_count]:
            print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
