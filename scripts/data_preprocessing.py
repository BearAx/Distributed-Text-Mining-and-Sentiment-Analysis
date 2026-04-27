import pandas as pd
import re
import string
import json
from collections import Counter
from pathlib import Path
from typing import Union


# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "all-data.csv"
OUTPUT_DIR = BASE_DIR / "output_data"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Possible names of text columns
POSSIBLE_TEXT_COLUMNS = [
    "text", "review", "sentence", "headline", "tweet",
    "content", "body", "message", "document", "news"
]


# LOAD DATA
def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            encoding="ISO-8859-1",
            header=None,
            names=["label", "text"]
        )

    elif suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)

    elif suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": lines})

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# DETECT TEXT COLUMN
def detect_text_column(df: pd.DataFrame) -> str:
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in POSSIBLE_TEXT_COLUMNS:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # If there is no explicit name, let's try to find the first object/string column.
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    raise ValueError("The text column could not be detected automatically.")


# TEXT CLEANING
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]",
                  " ", text)  # delete punctuation
    # normalize the spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    if not text:
        return []
    return text.split()


# MAIN PIPELINE
def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1] Uploading a dataset: {INPUT_FILE}")
    df = load_dataset(INPUT_FILE)
    print(f"    Lines uploaded: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Defining a text column
    text_col = TEXT_COLUMN if TEXT_COLUMN is not None else detect_text_column(
        df)
    print(f"[2] Use a text column: {text_col}")

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset.")

    # Create doc_id
    df = df.copy()
    df["doc_id"] = range(1, len(df) + 1)

    # Deleting entries without text
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""].copy()

    # Removing duplicates in the text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col]).copy()
    after_dedup = len(df)

    print(f"[3] After deleting the empty lines: {len(df)}")
    print(f"[4] Duplicates removed: {before_dedup - after_dedup}")

    # Purification and tokenization
    print("[5] I clean up texts and tokenize them")
    df["cleaned_text"] = df[text_col].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)

    # Deleting entries without tokens
    before_empty_tokens = len(df)
    df = df[df["token_count"] > 0].copy()
    removed_empty_tokens = before_empty_tokens - len(df)

    print(f"[6] Deleted entries without tokens: {removed_empty_tokens}")
    print(f"    The final number of documents: {len(df)}")

    # Building a dictionary of frequencies
    print("[7] I'm creating a corpus dictionary")
    vocab_counter = Counter()
    for tokens in df["tokens"]:
        vocab_counter.update(tokens)

    unique_tokens = len(vocab_counter)
    total_tokens = sum(vocab_counter.values())

    print(f"    Total tokens: {total_tokens}")
    print(f"    Unique tokens: {unique_tokens}")

    # Preparing token records
    token_records = []
    for _, row in df.iterrows():
        record = {
            "doc_id": int(row["doc_id"]),
            "original_text": row[text_col],
            "cleaned_text": row["cleaned_text"],
            "tokens": row["tokens"]
        }
        if LABEL_COLUMN and LABEL_COLUMN in df.columns:
            record["label"] = row[LABEL_COLUMN]
        token_records.append(record)

    # Save cleaned dataset
    cleaned_columns = ["doc_id", text_col,
                       "cleaned_text", "tokens", "token_count"]
    if LABEL_COLUMN and LABEL_COLUMN in df.columns:
        cleaned_columns.append(LABEL_COLUMN)

    cleaned_df = df[cleaned_columns].copy()
    cleaned_df.to_csv(output_path / "cleaned_dataset.csv",
                      index=False, encoding="utf-8")
    print(f"[8] Saved cleaned_dataset.csv")

    # Save tokens.json
    with open(output_path / "tokens.json", "w", encoding="utf-8") as f:
        json.dump(token_records, f, ensure_ascii=False, indent=2)
    print(f"[9] Saved tokens.json")

    # Save vocabulary.json
    with open(output_path / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(dict(vocab_counter.most_common()),
                  f, ensure_ascii=False, indent=2)
    print(f"[10] Saved vocabulary.json")

    # Save top_words.csv
    vocab_df = pd.DataFrame(vocab_counter.most_common(),
                            columns=["token", "frequency"])
    vocab_df.to_csv(output_path / "top_words.csv",
                    index=False, encoding="utf-8")
    print(f"[11] Saved top_words.csv")

    # Save summary.json
    summary = {
        "input_file": str(INPUT_FILE),
        "text_column": text_col,
        "documents_count": int(len(df)),
        "total_tokens": int(total_tokens),
        "unique_tokens": int(unique_tokens),
        "avg_tokens_per_document": round(float(df["token_count"].mean()), 2) if len(df) > 0 else 0,
        "top_20_words": vocab_counter.most_common(20)
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[12] Saved summary.json")

    # Showing the preview
    print("\n===== PREVIEW =====")
    print(df[[text_col, "cleaned_text", "tokens"]].head(
        5).to_string(index=False))

    print("\n===== TOP 20 WORDS =====")
    for word, freq in vocab_counter.most_common(20):
        print(f"{word}: {freq}")

    print(f"\nDone. The results are in the folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
