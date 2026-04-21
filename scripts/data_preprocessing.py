import pandas as pd
import re
import string
import json
from collections import Counter
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_FILE = "all-data.csv"      # замени на свой файл
# например "text" / "Sentence" / "review"; если None, скрипт попробует найти сам
TEXT_COLUMN = "text"
# например "label" / "sentiment"; можно оставить None
LABEL_COLUMN = "label"
OUTPUT_DIR = "output_data"

# Возможные названия текстовых колонок
POSSIBLE_TEXT_COLUMNS = [
    "text", "review", "sentence", "headline", "tweet",
    "content", "body", "message", "document", "news"
]

# =========================
# LOAD DATA
# =========================


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

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
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

# =========================
# DETECT TEXT COLUMN
# =========================


def detect_text_column(df: pd.DataFrame) -> str:
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in POSSIBLE_TEXT_COLUMNS:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # Если явного имени нет, попробуем найти первую object/string колонку
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    raise ValueError("Не удалось определить текстовую колонку автоматически.")

# =========================
# TEXT CLEANING
# =========================


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]",
                  " ", text)  # убираем punctuation
    # нормализуем пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    if not text:
        return []
    return text.split()

# =========================
# MAIN PIPELINE
# =========================


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1] Загружаю датасет: {INPUT_FILE}")
    df = load_dataset(INPUT_FILE)
    print(f"    Загружено строк: {len(df)}")
    print(f"    Колонки: {list(df.columns)}")

    # Определяем текстовую колонку
    text_col = TEXT_COLUMN if TEXT_COLUMN is not None else detect_text_column(
        df)
    print(f"[2] Использую текстовую колонку: {text_col}")

    if text_col not in df.columns:
        raise ValueError(f"Колонка '{text_col}' не найдена в датасете.")

    # Создаем doc_id
    df = df.copy()
    df["doc_id"] = range(1, len(df) + 1)

    # Удаляем записи без текста
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""].copy()

    # Удаляем дубликаты по тексту
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col]).copy()
    after_dedup = len(df)

    print(f"[3] После удаления пустых строк: {len(df)}")
    print(f"[4] Удалено дубликатов: {before_dedup - after_dedup}")

    # Очистка и токенизация
    print("[5] Очищаю тексты и токенизирую")
    df["cleaned_text"] = df[text_col].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)

    # Удаляем записи без токенов
    before_empty_tokens = len(df)
    df = df[df["token_count"] > 0].copy()
    removed_empty_tokens = before_empty_tokens - len(df)

    print(f"[6] Удалено записей без токенов: {removed_empty_tokens}")
    print(f"    Финальное число документов: {len(df)}")

    # Строим словарь частот
    print("[7] Формирую словарь корпуса")
    vocab_counter = Counter()
    for tokens in df["tokens"]:
        vocab_counter.update(tokens)

    unique_tokens = len(vocab_counter)
    total_tokens = sum(vocab_counter.values())

    print(f"    Всего токенов: {total_tokens}")
    print(f"    Уникальных токенов: {unique_tokens}")

    # Подготовка token records
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

    # Сохраняем cleaned dataset
    cleaned_columns = ["doc_id", text_col,
                       "cleaned_text", "tokens", "token_count"]
    if LABEL_COLUMN and LABEL_COLUMN in df.columns:
        cleaned_columns.append(LABEL_COLUMN)

    cleaned_df = df[cleaned_columns].copy()
    cleaned_df.to_csv(output_path / "cleaned_dataset.csv",
                      index=False, encoding="utf-8")
    print(f"[8] Сохранен cleaned_dataset.csv")

    # Сохраняем tokens.json
    with open(output_path / "tokens.json", "w", encoding="utf-8") as f:
        json.dump(token_records, f, ensure_ascii=False, indent=2)
    print(f"[9] Сохранен tokens.json")

    # Сохраняем vocabulary.json
    with open(output_path / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(dict(vocab_counter.most_common()),
                  f, ensure_ascii=False, indent=2)
    print(f"[10] Сохранен vocabulary.json")

    # Сохраняем top_words.csv
    vocab_df = pd.DataFrame(vocab_counter.most_common(),
                            columns=["token", "frequency"])
    vocab_df.to_csv(output_path / "top_words.csv",
                    index=False, encoding="utf-8")
    print(f"[11] Сохранен top_words.csv")

    # Сохраняем summary.json
    summary = {
        "input_file": INPUT_FILE,
        "text_column": text_col,
        "documents_count": int(len(df)),
        "total_tokens": int(total_tokens),
        "unique_tokens": int(unique_tokens),
        "avg_tokens_per_document": round(float(df["token_count"].mean()), 2) if len(df) > 0 else 0,
        "top_20_words": vocab_counter.most_common(20)
    }

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[12] Сохранен summary.json")

    # Показываем превью
    print("\n===== PREVIEW =====")
    print(df[[text_col, "cleaned_text", "tokens"]].head(
        5).to_string(index=False))

    print("\n===== TOP 20 WORDS =====")
    for word, freq in vocab_counter.most_common(20):
        print(f"{word}: {freq}")

    print(f"\nГотово. Результаты лежат в папке: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
