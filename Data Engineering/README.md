# Data Engineering

This module completes the **Data Engineer** role in the pipeline:

**Data → Preprocessing → Lexicon scoring → Mapper → Parallel execution → Reducer → Metrics**

## What it does
- loads dataset from `.txt`, `.csv`, `.json`, `.jsonl`
- cleans text:
  - lowercase
  - removes punctuation
  - normalizes spaces
- tokenizes text into a list of words
- exports the result as JSON for the next team member

## Output format
Each processed document looks like this:

```json
{
  "doc_id": 1,
  "original_text": "I love this product!",
  "cleaned_text": "i love this product",
  "tokens": ["i", "love", "this", "product"]
}
```

## Run examples

### CSV
```bash
python data_engineer_preprocessing.py reviews.csv --text-column review --output tokens.json --preview
```

### JSONL
```bash
python data_engineer_preprocessing.py tweets.jsonl --output tokens.json --preview
```

### TXT (one document per line)
```bash
python data_engineer_preprocessing.py headlines.txt --output tokens.json --preview
```

## Good default text column names
If `--text-column` is omitted, the script tries to auto-detect one of these keys:
- `text`
- `review`
- `tweet`
- `headline`
- `content`
- `body`
- `message`
- `sentence`
- `document`

## Handoff to next role
The next teammate can directly use the `tokens` array from the output JSON for lexicon-based sentiment scoring.
