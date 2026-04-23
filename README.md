# Distributed Text Mining and Sentiment Analysis

## Project Overview

This project implements a **distributed text mining pipeline** for **lexicon-based sentiment analysis** using a **MapReduce-style architecture**.

The goal is to process a dataset of text documents, apply basic preprocessing, classify each document into **positive**, **negative**, or **neutral** sentiment using a **pre-trained lexicon**, run the document-level logic in a parallelizable way, and aggregate the final results.

---

## Project Objective

The system is designed to:

- load a real text dataset;
- preprocess each document;
- tokenize the cleaned text;
- classify sentiment with a **pre-trained sentiment lexicon**;
- process documents with a **mapper**;
- aggregate all outputs with a **reducer**;
- support **parallel execution**;
- evaluate correctness and produce summary statistics.

### Expected Final Output

- one **sentiment label per document**;
- a final summary of:
  - positive documents,
  - negative documents,
  - neutral documents;
- optional validation metrics such as **accuracy** if labels are available.

---

## Full Pipeline

```text
Data → Preprocessing → Lexicon Scoring → Mapper → Parallel Execution → Reducer → Metrics
```

---

## Team Structure and Responsibilities

### 1. Data Engineer

**Responsibilities:**

- load the dataset;
- handle formatting or encoding issues;
- clean the text;
- tokenize the documents;
- prepare outputs for downstream stages.

**Main preprocessing operations:**

- convert text to lowercase;
- remove punctuation;
- normalize whitespace;
- split text into tokens.

**Inputs:**

- raw dataset file (`.csv`, `.json`, `.jsonl`, or `.txt`)

**Outputs:**

- cleaned dataset;
- token list per document;
- vocabulary / word-frequency artifacts.

**Example output:**

```json
{
  "doc_id": 1,
  "original_text": "The company posted strong profits.",
  "cleaned_text": "the company posted strong profits",
  "tokens": ["the", "company", "posted", "strong", "profits"]
}
```

**Owner:** [BearAx](https://github.com/BearAx)

**Status:** `Completed`

---

### 2. Lexicon Specialist

**Responsibilities:**

- choose a suitable **pre-trained sentiment lexicon**;
- define token-level sentiment scores;
- implement document-level scoring;
- map the final score into:
  - `positive`,
  - `negative`,
  - `neutral`.

**Inputs:**

- tokenized documents;
- sentiment lexicon.

**Outputs:**

- document sentiment score;
- predicted sentiment label.

**Example lexicon logic:**

- positive word → `+1`
- negative word → `-1`
- unknown / neutral word → `0`

**Example classification rule:**

- score > 0 → `positive`
- score < 0 → `negative`
- score = 0 → `neutral`

**Example output:**

```json
{
  "doc_id": 1,
  "tokens": ["profit", "growth", "strong"],
  "score": 3,
  "predicted_sentiment": "positive"
}
```

**Owner:** [Telman3000](https://github.com/Telman3000)

**Status:** `Completed`

---

### 3. MapReduce Developer

**Responsibilities:**

- implement the **mapper**;
- implement the **reducer**;
- define intermediate data structure;
- standardize document-level and summary-level outputs.

**Inputs:**

- scored or tokenized documents.

**Outputs:**

- mapper results;
- reducer summary.

**Example mapper output:**

```json
{
  "doc_id": 17,
  "predicted_sentiment": "negative"
}
```

**Example reducer output:**

```json
{
  "positive": 120,
  "negative": 75,
  "neutral": 34,
  "total_documents": 229
}
```

**Owner:** [LeoPython2006](https://github.com/LeoPython2006)

**Status:** `[Completed / In Progress / Pending]`

---

### 4. Parallelization Engineer

**Responsibilities:**

- split the dataset into chunks;
- execute mapper logic in parallel;
- compare runtime for different worker counts or chunk sizes;
- verify that sequential and parallel results are consistent.

**Inputs:**

- mapper function;
- prepared/scored document data.

**Outputs:**

- chunk-level results;
- runtime comparison;
- scalability observations.

**Owner:** [uSs3ewa](https://github.com/uSs3ewa)

**Status:** `Completed`

---

### 5. Validation and Metrics

**Responsibilities:**

- verify pipeline correctness;
- compare predictions with true labels if labels exist;
- compute evaluation metrics;
- produce summary statistics for the report and presentation.

**Possible metrics:**

- accuracy;
- class distribution;
- total documents processed;
- average tokens per document;
- runtime under different chunk configurations.

**Owner:** [Mysteri0K1ng](https://github.com/Mysteri0K1ng)

**Status:** `[Completed / In Progress / Pending]`

---

## Contribution Table

| Stage | Role | Main Work | Main Inputs | Main Outputs | Owner | Status |
|------|------|-----------|-------------|--------------|-------|--------|
| 1 | Data Engineer | Load, clean, tokenize text | Raw dataset | Cleaned text, tokens | [BearAx](https://github.com/BearAx) | `Done` |
| 2 | Lexicon Specialist | Score tokens with lexicon | Tokens, lexicon | Score, predicted sentiment | [uSs3ewa](https://github.com/uSs3ewa) | `[Status]` |
| 3 | MapReduce Developer | Implement mapper and reducer | Scored documents | Aggregated counts | [LeoPython2006](https://github.com/LeoPython2006) | `[Status]` |
| 4 | Parallelization Engineer | Chunking and parallel execution | Mapper logic | Runtime/scalability results | [Telman3000](https://github.com/Telman3000) | `Done` |
| 5 | Validation & Metrics | Accuracy and summary statistics | Predictions, labels | Metrics, evaluation report | [Mysteri0K1ng](https://github.com/Mysteri0K1ng) | `[Status]` |

---

## Detailed Pipeline Logic

### Stage 1 — Data Loading

The dataset is loaded into memory from a supported format such as CSV, JSON, JSONL, or TXT.

**Typical fields:**

- `doc_id`
- `text`
- `label` *(optional, used for evaluation)*

**Important considerations:**

- encoding issues may occur;
- some datasets may not contain headers;
- empty rows must be removed;
- duplicated documents may need to be handled.

---

### Stage 2 — Preprocessing

Each document is transformed into a clean and consistent format.

**Processing steps:**

1. convert text to lowercase;
2. remove punctuation;
3. normalize multiple spaces;
4. tokenize into a list of words.

**Example:**

Original text:

```text
The company posted strong profits!
```

Cleaned text:

```text
the company posted strong profits
```

Tokens:

```python
["the", "company", "posted", "strong", "profits"]
```

**Why simple preprocessing was chosen:**

- it directly satisfies the assignment;
- it is transparent and easy to explain;
- it keeps the pipeline readable;
- it avoids unnecessary NLP complexity.

**Not included at this stage:**

- stemming;
- lemmatization;
- stopword removal;
- POS tagging;
- n-grams.

---

### Stage 3 — Lexicon-Based Sentiment Scoring

This stage performs the core classification logic.

The project requirement states that sentiment should be classified using a **pre-trained lexicon**. That means the sentiment signal comes from an already existing dictionary of words rather than from a trained machine learning model.

**Example lexicon:**

```json
{
  "good": 1,
  "great": 1,
  "excellent": 1,
  "profit": 1,
  "growth": 1,
  "bad": -1,
  "poor": -1,
  "loss": -1,
  "decline": -1
}
```

**Scoring process:**

- each token is looked up in the lexicon;
- token scores are summed;
- the total score determines document sentiment.

**Example:**

Tokens:

```python
["company", "reported", "strong", "profit"]
```

Token scores:

- `company` → `0`
- `reported` → `0`
- `strong` → `+1`
- `profit` → `+1`

Total score:

```python
2
```

Predicted sentiment:

```text
positive
```

**Classification rule:**

- score > 0 → `positive`
- score < 0 → `negative`
- score = 0 → `neutral`

---

### Stage 4 — Mapper

The mapper processes each document independently and emits a structured document-level result.

**Example mapper output:**

```json
{
  "doc_id": 24,
  "score": -2,
  "predicted_sentiment": "negative"
}
```

**Why this is a good map stage:**

- each document can be processed independently;
- no cross-document dependency is needed;
- it fits the MapReduce model naturally.

---

### Stage 5 — Parallel Execution

The dataset is split into chunks and processed in parallel.

**Example chunking strategy:**

If the dataset contains 1000 documents:

- chunk 1 → docs 1–250
- chunk 2 → docs 251–500
- chunk 3 → docs 501–750
- chunk 4 → docs 751–1000

Each chunk runs the same mapper logic independently.

**Benefits:**

- faster execution on larger datasets;
- better demonstration of distributed processing;
- measurable scalability;
- cleaner separation of processing units.

---

### Stage 6 — Reducer

The reducer aggregates the mapper outputs into corpus-level summary results.

**Example reducer input:**

```json
[
  {"doc_id": 1, "predicted_sentiment": "positive"},
  {"doc_id": 2, "predicted_sentiment": "neutral"},
  {"doc_id": 3, "predicted_sentiment": "negative"}
]
```

**Example reducer output:**

```json
{
  "positive": 1,
  "negative": 1,
  "neutral": 1,
  "total_documents": 3
}
```

**Optional reducer extensions:**

- class percentages;
- average sentiment score;
- per-chunk consistency checks.

---

### Stage 7 — Validation and Metrics

The final stage measures correctness and produces evaluation outputs.

**If labels are available:**

- compare predicted sentiment with true labels;
- compute accuracy;
- inspect misclassified examples.

**If labels are not available:**

- report prediction distribution;
- report total processed documents;
- report runtime and scalability results;
- check output consistency.

**Validation questions:**

- Does preprocessing work correctly?
- Does lexicon scoring behave as expected?
- Does parallel execution preserve the same result as sequential execution?
- Are final counts internally consistent?

---

## Data Flow Between Stages

### Data Engineer → Lexicon Specialist

**Passes:**

- cleaned text;
- tokenized documents;
- optional labels.

**Recommended files:**

- `cleaned_dataset.csv`
- `tokens.json`

---

### Lexicon Specialist → MapReduce Developer

**Passes:**

- lexicon;
- scoring logic;
- predicted sentiment per document.

**Recommended files:**

- `lexicon.json`
- `scored_documents.json`

---

### MapReduce Developer → Parallelization Engineer

**Passes:**

- mapper function;
- reducer function;
- intermediate document-level structure.

**Recommended files:**

- `mapper.py`
- `reducer.py`

---

### Parallelization Engineer → Validation and Metrics

**Passes:**

- final predictions;
- reduced summary;
- runtime/scalability results.

**Recommended files:**

- `document_sentiments.csv`
- `reduced_summary.json`
- `runtime_results.csv`

---

## Example End-to-End Data Transformation

### Input dataset record

```json
{
  "doc_id": 3,
  "text": "The firm reported heavy losses.",
  "label": "negative"
}
```

### After preprocessing

```json
{
  "doc_id": 3,
  "cleaned_text": "the firm reported heavy losses",
  "tokens": ["the", "firm", "reported", "heavy", "losses"]
}
```

### After lexicon scoring

```json
{
  "doc_id": 3,
  "tokens": ["the", "firm", "reported", "heavy", "losses"],
  "score": -2,
  "predicted_sentiment": "negative"
}
```

### After reducer

```json
{
  "positive": 1420,
  "negative": 980,
  "neutral": 740,
  "total_documents": 3140,
  "accuracy": 0.84
}
```

> If accuracy is not implemented yet, the `accuracy` field can be omitted.

---

## Why These Design Choices Were Made

### Why a lexicon-based approach?

A **pre-trained lexicon** was chosen because:

- it is transparent and interpretable;
- it does not require model training;
- it is easy to explain in a written report;
- it fits naturally into a MapReduce pipeline;
- it is suitable for an educational distributed systems project.

### Why MapReduce?

MapReduce is a good fit because document-level sentiment classification is independent for each document.

- **Map:** classify each document separately;
- **Reduce:** aggregate all document-level outputs.

### Why parallel chunking?

Chunk-based parallel processing was chosen because:

- it is simple to implement;
- it simulates distributed execution clearly;
- it makes runtime comparison possible;
- it improves the engineering value of the project.

### Why simple preprocessing?

The assignment explicitly requires:

- dataset loading;
- lowercase conversion;
- punctuation removal;
- tokenization.

So the preprocessing stage intentionally stays simple and aligned with the specification.

---

## How to Run the Project

### Recommended execution order

```bash
python scripts/data_preprocessing.py
python scripts/lexicon_scoring.py
python scripts/mapper.py
python scripts/parallel_runner.py
python scripts/validation_metrics.py
```

### Stage 4 — Parallel execution (implemented)

Runs the document-level scoring in parallel, writes chunk-level results, benchmarks runtime
for different worker counts / chunk sizes, and (optionally) verifies that sequential and
parallel outputs are consistent.

```bash
# from repo root
python scripts/parallel_runner.py --workers 4 --chunk-size 500 --verify
```

Outputs (saved to `output_data/`):

- `parallel_scored_documents.json`
- `parallel_sentiment_summary.json`
- `chunk_level_results.json`
- `runtime_results.csv`

---

### Why did you use a lexicon-based approach instead of machine learning?

Because the project requirement focuses on a **pre-trained lexicon**, and this approach is easier to explain, easier to integrate into a distributed pipeline, and does not require training data.

### Why is MapReduce suitable here?

Because each document can be processed independently during the map phase, and the final results can be aggregated efficiently during the reduce phase.

### Why did you classify into positive / negative / neutral?

Because the project specification requires these sentiment categories, and they provide a clear interpretation of the corpus.

### How do you validate correctness?

By checking preprocessing outputs, verifying lexicon scoring on sample documents, comparing sequential and parallel results, and using labels for accuracy when available.

### What are the limitations?

The lexicon-based approach may miss context, sarcasm, domain-specific nuance, and complex linguistic structure.

---

## Strengths of the Project

- modular architecture;
- clear role separation;
- transparent and explainable logic;
- easy handoff between teammates;
- naturally parallelizable workflow;
- suitable for both implementation and presentation.

---

## Limitations

- lexicon-based scoring is simpler than trained NLP models;
- sentiment can be misclassified if context is subtle;
- preprocessing is basic;
- neutral sentiment can be harder to detect precisely;
- performance depends on lexicon quality.

These limitations should be acknowledged in the report and oral presentation.

---

## Possible Future Improvements

- add stemming or lemmatization;
- remove stopwords;
- use weighted lexicon scores;
- add domain-specific lexicons;
- support n-gram sentiment handling;
- compare against machine learning baselines;
- deploy on Spark or Hadoop.

---

## Current Project Status

| Component | Description | Status |
|----------|-------------|--------|
| Dataset loading | Raw dataset is available and readable | `Done` |
| Preprocessing | Lowercase + punctuation removal + tokenization | `[Status]` |
| Lexicon scoring | Pre-trained lexicon scoring implemented | `[Status]` |
| Mapper | Document-level mapper implemented | `[Status]` |
| Reducer | Aggregation logic implemented | `[Status]` |
| Parallel execution | Chunk-based parallelism tested | `Done` |
| Metrics | Accuracy / summary statistics prepared | `[Status]` |
| Documentation | README and role-specific docs prepared | `Done` |

---

## Conclusion

This project demonstrates a complete workflow for **distributed text mining and sentiment analysis**.

It combines:

- data preprocessing;
- lexicon-based sentiment classification;
- MapReduce logic;
- parallel execution;
- result aggregation;
- validation and reporting.
