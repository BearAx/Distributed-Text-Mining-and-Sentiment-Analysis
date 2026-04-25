# Distributed Text Mining and Sentiment Analysis

## Project Overview

This project implements a **distributed text mining pipeline** for **lexicon-based sentiment analysis** using a **MapReduce-style workflow**.

The pipeline processes a labeled dataset of financial news statements, performs text preprocessing, applies a **pre-trained sentiment lexicon**, generates document-level sentiment predictions, aggregates results with mapper/reducer logic, benchmarks parallel execution, and validates the final outputs with summary metrics.

## Pipeline

```text
Data → Preprocessing → Lexicon Scoring → Mapper → Parallel Execution → Reducer → Metrics
```

## Dataset

The project uses the file:

- `data/all-data.csv`

The dataset contains sentiment-labeled financial text statements and is used throughout the pipeline for preprocessing, sentiment prediction, and validation.

## Team Roles

### 1. Data Engineer
**Owner:** [BearAx](https://github.com/BearAx)  
**Status:** `Completed`

**Responsibilities**
- load the dataset;
- handle file encoding and format issues;
- clean the text;
- tokenize documents;
- generate vocabulary and preprocessing statistics.

**Implemented outputs**
- `output_data/cleaned_dataset.csv`
- `output_data/tokens.json`
- `output_data/vocabulary.json`
- `output_data/top_words.csv`
- `output_data/summary.json`

---

### 2. Lexicon Specialist
**Owner:** [Telman3000](https://github.com/Telman3000)  
**Status:** `Completed`

**Responsibilities**
- define or provide a pre-trained sentiment lexicon;
- assign token-level sentiment scores;
- compute document-level sentiment scores;
- classify each document as `positive`, `negative`, or `neutral`.

**Implemented outputs**
- `output_lexicon/scored_documents.json`
- `output_lexicon/sentiment_summary.json`

---

### 3. MapReduce Developer
**Owner:** [LeoPython2006](https://github.com/LeoPython2006)  
**Status:** `Completed`

**Responsibilities**
- implement the mapper;
- implement the reducer;
- produce document-level map results;
- aggregate sentiment counts through reducer logic.

**Implemented outputs**
- `output_mapreduce/mapper_results.json`
- `output_mapreduce/reducer_summary.json`

---

### 4. Parallelization Engineer
**Owner:** [uSs3ewa](https://github.com/uSs3ewa)  
**Status:** `Completed`

**Responsibilities**
- split the dataset into chunks;
- execute document scoring in parallel;
- compare sequential and parallel runtime;
- save chunk-level and benchmark results.

**Implemented outputs**
- `output_parallel/parallel_scored_documents.json`
- `output_parallel/parallel_sentiment_summary.json`
- `output_parallel/chunk_level_results.json`
- `output_parallel/runtime_results.csv`

---

### 5. Validation & Metrics
**Owner:** [Mysteri0K1ng](https://github.com/Mysteri0K1ng)  
**Status:** `Completed`

**Responsibilities**
- verify correctness of predictions;
- compute accuracy if labels exist;
- generate summary statistics;
- save validated document-level results and evaluation metrics.

**Implemented outputs**
- `output_validation/validated_predictions.json`
- `output_validation/metrics_summary.json`

---

## Contribution Table

| Stage | Role | Main Work | Main Outputs | Owner | Status |
|------|------|-----------|--------------|-------|--------|
| 1 | Data Engineer | Dataset loading, cleaning, tokenization | `output_data/*` | [BearAx](https://github.com/BearAx) | `Done` |
| 2 | Lexicon Specialist | Lexicon scoring and sentiment assignment | `output_lexicon/*` | [Telman3000](https://github.com/Telman3000) | `Done` |
| 3 | MapReduce Developer | Mapper and reducer implementation | `output_mapreduce/*` | [LeoPython2006](https://github.com/LeoPython2006) | `Done` |
| 4 | Parallelization Engineer | Chunking, multiprocessing, runtime benchmarking | `output_parallel/*` | [uSs3ewa](https://github.com/uSs3ewa) | `Done` |
| 5 | Validation & Metrics | Accuracy, class metrics, confusion matrix | `output_validation/*` | [Mysteri0K1ng](https://github.com/Mysteri0K1ng) | `Done` |

## Project Structure

```text
Distributed-Text-Mining-and-Sentiment-Analysis/
│
├── data/
│   ├── all-data.csv
│   └── sentiment_lexicon.json
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── lexicon_scoring.py
│   ├── map_reduce_developer.py
│   ├── parallel_runner.py
│   └── validation_metrics.py
│
├── output_data/
├── output_lexicon/
├── output_mapreduce/
├── output_parallel/
├── output_validation/
│
├── README.md
├── LICENSE
└── .gitignore
```

## Detailed Stage Logic

### Stage 1 — Preprocessing
The preprocessing stage:
1. loads `all-data.csv`;
2. handles encoding and header format;
3. converts text to lowercase;
4. removes punctuation;
5. normalizes whitespace;
6. tokenizes text into word lists;
7. builds a vocabulary and preprocessing summary.

**Observed preprocessing results**
- documents after cleaning: **4838**
- total tokens: **103049**
- unique tokens: **10103**
- average tokens per document: **21.3**

### Stage 2 — Lexicon Scoring
The lexicon stage loads `output_data/tokens.json` and `data/sentiment_lexicon.json`, then:
- scores each token using the sentiment lexicon;
- sums token scores into a document score;
- assigns:
  - `positive` if score > 0
  - `negative` if score < 0
  - `neutral` if score = 0

**Observed lexicon-stage results**
- lexicon terms: **78**
- positive documents: **952**
- negative documents: **272**
- neutral documents: **3614**
- evaluated documents: **4838**
- accuracy: **0.6807**

### Stage 3 — Mapper / Reducer
The MapReduce stage implements:
- a **mapper**, which transforms one tokenized document into a structured prediction record;
- a **reducer**, which aggregates counts across all mapped records.

**Mapper output example**
```json
{
  "doc_id": 24,
  "tokens": ["company", "reported", "profit"],
  "score": 2,
  "predicted_sentiment": "positive",
  "true_label": "positive"
}
```

**Reducer output fields**
- `documents_count`
- `positive_documents`
- `negative_documents`
- `neutral_documents`
- `accuracy`

**Observed reducer-stage results**
- documents_count: **4838**
- positive_documents: **952**
- negative_documents: **272**
- neutral_documents: **3614**
- accuracy: **0.6807**

### Stage 4 — Parallel Execution
The parallel stage reuses the sentiment scoring logic and:
- splits tokenized documents into chunks;
- processes them with multiprocessing;
- computes chunk-level summaries;
- compares sequential and parallel runtime;
- optionally verifies result consistency.

**Observed parallel-stage results**
- workers: **8**
- chunk size: **500**
- sequential time: **0.021747 s**
- parallel time: **0.245139 s**
- speedup: **0.0887x**

> On this dataset and configuration, parallel execution is **slower** than sequential execution because multiprocessing overhead dominates the workload. This does **not** mean the stage is incorrect; it shows a realistic benchmark outcome on a relatively small task.

### Stage 5 — Validation & Metrics
The validation stage:
- reads prediction outputs;
- checks whether ground-truth labels are available;
- computes accuracy and summary counts;
- computes per-class precision, recall, and F1;
- builds a confusion matrix;
- saves validated document-level results.

**Observed validation results**
- evaluated documents: **4838**
- correct predictions: **3293**
- accuracy: **0.6807**
- average tokens per document: **21.3**
- average score: **0.1875**

**Class metrics**
- positive: precision **0.6061**, recall **0.4236**, F1 **0.4987**
- negative: precision **0.6360**, recall **0.2864**, F1 **0.3949**
- neutral: precision **0.7037**, recall **0.8854**, F1 **0.7842**

## End-to-End Output Consistency Check

The current project outputs are internally consistent:

- `output_lexicon/sentiment_summary.json`
- `output_mapreduce/reducer_summary.json`
- `output_parallel/parallel_sentiment_summary.json`
- `output_validation/metrics_summary.json`

All of them agree on the same document count and sentiment distribution:
- documents: **4838**
- positive: **952**
- negative: **272**
- neutral: **3614**
- accuracy: **0.6807**

This is a strong sign that the pipeline stages are aligned correctly.

## How to Run the Project

Run all commands from the **repository root**.

### 1. Preprocessing
```bash
python scripts/data_preprocessing.py
```

### 2. Lexicon scoring
```bash
python scripts/lexicon_scoring.py
```

### 3. Mapper / reducer
```bash
python scripts/map_reduce_developer.py
```

### 4. Parallel execution
```bash
python scripts/parallel_runner.py --workers 4 --chunk-size 500 --verify
```

### 5. Validation and metrics
```bash
python scripts/validation_metrics.py
```

## Design Decisions

### Why lexicon-based sentiment analysis?
A pre-trained lexicon was chosen because:
- it is interpretable;
- it does not require model training;
- it fits a modular MapReduce pipeline;
- it is easy to validate and explain.

### Why MapReduce?
MapReduce matches this task naturally:
- **Map:** process each document independently;
- **Reduce:** aggregate all document-level outputs.

### Why chunk-based parallelism?
Chunking was chosen because:
- it is simple to implement;
- it clearly demonstrates distributed thinking;
- it allows measurable runtime benchmarking.

### Why simple preprocessing?
The preprocessing stage intentionally stays aligned with the assignment:
- lowercase conversion;
- punctuation removal;
- tokenization.

## Strengths

- clear modular architecture;
- separate outputs for each stage;
- reproducible pipeline;
- consistent summary counts across stages;
- built-in validation and benchmarking.

## Limitations

- the lexicon is relatively small (**78 terms**);
- the approach may miss context, sarcasm, and domain nuance;
- parallel processing is slower than sequential processing on this dataset due to overhead;
- preprocessing is intentionally basic and does not include lemmatization or stopword removal.

## Conclusion

The current project is **functionally complete** for the required assignment scope.

It correctly implements:
- preprocessing;
- lexicon-based sentiment scoring;
- mapper/reducer aggregation;
- parallel execution benchmarking;
- validation and metrics.

The README is now aligned with the actual repository structure, actual script names, actual output folders, and the current measured results.
