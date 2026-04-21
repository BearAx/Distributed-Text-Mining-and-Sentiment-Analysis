# Distributed Text Mining and Sentiment Analysis

This stage completes the **Data Engineer** part of the project pipeline:

**Data → Preprocessing → Lexicon scoring → Mapper → Parallel execution → Reducer → Metrics**

---

## Overview

At this stage, the raw dataset was loaded and prepared for the next steps in the pipeline.

The following preprocessing operations were applied to each document:

1. **Load the dataset**
2. **Clean the text**
   - convert to lowercase
   - remove punctuation
   - normalize extra spaces
3. **Tokenize the text**
   - split each cleaned document into a list of words
4. **Save the outputs**
   - cleaned dataset
   - tokenized documents
   - corpus vocabulary
   - summary statistics

---

## Input File

Main input dataset:

- `all-data.csv`

For this dataset, the following fields were used:

- `label` — original sentiment label from the dataset
- `text` — original document text

---

## Output Files

After running the preprocessing script, the following files were generated inside:

- `output_data_engineer/cleaned_dataset.csv`
- `output_data_engineer/tokens.json`
- `output_data_engineer/vocabulary.json`
- `output_data_engineer/top_words.csv`
- `output_data_engineer/summary.json`

---

## Output File Descriptions

### 1. `cleaned_dataset.csv`

Main structured output of the preprocessing stage.

Contains:

- `doc_id` — unique document identifier
- `label` — original label from the dataset
- `text` — original raw text
- `cleaned_text` — processed text after cleaning
- `tokens` — token list for the document
- `token_count` — number of tokens in the document

Example structure:

| doc_id | label    | text                        | cleaned_text                | tokens                                   | token_count |
|-------:|----------|-----------------------------|-----------------------------|------------------------------------------|------------:|
| 1      | positive | The company posted profits. | the company posted profits  | ['the', 'company', 'posted', 'profits']  | 4           |

This is the most useful file if you need text, labels, and tokens together.

---

### 2. `tokens.json`

This file stores the tokenized representation of each document.

Example item:

```json
{
  "doc_id": 1,
  "original_text": "The company posted strong profits.",
  "cleaned_text": "the company posted strong profits",
  "tokens": ["the", "company", "posted", "strong", "profits"],
  "label": "positive"
}
