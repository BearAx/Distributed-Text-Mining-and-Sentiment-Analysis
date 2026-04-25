import json
from pathlib import Path
from typing import List, Dict, Tuple

# Re‑use the existing lexicon and token loading utilities
TOKENS_FILE = Path('output_data/tokens.json')
LEXICON_FILE = Path('data/sentiment_lexicon.json')


def load_tokens(tokens_path: Path) -> List[Dict]:
    """Load tokenized documents produced by ``data_preprocessing.py``.

    Returns a list of records where each record contains at least ``doc_id``
    and ``tokens``.
    """
    with open(tokens_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_lexicon(lexicon_path: Path) -> Dict[str, int]:
    """Load the sentiment lexicon and flatten any nested blocks.

    The lexicon may be a flat ``word -> score`` mapping or a mapping with
    ``positive``/``negative`` sub‑objects. All keys are lower‑cased for
    case‑insensitive lookup.
    """
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    flattened: Dict[str, int] = {}
    if any(isinstance(v, dict) for v in raw.values()):
        for block in raw.values():
            if isinstance(block, dict):
                for word, score in block.items():
                    flattened[word.lower()] = int(score)
    else:
        for word, score in raw.items():
            flattened[word.lower()] = int(score)
    return flattened


def score_tokens(tokens: List[str], lexicon: Dict[str, int]) -> int:
    """Return a single sentiment score for a list of tokens.

    Positive scores increase the total, negative scores decrease it.
    """
    total = 0
    for token in tokens:
        total += lexicon.get(token.lower(), 0)
    return total


class MapReduceDeveloper:
    """A tiny Map‑Reduce style helper for sentiment analysis.

    * ``map`` converts a single document (a dict with ``doc_id`` and ``tokens``)
      into a sentiment label (``positive``, ``negative`` or ``neutral``).
    * ``reduce`` aggregates the results of many ``map`` calls into a summary
      dictionary containing counts per sentiment and overall accuracy if the
      true label is present.
    """

    def __init__(self) -> None:
        self.lexicon = load_lexicon(LEXICON_FILE)

    @staticmethod
    def _score_to_label(score: int) -> str:
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def map(self, document: Dict) -> Dict:
        """Process a single document and return a result dict.

        Expected ``document`` shape::

            {
                "doc_id": int,
                "tokens": List[str],
                "label": "positive"|"negative"|"neutral"  # optional
            }
        """
        doc_id = document.get("doc_id")
        tokens = document.get("tokens", [])
        score = score_tokens(tokens, self.lexicon)
        predicted = self._score_to_label(score)
        result = {
            "doc_id": doc_id,
            "predicted": predicted,
            "score": score,
        }
        if "label" in document:
            result["true_label"] = document["label"].lower()
        return result

    def reduce(self, mapped_results: List[Dict]) -> Dict:
        """Aggregate a list of ``map`` results.

        Returns a summary with counts per sentiment, optionally accuracy
        if true labels are present.
        """
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        evaluated = 0
        correct = 0
        for r in mapped_results:
            pred = r.get("predicted")
            if pred in counts:
                counts[pred] += 1
            true = r.get("true_label")
            if true:
                evaluated += 1
                if true == pred:
                    correct += 1
        summary: Dict = {
            "documents": len(mapped_results),
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
        }
        if evaluated:
            summary["accuracy"] = round(correct / evaluated, 4)
        return summary
