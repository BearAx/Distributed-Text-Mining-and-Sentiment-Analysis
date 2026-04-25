import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

# Allow importing sibling script modules when executed from repo root:
#   python scripts/parallel_runner.py
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lexicon_scoring import (  # noqa: E402
    load_lexicon,
    load_tokens,
    score_to_label,
    score_tokens,
)


SentimentCounts = Dict[str, int]


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: int
    records: List[Dict]
    counts: SentimentCounts
    evaluated: int
    correct: int


def chunked(items: List[Dict], chunk_size: int) -> List[List[Dict]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def _init_counts() -> SentimentCounts:
    return {"positive": 0, "negative": 0, "neutral": 0}


def map_chunk(args: Tuple[int, List[Dict], Dict[str, int]]) -> ChunkResult:
    chunk_id, records, lexicon = args

    counts = _init_counts()
    evaluated = 0
    correct = 0
    out_records: List[Dict] = []

    for record in records:
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
            "predicted_sentiment": predicted,
        }

        true_label = record.get("label")
        if isinstance(true_label, str):
            normalized_label = true_label.strip().lower()
            out["true_label"] = normalized_label
            if normalized_label in counts:
                evaluated += 1
                if normalized_label == predicted:
                    correct += 1

        out_records.append(out)

    return ChunkResult(
        chunk_id=chunk_id,
        records=out_records,
        counts=counts,
        evaluated=evaluated,
        correct=correct,
    )


def reduce_chunks(chunks: Iterable[ChunkResult]) -> Tuple[List[Dict], Dict]:
    all_records: List[Dict] = []
    total_counts = _init_counts()
    evaluated = 0
    correct = 0

    for chunk in chunks:
        all_records.extend(chunk.records)
        for k in total_counts.keys():
            total_counts[k] += int(chunk.counts.get(k, 0))
        evaluated += int(chunk.evaluated)
        correct += int(chunk.correct)

    accuracy: Optional[float]
    accuracy = round(correct / evaluated, 4) if evaluated > 0 else None

    summary = {
        "documents_count": len(all_records),
        "positive_documents": total_counts["positive"],
        "negative_documents": total_counts["negative"],
        "neutral_documents": total_counts["neutral"],
        "evaluated_documents": evaluated,
        "accuracy": accuracy,
    }

    return all_records, summary


def _records_signature(records: List[Dict]) -> Dict[int, Tuple[int, str]]:
    """
    Stable comparison between sequential and parallel outputs.
    Uses only deterministic fields: (doc_id -> (score, predicted_sentiment)).
    """
    sig: Dict[int, Tuple[int, str]] = {}
    for r in records:
        doc_id = int(r["doc_id"])
        sig[doc_id] = (int(r["score"]), str(r["predicted_sentiment"]))
    return sig


def run_sequential(token_records: List[Dict], lexicon: Dict[str, int]) -> Tuple[List[Dict], Dict, float]:
    start = perf_counter()
    chunk = map_chunk((0, token_records, lexicon))
    records, summary = reduce_chunks([chunk])
    elapsed = perf_counter() - start
    return records, summary, elapsed


def run_parallel(
    token_records: List[Dict],
    lexicon: Dict[str, int],
    workers: int,
    chunk_size: int,
) -> Tuple[List[Dict], Dict, float, List[Dict]]:
    from multiprocessing import get_context

    if workers <= 0:
        raise ValueError("workers must be > 0")

    chunks = chunked(token_records, chunk_size=chunk_size)
    tasks = [(i, ch, lexicon) for i, ch in enumerate(chunks)]

    start = perf_counter()
    if workers == 1:
        chunk_results = [map_chunk(t) for t in tasks]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            chunk_results = pool.map(map_chunk, tasks)

    records, summary = reduce_chunks(chunk_results)
    elapsed = perf_counter() - start

    chunk_level = []
    for cr in chunk_results:
        chunk_level.append(
            {
                "chunk_id": cr.chunk_id,
                "documents": len(cr.records),
                "positive": cr.counts["positive"],
                "negative": cr.counts["negative"],
                "neutral": cr.counts["neutral"],
            }
        )

    return records, summary, elapsed, chunk_level


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: run mapper logic in parallel and benchmark speedup."
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=BASE_DIR / "output_data" / "tokens.json",
        help="Path to tokenized documents (default: output_data/tokens.json)",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=BASE_DIR / "data" / "sentiment_lexicon.json",
        help="Path to sentiment lexicon JSON (default: data/sentiment_lexicon.json)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=BASE_DIR / "output_parallel",
        help="Output directory (default: output_parallel)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of worker processes (default: cpu_count//2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Documents per chunk (default: 500)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify sequential vs parallel outputs match",
    )
    parser.add_argument(
        "--sweep-workers",
        type=str,
        default="1,2,4",
        help="Comma-separated worker counts for benchmark CSV (default: 1,2,4)",
    )
    parser.add_argument(
        "--sweep-chunk-sizes",
        type=str,
        default="200,500,1000",
        help="Comma-separated chunk sizes for benchmark CSV (default: 200,500,1000)",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenized documents from {args.tokens}")
    token_records = load_tokens(args.tokens)
    print(f"Loaded {len(token_records)} documents")

    print(f"Loading sentiment lexicon from {args.lexicon}")
    lexicon = load_lexicon(args.lexicon)
    print(f"Loaded {len(lexicon)} lexicon terms")

    seq_records, seq_summary, seq_time = run_sequential(token_records, lexicon)
    print(f"Sequential time: {seq_time:.4f}s")

    par_records, par_summary, par_time, chunk_level = run_parallel(
        token_records,
        lexicon,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )
    speedup = seq_time / par_time if par_time > 0 else math.inf
    print(
        f"Parallel time (workers={args.workers}, chunk_size={args.chunk_size}): {par_time:.4f}s")
    print(f"Speedup vs sequential: {speedup:.2f}x")

    if args.verify:
        if _records_signature(seq_records) != _records_signature(par_records):
            raise SystemExit(
                "Verification failed: sequential and parallel outputs differ.")
        print("Verification OK: sequential and parallel outputs match.")

    # Save outputs for the chosen configuration
    with open(args.outdir / "parallel_scored_documents.json", "w", encoding="utf-8") as f:
        json.dump(par_records, f, ensure_ascii=False, indent=2)

    parallel_summary_out = dict(par_summary)
    parallel_summary_out.update(
        {
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "sequential_time_sec": round(seq_time, 6),
            "parallel_time_sec": round(par_time, 6),
            "speedup": round(speedup, 4) if math.isfinite(speedup) else None,
        }
    )

    with open(args.outdir / "parallel_sentiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(parallel_summary_out, f, ensure_ascii=False, indent=2)

    with open(args.outdir / "chunk_level_results.json", "w", encoding="utf-8") as f:
        json.dump(chunk_level, f, ensure_ascii=False, indent=2)

    # Benchmark sweep
    sweep_workers = [int(x.strip())
                     for x in args.sweep_workers.split(",") if x.strip()]
    sweep_chunk_sizes = [int(x.strip())
                         for x in args.sweep_chunk_sizes.split(",") if x.strip()]

    bench_rows = []
    for w in sweep_workers:
        for cs in sweep_chunk_sizes:
            _, _, t, _ = run_parallel(
                token_records, lexicon, workers=w, chunk_size=cs)
            bench_rows.append(
                {
                    "workers": w,
                    "chunk_size": cs,
                    "parallel_time_sec": round(t, 6),
                }
            )

    bench_path = args.outdir / "runtime_results.csv"
    with open(bench_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["workers", "chunk_size", "parallel_time_sec"])
        writer.writeheader()
        writer.writerows(bench_rows)

    print(
        f"Saved outputs to {args.outdir}/ (parallel_scored_documents.json, summaries, runtime_results.csv)")


if __name__ == "__main__":
    main()
