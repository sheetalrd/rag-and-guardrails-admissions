"""Measure how well the search pipeline finds the right chunks.

Loads data/evalset.json, runs the search pipeline for every question, and prints all six
retrieval metrics for K = 1, 3, 5 and 10, followed by the three questions the search
handled worst.

Run from the project root:

    python scripts/05_evaluate.py                # with query rewriting (calls the LLM)
    python scripts/05_evaluate.py --no-rewrite   # search with the question exactly as typed
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from admissions_agent.evaluation.metrics import (  # noqa: E402
    average_precision,
    f1_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from admissions_agent.rag.query_rewriter import rewrite_query  # noqa: E402
from admissions_agent.rag.retriever import retrieve  # noqa: E402

EVAL_SET_PATH = PROJECT_ROOT / "data" / "evalset.json"
K_VALUES = [1, 3, 5, 10]
WORST_K = 5  # the K used to rank the questions from worst to best
WORST_COUNT = 3

METRICS = {
    "Precision": precision_at_k,
    "Recall": recall_at_k,
    "F1": f1_at_k,
    "MRR": reciprocal_rank,
    "MAP": average_precision,
    "NDCG": ndcg_at_k,
}


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def run_search(eval_set: list[dict], use_rewrite: bool) -> list[dict]:
    """Search once per question with the largest K; smaller K values reuse the same results."""
    results = []
    for number, item in enumerate(eval_set, start=1):
        print(f"  searching {number}/{len(eval_set)}", end="\r")
        question = item["question"]
        search_query = rewrite_query(question) if use_rewrite else question
        chunks = retrieve(search_query, top_k=max(K_VALUES))
        results.append(
            {
                "question": question,
                "search_query": search_query,
                "retrieved": [chunk.id for chunk in chunks],
                "relevant": item["relevant"],
            }
        )
    print(" " * 40, end="\r")
    return results


def print_metrics_table(results: list[dict]) -> None:
    """Print one row per K. The layout is a Markdown table, so it can be pasted into the report."""
    print("| K  | " + " | ".join(f"{name:>9}" for name in METRICS) + " |")
    print("|----|" + "|".join("-" * 11 for _ in METRICS) + "|")
    for k in K_VALUES:
        averages = [
            mean([metric(result["retrieved"], result["relevant"], k) for result in results])
            for metric in METRICS.values()
        ]
        print(f"| {k:<2} | " + " | ".join(f"{value:>9.3f}" for value in averages) + " |")


def badness(result: dict) -> tuple[float, float, float]:
    """Sort key: lowest NDCG first, then lowest Recall, then lowest reciprocal rank."""
    retrieved, relevant = result["retrieved"], result["relevant"]
    return (
        ndcg_at_k(retrieved, relevant, WORST_K),
        recall_at_k(retrieved, relevant, WORST_K),
        reciprocal_rank(retrieved, relevant, WORST_K),
    )


def print_worst_questions(results: list[dict]) -> None:
    for result in sorted(results, key=badness)[:WORST_COUNT]:
        retrieved, relevant = result["retrieved"], result["relevant"]
        print()
        print(f"Question: {result['question']}")
        if result["search_query"] != result["question"]:
            print(f"  search query: {result['search_query']}")
        print(
            f"  NDCG@{WORST_K} {ndcg_at_k(retrieved, relevant, WORST_K):.3f}"
            f"   Recall@{WORST_K} {recall_at_k(retrieved, relevant, WORST_K):.3f}"
            f"   Precision@{WORST_K} {precision_at_k(retrieved, relevant, WORST_K):.3f}"
        )
        print("  expected:")
        for chunk_id, grade in relevant.items():
            found = "found" if chunk_id in retrieved[:WORST_K] else "missed"
            print(f"    grade {grade}  {chunk_id}  ({found})")
        print(f"  retrieved (top {WORST_K}):")
        for rank, chunk_id in enumerate(retrieved[:WORST_K], start=1):
            grade = relevant.get(chunk_id, 0)
            label = f"grade {grade}" if grade > 0 else "not relevant"
            print(f"    {rank}. {chunk_id}  ({label})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the search pipeline on data/eval_set.json.")
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="skip the query rewriting step and search with the question as typed",
    )
    args = parser.parse_args()

    eval_set = load_eval_set()
    mode = "without query rewriting" if args.no_rewrite else "with query rewriting"
    print(f"Evaluating {len(eval_set)} questions {mode}")

    results = run_search(eval_set, use_rewrite=not args.no_rewrite)

    print()
    print(f"Retrieval metrics ({mode}), averaged over {len(results)} questions")
    print()
    print_metrics_table(results)

    print()
    print(f"The {WORST_COUNT} questions with the worst scores (ranked by NDCG@{WORST_K})")
    print_worst_questions(results)


if __name__ == "__main__":
    main()