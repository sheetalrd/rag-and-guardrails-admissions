"""Retrieval metrics, written in plain Python (no metric libraries).

Conventions shared by every function:

- `retrieved` is the list of chunk ids the retriever returned, best first.
- `relevant` maps chunk id -> relevance grade (2 = fully answers, 1 = partly relevant).
  Ids that are not listed have grade 0, and any grade above 0 counts as relevant.
- `k` is the cut-off: only the first `k` retrieved ids are looked at.

Every metric returns a float between 0.0 and 1.0, and returns 0.0 instead of dividing by zero.
"""

from math import log2

Relevance = dict[str, int]
# One evaluated question: what the retriever returned, and what should have been returned.
Run = tuple[list[str], Relevance]


def top_k(retrieved: list[str], k: int) -> list[str]:
    """Return the first `k` retrieved ids, dropping repeats so no chunk is counted twice."""
    if k < 1:
        raise ValueError("k must be at least 1")

    seen = set()
    unique = []
    for chunk_id in retrieved:
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique[:k]


def relevant_ids(relevant: Relevance) -> set[str]:
    """Return the ids that count as relevant (grade above 0)."""
    return {chunk_id for chunk_id, grade in relevant.items() if grade > 0}


def count_hits(retrieved: list[str], relevant: Relevance, k: int) -> int:
    """Count how many of the top `k` results are relevant."""
    wanted = relevant_ids(relevant)
    return sum(1 for chunk_id in top_k(retrieved, k) if chunk_id in wanted)


def precision_at_k(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """Relevant results in the top K, divided by K.

    The divisor is always K, so returning fewer than K results lowers precision,
    and returning nothing gives 0.0.
    """
    return count_hits(retrieved, relevant, k) / k


def recall_at_k(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """Relevant results in the top K, divided by the total number of relevant chunks."""
    total_relevant = len(relevant_ids(relevant))
    if total_relevant == 0:
        return 0.0
    return count_hits(retrieved, relevant, k) / total_relevant


def f1_at_k(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K: 2 x P x R / (P + R)."""
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """1 / rank of the first relevant result in the top K, or 0.0 if there is none."""
    wanted = relevant_ids(relevant)
    for rank, chunk_id in enumerate(top_k(retrieved, k), start=1):
        if chunk_id in wanted:
            return 1 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """Sum of Precision@i at every rank i that holds a relevant result, divided by the total relevant chunks."""
    wanted = relevant_ids(relevant)
    if not wanted:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(top_k(retrieved, k), start=1):
        if chunk_id in wanted:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(wanted)


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain: sum over ranks i of grade_i / log2(i + 1)."""
    return sum(grade / log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(retrieved: list[str], relevant: Relevance, k: int) -> float:
    """DCG@K divided by IDCG@K, where IDCG@K is the DCG of the perfect ordering."""
    actual_grades = [relevant.get(chunk_id, 0) for chunk_id in top_k(retrieved, k)]
    ideal_grades = sorted((grade for grade in relevant.values() if grade > 0), reverse=True)[:k]

    ideal = dcg(ideal_grades)
    if ideal == 0:
        return 0.0
    return dcg(actual_grades) / ideal


def mean(values: list[float]) -> float:
    """Average of a list, or 0.0 for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_reciprocal_rank(runs: list[Run], k: int) -> float:
    """MRR: average over questions of the reciprocal rank."""
    return mean([reciprocal_rank(retrieved, relevant, k) for retrieved, relevant in runs])


def mean_average_precision(runs: list[Run], k: int) -> float:
    """MAP: average over questions of the average precision."""
    return mean([average_precision(retrieved, relevant, k) for retrieved, relevant in runs])


def evaluate(runs: list[Run], k: int) -> dict[str, float]:
    """Average every metric over all questions. Each run is (retrieved ids, relevant grades)."""
    return {
        "precision": mean([precision_at_k(retrieved, relevant, k) for retrieved, relevant in runs]),
        "recall": mean([recall_at_k(retrieved, relevant, k) for retrieved, relevant in runs]),
        "f1": mean([f1_at_k(retrieved, relevant, k) for retrieved, relevant in runs]),
        "mrr": mean_reciprocal_rank(runs, k),
        "map": mean_average_precision(runs, k),
        "ndcg": mean([ndcg_at_k(retrieved, relevant, k) for retrieved, relevant in runs]),
    }