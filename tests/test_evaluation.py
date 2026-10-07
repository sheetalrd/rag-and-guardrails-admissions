"""Tests for the retrieval metrics.

Every expected number below is worked out by hand in the comment next to it.
Nothing here needs an LLM or the database: the metrics only take lists and dictionaries.

One shared example is used for most tests:

    retrieved (best first): A, X, B, Y
    relevant grades:        A = 2, B = 1, C = 2     (X and Y are not relevant, C was never found)

So ranks 1 and 3 hold relevant chunks, and there are 3 relevant chunks in total.
"""
import sys
from math import log2
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent[1]/ "src"))

from admissions_agent.evaluation.metrics import (#noqa: E402
    average_precision,
    evaluate,
    f1_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["A", "X", "B", "Y"]
RELEVANT = {"A": 2, "B": 1, "C": 2}

# --- Precision@K: relevant results in top K / K ---------------------------------------
def test_precision_at_k():
   # Top 4 = A, X, B, Y -> 2 relevant (A, B) out of 4 -> 2 / 4 = 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.5)
    # Top 1 = A -> 1 relevant out of 1 -> 1.0
    assert precision_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(1.0)
    # Top 2 = A, X -> 1 relevant out of 2 -> 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, k=2) == pytest.approx(0.5)

def test_precision_divides_by_k_even_with_fewer_results():
    # Only 1 result came back, it is relevant, but K = 4 -> 1 / 4 = 0.25
    assert precision_at_k(["A"], RELEVANT, k=4) == pytest.approx(0.25)

# --- Recall@K: relevant results in top K / total relevant chunks -----------------------
def test_recall_at_k():
    # Top 4 holds A and B, there are 3 relevant chunks -> 2 / 3 = 0.6667
    assert recall_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(2 / 3)
    # Top 2 holds only A -> 1 / 3 = 0.3333
    assert recall_at_k(RETRIEVED, RELEVANT, k=2) == pytest.approx(1 / 3)

# --- F1@K: 2 x P x R / (P + R) ---------------------------------------------------------
def test_f1_at_k():
    # P = 0.5, R = 2/3 -> 2 x 0.5 x 0.6667 / (0.5 + 0.6667) = 0.6667 / 1.1667 = 0.5714
    assert f1_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.5714, abs=1e-4)
    assert f1_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(4 / 7)

def test_f1_is_one_for_a_perfect_result():
    # All 3 relevant chunks in the top 3 -> P = 1, R = 1 -> F1 = 1
    assert f1_at_k(["A", "C", "B"], RELEVANT, k=3) == pytest.approx(1.0)

# --- MRR: average over questions of 1 / rank of first relevant result ------------------
def test_reciprocal_rank():
    # First relevant result (A) is at rank 1 -> 1 / 1 = 1.0
    assert reciprocal_rank(RETRIEVED, RELEVANT, k=4) == pytest.approx(1.0)
    # First relevant result (B) is at rank 3 -> 1 / 3 = 0.3333
    assert reciprocal_rank(["X", "Y", "B"], RELEVANT, k=3) == pytest.approx(1 / 3)
    # Same list cut at K = 2: B is outside the top 2 -> 0
    assert reciprocal_rank(["X", "Y", "B"], RELEVANT, k=2) == 0.0

def test_mean_reciprocal_rank():
    runs = [
        (["A", "X"], RELEVANT),  # first relevant at rank 1 -> 1
        (["X", "A"], RELEVANT),  # first relevant at rank 2 -> 0.5
        (["X", "Y"], RELEVANT),  # none -> 0
    ]
    # (1 + 0.5 + 0) / 3 = 0.5
    assert mean_reciprocal_rank(runs, k=2) == pytest.approx(0.5)

# --- MAP: average over questions of AP -------------------------------------------------
# AP = sum of Precision@i at every rank i that holds a relevant result / total relevant

def test_average_precision():
    # Relevant at rank 1 (Precision@1 = 1/1) and rank 3 (Precision@3 = 2/3).
    # AP = (1 + 0.6667) / 3 relevant chunks = 0.5556
    assert average_precision(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.5556, abs=1e-4)
    assert average_precision(RETRIEVED, RELEVANT, k=4) == pytest.approx(5 / 9)
    # Cut at K = 2: only rank 1 counts -> 1 / 3 = 0.3333
    assert average_precision(RETRIEVED, RELEVANT, k=2) == pytest.approx(1 / 3)


def test_mean_average_precision():
    runs = [
        (RETRIEVED, RELEVANT),  # AP = 5/9 (see above)
        (["X", "Y", "Z", "W"], RELEVANT),  # nothing relevant -> AP = 0
    ]
    # (5/9 + 0) / 2 = 0.2778
    assert mean_average_precision(runs, k=4) == pytest.approx(5 / 18)


# --- NDCG@K: DCG@K / IDCG@K, with DCG = sum of grade_i / log2(i + 1) -------------------


def test_ndcg_at_k():
    # Grades in retrieved order: 2, 0, 1, 0
    # DCG  = 2/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) = 2 + 0 + 0.5 + 0 = 2.5
    # Perfect order of the grades: 2, 2, 1
    # IDCG = 2/log2(2) + 2/log2(3) + 1/log2(4) = 2 + 1.2619 + 0.5 = 3.7619
    # NDCG = 2.5 / 3.7619 = 0.6646
    assert ndcg_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(0.6646, abs=1e-4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, k=4) == pytest.approx(2.5 / (2 + 2 / log2(3) + 0.5))


def test_ndcg_rewards_putting_the_higher_grade_first():
    # Both lists hold the same three relevant chunks; only the order differs.
    # Perfect order (2, 2, 1) -> DCG = IDCG -> 1.0
    assert ndcg_at_k(["A", "C", "B"], RELEVANT, k=3) == pytest.approx(1.0)
    # Grade 1 first (1, 2, 2): DCG = 1 + 1.2619 + 1 = 3.2619 -> 3.2619 / 3.7619 = 0.8671
    assert ndcg_at_k(["B", "A", "C"], RELEVANT, k=3) == pytest.approx(0.8671, abs=1e-4)


def test_ndcg_ideal_ordering_is_also_cut_at_k():
    # K = 1 and the top result has grade 2: DCG = 2, IDCG@1 = 2 -> 1.0
    assert ndcg_at_k(["A"], RELEVANT, k=1) == pytest.approx(1.0)


# --- Edge cases ------------------------------------------------------------------------

ALL_METRICS = [precision_at_k, recall_at_k, f1_at_k, reciprocal_rank, average_precision, ndcg_at_k]


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_no_results_at_all_gives_zero(metric):
    assert metric([], RELEVANT, k=4) == 0.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_no_relevant_chunk_retrieved_gives_zero(metric):
    assert metric(["X", "Y", "Z"], RELEVANT, k=4) == 0.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_question_without_relevant_chunks_gives_zero(metric):
    assert metric(["A", "B"], {}, k=4) == 0.0


def test_f1_is_zero_when_precision_plus_recall_is_zero():
    # P = 0 and R = 0, so 2 x P x R / (P + R) would divide by zero.
    assert f1_at_k(["X"], RELEVANT, k=1) == 0.0


def test_grade_zero_does_not_count_as_relevant():
    assert precision_at_k(["A"], {"A": 0}, k=1) == 0.0
    assert recall_at_k(["A"], {"A": 0}, k=1) == 0.0


def test_a_repeated_chunk_is_counted_once():
    # A appears three times but is one chunk -> 1 of 3 relevant chunks found
    assert recall_at_k(["A", "A", "A"], RELEVANT, k=4) == pytest.approx(1 / 3)


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError):
        precision_at_k(RETRIEVED, RELEVANT, k=0)


# --- All metrics together --------------------------------------------------------------


def test_evaluate_averages_every_metric_over_the_questions():
    runs = [
        (RETRIEVED, RELEVANT),  # P 0.5, R 2/3, F1 4/7, RR 1, AP 5/9, NDCG 0.6646
        ([], RELEVANT),  # everything 0
    ]
    scores = evaluate(runs, k=4)
    # Each average is the first question's value divided by 2.
    assert scores["precision"] == pytest.approx(0.25)
    assert scores["recall"] == pytest.approx(1 / 3)
    assert scores["f1"] == pytest.approx(2 / 7)
    assert scores["mrr"] == pytest.approx(0.5)
    assert scores["map"] == pytest.approx(5 / 18)
    assert scores["ndcg"] == pytest.approx(0.3323, abs=1e-4)


def test_evaluate_with_no_questions_gives_zero():
    assert evaluate([], k=4) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0}