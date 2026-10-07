import csv
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline, make_union

from admissions_agent.config import settings
from admissions_agent.guardrails.result import GuardrailResult

GUARD_NAME = "nlu"
ALLOWED_LABEL = "admissions"


def load_training_data(path: Path) -> tuple[list[str], list[str]]:
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row["text"] for row in rows], [row["label"] for row in rows]


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    # Word features capture meaning; character features keep typos like "admissions" recognisable.
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    classifier = make_pipeline(features, LogisticRegression(C=10, max_iter=1000))
    classifier.fit(texts, labels)
    return classifier


@lru_cache
def get_classifier() -> Pipeline:
    return train_classifier(*load_training_data(settings.training_csv))


def classify(text: str) -> tuple[str, float]:
    classifier = get_classifier()
    probabilities = classifier.predict_proba([text])[0]
    best = probabilities.argmax()
    return str(classifier.classes_[best]), float(probabilities[best])


def check_input_nlu(text: str) -> GuardrailResult:
    label, confidence = classify(text)
    reason = f"classified as '{label}' (confidence {confidence:.2f})"
    return GuardrailResult(GUARD_NAME, label == ALLOWED_LABEL, reason)
