"""Second-stage prompt-injection classifier: TF-IDF + logistic regression.

The model is trained offline by src/train_classifier.py (scikit-learn) and
exported to plain JSON, so scoring needs no ML dependencies at runtime and the
gateway image stays small. Scoring re-implements scikit-learn's
TfidfVectorizer (lowercase, word 1-2 grams, sublinear tf, l2 norm) exactly;
a test checks the scores match scikit-learn.

Text is PII-redacted before scoring, in training and at runtime, so an email
address or phone number is scored as [EMAIL] / [PHONE] and cannot sway it.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from src.guards import Finding, detect_pii, redact

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "injection_clf.json"

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token_pattern


def prepare(text: str) -> str:
    return redact(text, detect_pii(text))


class InjectionClassifier:
    def __init__(self, model: dict):
        self.vocabulary: dict[str, int] = model["vocabulary"]
        self.idf: list[float] = model["idf"]
        self.coef: list[float] = model["coef"]
        self.intercept: float = model["intercept"]
        self.threshold: float = model["threshold"]
        self.ngram_range: tuple[int, int] = tuple(model["ngram_range"])
        self.sublinear_tf: bool = model["sublinear_tf"]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "InjectionClassifier":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def _ngrams(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        lo, hi = self.ngram_range
        return [" ".join(tokens[i:i + n]) for n in range(lo, hi + 1) for i in range(len(tokens) - n + 1)]

    def score(self, text: str) -> float:
        """Probability (0..1) that the text is an injection or jailbreak attempt."""
        counts: dict[int, int] = {}
        for gram in self._ngrams(prepare(text)):
            idx = self.vocabulary.get(gram)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        weights = {i: (1 + math.log(c) if self.sublinear_tf else c) * self.idf[i] for i, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        z = self.intercept + sum(w / norm * self.coef[i] for i, w in weights.items())
        return 1 / (1 + math.exp(-z))


@lru_cache(maxsize=1)
def default_classifier() -> InjectionClassifier:
    return InjectionClassifier.load()


def detect_injection_ml(text: str, threshold: float | None = None,
                        classifier: InjectionClassifier | None = None) -> list[Finding]:
    """A single finding covering the whole text if the classifier score reaches the threshold."""
    clf = classifier or default_classifier()
    p = clf.score(text)
    if p >= (clf.threshold if threshold is None else threshold):
        return [Finding("prompt_injection", "ml_classifier", (0, len(text)), round(p, 3))]
    return []
