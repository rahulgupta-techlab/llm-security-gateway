"""Train the second-stage injection classifier on public datasets.

    pip install -r requirements-train.txt
    python -m src.train_classifier

Downloads three public datasets (pinned to a commit), trains TF-IDF + logistic
regression, picks the decision threshold on a validation split, measures
precision/recall on a held-out test split and writes:

    models/injection_clf.json          # vocabulary, idf, weights, threshold
    models/injection_clf_metrics.json  # held-out metrics, per source

Datasets (attack = 1, benign = 0):
- verazuo/jailbreak_llms (MIT): in-the-wild jailbreak prompts (1) and regular prompts (0)
- meta-llama/PurpleLlama CyberSecEval prompt_injection.json (MIT): injected user inputs (1)
- paul-rottger/xstest (CC-BY-4.0): prompts that sound risky but contain no injection (0)
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models"

SOURCES = {
    "jailbreak_llms_jailbreak.csv": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/"
        "4f4031bf8be187f4478c7f94f42b08714722c12e/data/prompts/jailbreak_prompts_2023_12_25.csv",
    "jailbreak_llms_regular.csv": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/"
        "4f4031bf8be187f4478c7f94f42b08714722c12e/data/prompts/regular_prompts_2023_12_25.csv",
    "purplellama_prompt_injection.json": "https://raw.githubusercontent.com/meta-llama/PurpleLlama/"
        "f64eebd53003a87d9c2d2a3a712133534c214c8d/CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json",
    "xstest_prompts.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/"
        "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv",
}

SEED = 0
MAX_BENIGN_FPR = 0.01  # threshold = lowest score that keeps validation FPR at or below this
VECTORIZER = dict(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)


def fetch() -> dict[str, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in SOURCES.items():
        path = CACHE / name
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def _csv_prompts(path: Path) -> set[str]:
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8", newline="") as f:
        return {r["prompt"].strip() for r in csv.DictReader(f) if r["prompt"].strip()}


def load_examples(paths: dict[str, Path]) -> list[tuple[str, int, str]]:
    """(text, label, source), deduplicated; a text labelled as an attack anywhere is an attack."""
    jailbreak = _csv_prompts(paths["jailbreak_llms_jailbreak.csv"])
    injection = {x["user_input"].strip()
                 for x in json.loads(paths["purplellama_prompt_injection.json"].read_text(encoding="utf-8"))}
    attacks = jailbreak | injection
    regular = _csv_prompts(paths["jailbreak_llms_regular.csv"]) - attacks
    xstest = _csv_prompts(paths["xstest_prompts.csv"]) - attacks - regular
    rows = ([(t, 1, "jailbreak_llms") for t in sorted(jailbreak)]
            + [(t, 1, "purplellama") for t in sorted(injection - jailbreak)]
            + [(t, 0, "jailbreak_llms_regular") for t in sorted(regular)]
            + [(t, 0, "xstest") for t in sorted(xstest)])
    return rows


def pick_threshold(scores, labels, max_fpr: float = MAX_BENIGN_FPR) -> float:
    benign = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    allowed = int(max_fpr * len(benign))  # benign examples allowed at or above the threshold
    return round(float(benign[allowed]) + 1e-6, 6) if allowed < len(benign) else 0.5


def metrics(scores, labels, sources, threshold: float) -> dict:
    pred = [s >= threshold for s in scores]
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum(not p and y for p, y in zip(pred, labels))
    tn = sum(not p and not y for p, y in zip(pred, labels))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    per_source = {}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        flagged = sum(pred[i] for i in idx)
        key = "recall" if labels[idx[0]] else "false_positive_rate"
        per_source[src] = {"n": len(idx), key: round(flagged / len(idx), 3)}
    return {
        "n": len(labels), "attacks": tp + fn, "benign": tn + fp,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "false_positive_rate": round(fp / max(fp + tn, 1), 3),
        "per_source": per_source,
    }


def export(vectorizer, model, threshold: float) -> dict:
    vocab = {term: int(i) for term, i in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])}
    return {
        "type": "tfidf_logreg",
        "ngram_range": list(vectorizer.ngram_range),
        "sublinear_tf": bool(vectorizer.sublinear_tf),
        "threshold": threshold,
        "intercept": float(model.intercept_[0]),
        "vocabulary": vocab,
        "idf": [float(x) for x in vectorizer.idf_],
        "coef": [float(x) for x in model.coef_[0]],
    }


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from src.classifier import InjectionClassifier, prepare

    rows = load_examples(fetch())
    strata = [f"{src}:{y}" for _, y, src in rows]
    train, test = train_test_split(rows, test_size=0.2, stratify=strata, random_state=SEED)
    train, val = train_test_split(train, test_size=0.125, stratify=[f"{s}:{y}" for _, y, s in train], random_state=SEED)

    vectorizer = TfidfVectorizer(**VECTORIZER)
    X = vectorizer.fit_transform([prepare(t) for t, _, _ in train])
    model = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(X, [y for _, y, _ in train])

    def scores(split):
        return model.predict_proba(vectorizer.transform([prepare(t) for t, _, _ in split]))[:, 1].tolist()

    threshold = pick_threshold(scores(val), [y for _, y, _ in val])
    test_scores = scores(test)
    result = {
        "datasets": SOURCES,
        "split": {"train": len(train), "validation": len(val), "test": len(test), "seed": SEED},
        "threshold": threshold,
        "threshold_rule": f"lowest score with validation benign FPR <= {MAX_BENIGN_FPR}",
        "test": metrics(test_scores, [y for _, y, _ in test], [s for _, _, s in test], threshold),
        "test_at_0.5": metrics(test_scores, [y for _, y, _ in test], [s for _, _, s in test], 0.5),
    }

    exported = export(vectorizer, model, threshold)
    clf = InjectionClassifier(exported)
    drift = max(abs(clf.score(t) - s) for (t, _, _), s in zip(test, test_scores))
    assert drift < 1e-9, f"pure-Python scorer differs from scikit-learn by {drift}"

    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "injection_clf.json").write_text(json.dumps(exported, separators=(",", ":")), encoding="utf-8")
    (MODEL_DIR / "injection_clf_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("split", "threshold", "test", "test_at_0.5")}, indent=2))


if __name__ == "__main__":
    main()
