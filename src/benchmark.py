"""Measure the injection detector on a labelled prompt set.

Compares the rules alone with rules + the ML classifier on data/prompts.jsonl
(hand-written, never used for training), then prints the classifier's
held-out metrics from training.

    python -m src.benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.classifier import default_classifier
from src.policy import Policy, check_prompt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "prompts.jsonl"
CLF_METRICS = ROOT / "models" / "injection_clf_metrics.json"


def evaluate(rows: list[dict], policy: Policy) -> dict:
    tp = fp = tn = fn = 0
    misses, false_alarms = [], []
    started = time.perf_counter()
    for row in rows:
        blocked = check_prompt(row["text"], policy).action == "block"
        if row["attack"] and blocked:
            tp += 1
        elif row["attack"]:
            fn += 1; misses.append(row["text"])
        elif blocked:
            fp += 1; false_alarms.append(row["text"])
        else:
            tn += 1
    per_prompt_ms = (time.perf_counter() - started) * 1000 / max(len(rows), 1)
    return {
        "attacks": tp + fn, "benign": tn + fp,
        "detection_rate": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "ms_per_prompt": round(per_prompt_ms, 3),
        "missed": misses, "false_alarms": false_alarms,
    }


def main() -> None:
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    default_classifier()  # load the model before timing, as the gateway does at startup
    for name, policy in [("rules", Policy(ml_classifier=False)), ("rules+ml", Policy())]:
        r = evaluate(rows, policy)
        print(f"[{name}] attacks={r['attacks']} benign={r['benign']}")
        print(f"detection_rate={r['detection_rate']:.2f} false_positive_rate={r['false_positive_rate']:.2f} "
              f"precision={r['precision']:.2f} ms/prompt={r['ms_per_prompt']}")
        for t in r["missed"]:
            print("  missed:", t)
        for t in r["false_alarms"]:
            print("  false alarm:", t)

    m = json.loads(CLF_METRICS.read_text(encoding="utf-8"))
    t = m["test"]
    print(f"[classifier, held-out test split] n={t['n']} threshold={m['threshold']} precision={t['precision']} "
          f"recall={t['recall']} f1={t['f1']} false_positive_rate={t['false_positive_rate']}")


if __name__ == "__main__":
    main()
