"""Measure the injection detector on a labelled prompt set.

    python -m src.benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.policy import Policy, check_prompt

DATA = Path(__file__).resolve().parent.parent / "data" / "prompts.jsonl"


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
    r = evaluate(rows, Policy())
    print(f"attacks={r['attacks']} benign={r['benign']}")
    print(f"detection_rate={r['detection_rate']:.2f} false_positive_rate={r['false_positive_rate']:.2f} "
          f"precision={r['precision']:.2f} ms/prompt={r['ms_per_prompt']}")
    for t in r["missed"]:
        print("  missed:", t)
    for t in r["false_alarms"]:
        print("  false alarm:", t)


if __name__ == "__main__":
    main()
