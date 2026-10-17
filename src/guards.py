"""Input/output guardrails for LLM traffic.

Deterministic, dependency-free detectors so they are fast enough to run on
every request and easy to test. Each detector returns a list of Finding
objects; the policy layer decides what to do with them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    kind: str        # "prompt_injection" | "pii" | "secret"
    label: str       # e.g. "ignore_instructions", "email", "iban"
    span: tuple[int, int]
    score: float     # 0..1 confidence


# --- prompt injection -------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, str, float]] = [
    ("ignore_instructions", r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|system)\b.{0,20}\b(instructions?|rules?|prompts?)\b", 0.9),
    ("reveal_system_prompt", r"\b(reveal|show|print|repeat|output)\b.{0,30}\b(system|hidden|initial)\b.{0,10}\b(prompt|instructions?|message)\b", 0.85),
    ("role_override", r"\b(you are now|act as|pretend to be|from now on you)\b.{0,40}\b(dan|jailbreak|unrestricted|without (any )?restrictions|developer mode)\b", 0.85),
    ("disable_safety", r"\b(disable|bypass|turn off|ignore)\b.{0,30}\b(safety|guardrails?|filters?|content policy|moderation)\b", 0.8),
    ("exfiltrate", r"\b(send|post|upload|exfiltrate)\b.{0,40}\b(https?://|api key|password|credentials|token)", 0.75),
    ("delimiter_attack", r"(</?(system|assistant)>|\[/?INST\]|###\s*(system|instruction))", 0.6),
]
_INJECTION_RE = [(label, re.compile(p, re.I | re.S), score) for label, p, score in _INJECTION_PATTERNS]


def detect_injection(text: str) -> list[Finding]:
    return [Finding("prompt_injection", label, m.span(), score)
            for label, rx, score in _INJECTION_RE
            for m in rx.finditer(text)]


# --- PII and secrets --------------------------------------------------------

def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
        alt = not alt
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


# Most specific patterns first: a span claimed by an earlier pattern is not
# re-labelled by a later, looser one (e.g. IBAN digits are not a phone number).
_PII_PATTERNS: list[tuple[str, str, str]] = [
    ("secret", "openai_key", r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ("secret", "aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("secret", "private_key", r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    ("pii", "email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("pii", "iban", r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b"),
    ("pii", "credit_card", r"\b(?:\d[ -]?){12,18}\d\b"),
    ("pii", "phone", r"(?<!\w)\+?\d{1,3}[\s/-]?\(?\d{2,4}\)?[\s/-]?\d{3,4}[\s/-]?\d{3,5}(?!\w)"),
]
_PII_RE = [(kind, label, re.compile(p)) for kind, label, p in _PII_PATTERNS]


def detect_pii(text: str) -> list[Finding]:
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []
    for kind, label, rx in _PII_RE:
        for m in rx.finditer(text):
            span = m.span()
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue  # already covered by a more specific pattern
            value = m.group(0)
            if label == "credit_card" and not _luhn_ok(re.sub(r"\D", "", value)):
                continue
            if label == "iban" and not _iban_ok(value):
                continue
            found.append(Finding(kind, label, span, 1.0))
            taken.append(span)
    return found


def redact(text: str, findings: list[Finding]) -> str:
    """Replace each finding span with a typed placeholder, e.g. [EMAIL]."""
    out = text
    for f in sorted(findings, key=lambda f: f.span[0], reverse=True):
        s, e = f.span
        out = out[:s] + f"[{f.label.upper()}]" + out[e:]
    return out
