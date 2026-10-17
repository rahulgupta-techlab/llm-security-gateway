"""Turn detector findings into a decision: allow, redact, or block."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.guards import Finding, detect_injection, detect_pii, redact


@dataclass
class Policy:
    block_injection_score: float = 0.75   # block if any injection finding scores >= this
    redact_pii_in_prompt: bool = True     # mask PII before it reaches the model
    block_secrets_in_prompt: bool = True  # never forward API keys / private keys
    redact_pii_in_response: bool = True   # mask PII the model leaks back


@dataclass
class Decision:
    action: str                     # "allow" | "redact" | "block"
    text: str                       # text to forward (possibly redacted)
    findings: list[Finding] = field(default_factory=list)
    reason: str = ""


def check_prompt(text: str, policy: Policy) -> Decision:
    injections = detect_injection(text)
    top = max((f.score for f in injections), default=0.0)
    if top >= policy.block_injection_score:
        labels = sorted({f.label for f in injections})
        return Decision("block", text, injections, f"prompt injection: {', '.join(labels)}")

    pii = detect_pii(text)
    secrets = [f for f in pii if f.kind == "secret"]
    if secrets and policy.block_secrets_in_prompt:
        return Decision("block", text, injections + pii, "secret in prompt: " + ", ".join(sorted({f.label for f in secrets})))
    if pii and policy.redact_pii_in_prompt:
        return Decision("redact", redact(text, pii), injections + pii, "pii redacted")
    return Decision("allow", text, injections + pii)


def check_response(text: str, policy: Policy) -> Decision:
    pii = detect_pii(text)
    if pii and policy.redact_pii_in_response:
        return Decision("redact", redact(text, pii), pii, "pii redacted in response")
    return Decision("allow", text, pii)
