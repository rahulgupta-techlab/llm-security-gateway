"""Prometheus metrics for the gateway, served at GET /metrics.

Each app gets its own registry, so tests (and several gateways in one process)
never share counters. Label values are kept to small fixed sets: block reasons
are categories, not the free-text reason, and redactions are counted per PII
label (email, iban, ...), never per value.
"""
from __future__ import annotations

from typing import Iterable

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from src.guards import Finding

# Upstream calls dominate request time; the screening buckets are for the guards alone.
REQUEST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
SCREENING_BUCKETS = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)


class GatewayMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "gateway_requests", "Chat completion requests by outcome "
            "(allowed, blocked, upstream_error, rejected, client_disconnected).",
            ["outcome", "stream"], registry=r)
        self.blocks = Counter(
            "gateway_blocks", "Prompts blocked by policy, by reason (injection_rules, injection_ml, secret).",
            ["reason"], registry=r)
        self.redactions = Counter(
            "gateway_redactions", "Values redacted, by stage (prompt, response) and label (email, iban, ...).",
            ["stage", "label"], registry=r)
        self.request_seconds = Histogram(
            "gateway_request_duration_seconds", "Time from request to last byte, upstream included.",
            ["outcome", "stream"], buckets=REQUEST_BUCKETS, registry=r)
        self.screening_seconds = Histogram(
            "gateway_prompt_screening_seconds", "Time spent screening the prompt (rules, ML classifier, PII).",
            buckets=SCREENING_BUCKETS, registry=r)

    def request(self, outcome: str, stream: bool, seconds: float) -> None:
        s = "true" if stream else "false"
        self.requests.labels(outcome, s).inc()
        self.request_seconds.labels(outcome, s).observe(seconds)

    def redacted(self, stage: str, findings: Iterable[Finding]) -> None:
        for f in findings:
            if f.kind != "prompt_injection":
                self.redactions.labels(stage, f.label).inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
