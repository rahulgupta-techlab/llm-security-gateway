"""OpenAI-compatible security gateway.

Clients call POST /v1/chat/completions exactly as they would call OpenAI or
vLLM. The gateway screens every user message, forwards allowed requests to the
upstream model, screens the answer, and records an audit event. With
`"stream": true` the answer is passed through as server-sent events and the
response guards run on the stream (see src/streaming.py). Requests, blocks,
redactions and latency are exported for Prometheus at GET /metrics.

    UPSTREAM_URL=http://localhost:8000/v1 uvicorn src.gateway:app --port 8080
"""
from __future__ import annotations

import itertools
import os
import time
import uuid
from typing import Any, Callable, Iterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from src.classifier import default_classifier
from src.metrics import GatewayMetrics
from src.policy import Policy, check_prompt, check_response
from src.streaming import StreamRedactor, parse_sse, sse


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None


Upstream = Callable[[dict[str, Any]], dict[str, Any]]
StreamUpstream = Callable[[dict[str, Any]], Iterator[dict[str, Any]]]  # yields chat.completion.chunk dicts


def http_upstream(base_url: str, api_key: str | None = None, timeout: float = 60.0) -> Upstream:
    """Forward to any OpenAI-compatible server (vLLM, OpenAI, Ollama...)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return call


def http_stream_upstream(base_url: str, api_key: str | None = None, timeout: float = 60.0,
                         transport: httpx.BaseTransport | None = None) -> StreamUpstream:
    """Stream from any OpenAI-compatible server and yield the parsed SSE chunks."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers) as r:
                r.raise_for_status()
                yield from parse_sse(r.iter_lines())

    return call


def create_app(upstream: Upstream, policy: Policy | None = None,
               stream_upstream: StreamUpstream | None = None) -> FastAPI:
    policy = policy or Policy()
    if policy.ml_classifier:
        default_classifier()  # load the model at startup, not on the first request
    app = FastAPI(title="LLM Security Gateway")
    app.state.audit = []  # in-memory audit log; swap for a real sink in production
    app.state.metrics = metrics = GatewayMetrics()

    def audit(event: dict[str, Any]) -> None:
        app.state.audit.append(event)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/audit")
    def get_audit(limit: int = 50) -> list[dict[str, Any]]:
        return app.state.audit[-limit:]

    @app.get("/metrics")
    def get_metrics() -> Response:
        return Response(metrics.render(), media_type=metrics.content_type)

    def screen_prompt(req: ChatRequest, rid: str, started: float) -> list[dict[str, str]]:
        messages = []
        with metrics.screening_seconds.time():
            decisions = [check_prompt(m.content, policy) if m.role == "user" else None for m in req.messages]
        for m, d in zip(req.messages, decisions):
            if d is None:
                messages.append(m.model_dump())
                continue
            if d.action == "block":
                audit({"id": rid, "stage": "prompt", "action": "block", "reason": d.reason})
                metrics.blocks.labels(d.category).inc()
                metrics.request("blocked", bool(req.stream), time.perf_counter() - started)
                raise HTTPException(status_code=400, detail={"error": "blocked_by_policy", "reason": d.reason, "request_id": rid})
            messages.append({"role": m.role, "content": d.text})
            if d.action == "redact":
                audit({"id": rid, "stage": "prompt", "action": "redact", "reason": d.reason})
                metrics.redacted("prompt", d.findings)
        return messages

    @app.post("/v1/chat/completions", response_model=None)
    def chat(req: ChatRequest) -> dict[str, Any] | StreamingResponse:
        rid, started = str(uuid.uuid4()), time.perf_counter()
        messages = screen_prompt(req, rid, started)  # blocks raise 400 before anything is streamed
        payload = req.model_dump(exclude_none=True) | {"messages": messages}
        if req.stream:
            return stream_chat(payload, rid, started)
        try:
            result = upstream(payload)
        except httpx.HTTPError as e:
            metrics.request("upstream_error", False, time.perf_counter() - started)
            raise HTTPException(status_code=502, detail={"error": "upstream_error", "message": str(e)}) from e

        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            d = check_response(msg.get("content") or "", policy)
            if d.action == "redact":
                msg["content"] = d.text
                audit({"id": rid, "stage": "response", "action": "redact", "reason": d.reason})
                metrics.redacted("response", d.findings)

        metrics.request("allowed", False, time.perf_counter() - started)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        audit({"id": rid, "stage": "done", "action": "allow", "latency_ms": latency_ms})
        result.setdefault("gateway", {})["request_id"] = rid
        return result

    def stream_chat(payload: dict[str, Any], rid: str, started: float) -> StreamingResponse:
        if stream_upstream is None:
            metrics.request("rejected", True, time.perf_counter() - started)
            raise HTTPException(status_code=400, detail={"error": "streaming_not_supported", "request_id": rid})
        chunks = iter(stream_upstream(payload))
        try:
            first = next(chunks, None)  # connect now, so a dead upstream is still a 502
        except httpx.HTTPError as e:
            metrics.request("upstream_error", True, time.perf_counter() - started)
            raise HTTPException(status_code=502, detail={"error": "upstream_error", "message": str(e)}) from e

        redactors: dict[int, StreamRedactor] = {}

        def redactor(index: int) -> StreamRedactor:
            if index not in redactors:
                redactors[index] = StreamRedactor(policy.stream_holdback_chars, policy.redact_pii_in_response)
            return redactors[index]

        def events() -> Iterator[str]:
            outcome = "client_disconnected"  # kept if the client goes away before the stream ends
            try:
                yield from stream_events()
                outcome = "allowed"
            except httpx.HTTPError as e:
                outcome = "upstream_error"
                audit({"id": rid, "stage": "stream", "action": "error", "reason": str(e)})
                yield sse({"error": {"type": "upstream_error", "message": str(e)}, "gateway": {"request_id": rid}})
            finally:
                for r in redactors.values():
                    for label, n in r.labels.items():
                        metrics.redactions.labels("response", label).inc(n)
                metrics.request(outcome, True, time.perf_counter() - started)

        def stream_events() -> Iterator[str]:
            last: dict[str, Any] | None = None
            for chunk in itertools.chain([first] if first is not None else [], chunks):
                last = chunk
                for choice in chunk.get("choices", []):
                    delta, r = choice.setdefault("delta", {}), redactor(choice.get("index", 0))
                    text = r.feed(delta.get("content") or "")
                    if choice.get("finish_reason"):
                        text += r.flush()
                    if text or "content" in delta:
                        delta["content"] = text
                chunk.setdefault("gateway", {})["request_id"] = rid
                yield sse(chunk)
            # Upstream ended without finish_reason for some choice: release what is held back.
            for index, r in redactors.items():
                tail = r.flush()
                if tail:
                    base = {k: v for k, v in (last or {}).items() if k in ("id", "object", "created", "model")}
                    yield sse(base | {"choices": [{"index": index, "delta": {"content": tail}, "finish_reason": None}],
                                      "gateway": {"request_id": rid}})
            labels = sorted(set().union(*(r.labels for r in redactors.values())))
            if labels:
                audit({"id": rid, "stage": "response", "action": "redact", "reason": "pii redacted in stream: " + ", ".join(labels)})
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            audit({"id": rid, "stage": "done", "action": "allow", "stream": True, "latency_ms": latency_ms})
            yield sse("[DONE]")

        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Request-ID": rid})

    return app


_upstream_url, _upstream_key = os.getenv("UPSTREAM_URL", "http://localhost:8000/v1"), os.getenv("UPSTREAM_API_KEY")
app = create_app(http_upstream(_upstream_url, _upstream_key), stream_upstream=http_stream_upstream(_upstream_url, _upstream_key))
