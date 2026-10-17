"""OpenAI-compatible security gateway.

Clients call POST /v1/chat/completions exactly as they would call OpenAI or
vLLM. The gateway screens every user message, forwards allowed requests to the
upstream model, screens the answer, and records an audit event.

    UPSTREAM_URL=http://localhost:8000/v1 uvicorn src.gateway:app --port 8080
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.policy import Policy, check_prompt, check_response


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


Upstream = Callable[[dict[str, Any]], dict[str, Any]]


def http_upstream(base_url: str, api_key: str | None = None, timeout: float = 60.0) -> Upstream:
    """Forward to any OpenAI-compatible server (vLLM, OpenAI, Ollama...)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return call


def create_app(upstream: Upstream, policy: Policy | None = None) -> FastAPI:
    policy = policy or Policy()
    app = FastAPI(title="LLM Security Gateway")
    app.state.audit = []  # in-memory audit log; swap for a real sink in production

    def audit(event: dict[str, Any]) -> None:
        app.state.audit.append(event)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/audit")
    def get_audit(limit: int = 50) -> list[dict[str, Any]]:
        return app.state.audit[-limit:]

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        rid, started = str(uuid.uuid4()), time.perf_counter()
        messages = []
        for m in req.messages:
            if m.role != "user":
                messages.append(m.model_dump())
                continue
            d = check_prompt(m.content, policy)
            if d.action == "block":
                audit({"id": rid, "stage": "prompt", "action": "block", "reason": d.reason})
                raise HTTPException(status_code=400, detail={"error": "blocked_by_policy", "reason": d.reason, "request_id": rid})
            messages.append({"role": m.role, "content": d.text})
            if d.action == "redact":
                audit({"id": rid, "stage": "prompt", "action": "redact", "reason": d.reason})

        payload = req.model_dump(exclude_none=True) | {"messages": messages}
        try:
            result = upstream(payload)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail={"error": "upstream_error", "message": str(e)}) from e

        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            d = check_response(msg.get("content") or "", policy)
            if d.action == "redact":
                msg["content"] = d.text
                audit({"id": rid, "stage": "response", "action": "redact", "reason": d.reason})

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        audit({"id": rid, "stage": "done", "action": "allow", "latency_ms": latency_ms})
        result.setdefault("gateway", {})["request_id"] = rid
        return result

    return app


app = create_app(http_upstream(os.getenv("UPSTREAM_URL", "http://localhost:8000/v1"), os.getenv("UPSTREAM_API_KEY")))
