"""Response guards for streamed (SSE) answers.

A streamed answer arrives in small pieces, so PII can be split across chunks
("max@co" + "rp.de"). StreamRedactor holds back the last `holdback` characters
and never cuts inside something that could still grow into a match; text is
only released once it can no longer change, and it is redacted on the way out.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterator

from src.guards import detect_pii, redact


class StreamRedactor:
    def __init__(self, holdback: int = 64, enabled: bool = True):
        self.holdback, self.enabled = holdback, enabled
        self.pending = ""   # received but not yet released
        self.context = ""   # last released raw character, for \b and lookbehinds
        self.labels: Counter[str] = Counter()  # redacted values per label

    def feed(self, text: str) -> str:
        """Add streamed text; return the part that is now safe to send."""
        if not self.enabled:
            return text
        self.pending += text
        return self._release(final=False)

    def flush(self) -> str:
        """End of stream: release (and redact) everything still held back."""
        if not self.enabled:
            return ""
        return self._release(final=True)

    def _release(self, final: bool) -> str:
        buf, n = self.context + self.pending, len(self.context)
        # A match starting in the context char began in text that was already sent.
        findings = [f for f in detect_pii(buf) if f.span[0] >= n]
        cut = len(buf) if final else max(n, len(buf) - self.holdback)
        for f in findings:  # spans never overlap, so one pass is enough
            if f.span[0] < cut < f.span[1]:
                cut = f.span[0]  # never split a (possibly still growing) match
        done = [f for f in findings if f.span[1] <= cut]
        self.labels.update(f.label for f in done)
        shifted = [type(f)(f.kind, f.label, (f.span[0] - n, f.span[1] - n), f.score) for f in done]
        released = redact(buf[n:cut], shifted)
        if cut > n:
            self.context = buf[cut - 1]
        self.pending = buf[cut:]
        return released


def parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yield the JSON payload of each `data:` line until `data: [DONE]`."""
    for line in lines:
        if not line.startswith("data:"):
            continue  # blank separators, comments, event/id fields
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)


def sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"
