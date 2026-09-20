"""A fake TypeSafe server for offline runner tests: deterministic Choice answers, optional drift."""

from __future__ import annotations

import hashlib
import json

import httpx2
import pytest


def fake_transport(model: str = "jev-1.13.0", fail_ids: set[str] | None = None) -> httpx2.MockTransport:
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        q = body["questions"]["q"]
        keys = list(q["criteria"])
        state_json = json.dumps(body["state"], sort_keys=True, ensure_ascii=False)
        if fail_ids and any(f in state_json for f in fail_ids):
            return httpx2.Response(500, json={"error": {"message": "boom"}})
        h = int(hashlib.sha256(state_json.encode()).hexdigest(), 16)
        top = keys[h % len(keys)]
        p_top = 0.5 + (h % 50) / 100  # 0.50..0.99
        rest = (1 - p_top) / max(len(keys) - 1, 1)
        probs = {k: (p_top if k == top else rest) for k in keys}
        return httpx2.Response(
            200,
            headers={"x-typesafe-request-id": f"req-{calls['n']:05d}"},
            json={
                "model": model,
                "usage": {"input_tokens": 100 + len(state_json), "output_tokens": 0},
                "answers": {"q": {"type": "choice", "choice": top, "confidence": round(p_top - 0.05, 3), "probabilities": probs}},
            },
        )

    t = httpx2.MockTransport(handler)
    t.calls = calls  # type: ignore[attr-defined]
    return t


@pytest.fixture
def transport():
    return fake_transport()
