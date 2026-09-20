import asyncio
import json

import pandas as pd
import pytest
from typesafe_sdk import AsyncTypeSafeClient

from jev_cyrillic_audit.run import MODEL, ROW_FIELDS, ModelDrift, Runner, done_pairs, parse_cell
from tests.conftest import fake_transport


def _items(n: int, dataset: str = "xnli") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [str(i) for i in range(n)],
            "gold": ["neutral"] * n,
            "state_en": [{"premise": f"p{i}", "hypothesis": f"h{i}"} for i in range(n)],
            "state_ru": [{"premise": f"п{i}", "hypothesis": f"г{i}"} for i in range(n)],
        }
    )


def _run(cell, items, passes, out_path, transport, **kw):
    async def go():
        async with AsyncTypeSafeClient(api_key="test", model=MODEL, transport=transport) as client:
            r = Runner(client, concurrency=4, rpm=6000, budget_usd=kw.get("budget"), planned_calls=len(items) * len(passes))
            await r.run_cell(cell, items, passes, out_path)
            return r
    return asyncio.run(go())


def test_rows_have_exact_schema_and_no_text(tmp_path):
    cell = parse_cell("xnli-ru", "en")
    out = tmp_path / "x.jsonl"
    r = _run(cell, _items(5), [0], out, fake_transport())
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5 and r.calls == 5
    for row in rows:
        assert tuple(row) == ROW_FIELDS
        assert row["model"] == MODEL and row["lang"] == "ru" and row["instr_lang"] == "en"
        assert abs(sum(row["probs"].values()) - 1) < 1e-9 and row["pred"] == max(row["probs"], key=row["probs"].get)
        assert "п" not in json.dumps(row, ensure_ascii=False)  # never the source text


def test_resume_skips_done_pairs_and_completes_pass_1(tmp_path):
    cell = parse_cell("xnli-en", "en")
    out = tmp_path / "x.jsonl"
    t1 = fake_transport()
    _run(cell, _items(6), [0], out, t1)
    assert t1.calls["n"] == 6 and len(done_pairs(out)) == 6
    t2 = fake_transport()
    _run(cell, _items(6), [0, 1], out, t2)          # pass 0 already on disk -> only pass 1 is called
    assert t2.calls["n"] == 6
    pairs = done_pairs(out)
    assert len(pairs) == 12 and {p for _, p in pairs} == {0, 1}


def test_failed_calls_go_to_errors_file_and_are_retried_on_resume(tmp_path):
    cell = parse_cell("massive-en", "en")
    items = pd.DataFrame({"item_id": ["1", "2", "3"], "gold": ["a"] * 3,
                          "state_en": [{"utterance": f"u{i}"} for i in (1, 2, 3)], "state_ru": [{"utterance": f"у{i}"} for i in (1, 2, 3)]})
    out = tmp_path / "m.jsonl"
    r = _run(cell, items, [0], out, fake_transport(fail_ids={"u2"}))
    assert r.calls == 2 and r.errors == 1
    assert len(out.with_suffix(".errors.jsonl").read_text().splitlines()) == 1
    assert done_pairs(out) == {("1", 0), ("3", 0)}
    r2 = _run(cell, items, [0], out, fake_transport())
    assert r2.calls == 1 and done_pairs(out) == {("1", 0), ("2", 0), ("3", 0)}


def test_model_drift_aborts_and_writes_nothing_for_drifted_rows(tmp_path):
    cell = parse_cell("xnli-en", "en")
    out = tmp_path / "x.jsonl"
    with pytest.raises(ModelDrift):
        _run(cell, _items(4), [0], out, fake_transport(model="jev-1.14.0"))
    assert done_pairs(out) == set()


def test_budget_guard(tmp_path):
    from jev_cyrillic_audit.run import BudgetExceeded
    cell = parse_cell("xnli-en", "en")
    with pytest.raises(BudgetExceeded):
        _run(cell, _items(60), [0], tmp_path / "x.jsonl", fake_transport(), budget=0.0000001)


def test_auth_error_aborts_instead_of_filling_the_errors_file(tmp_path):
    import httpx2
    from jev_cyrillic_audit.run import FatalAPIError
    t = httpx2.MockTransport(lambda req: httpx2.Response(401, json={"error": {"message": "bad key"}}))
    cell = parse_cell("xnli-en", "en")
    out = tmp_path / "x.jsonl"
    with pytest.raises(FatalAPIError):
        _run(cell, _items(50), [0], out, t)
    assert not out.with_suffix(".errors.jsonl").exists() or out.with_suffix(".errors.jsonl").read_text() == ""


def test_error_streak_trips_the_breaker(tmp_path):
    import httpx2
    from jev_cyrillic_audit.run import FatalAPIError, MAX_CONSECUTIVE_ERRORS
    t = httpx2.MockTransport(lambda req: httpx2.Response(503, json={"error": {"message": "down"}}))
    cell = parse_cell("xnli-en", "en")
    out = tmp_path / "x.jsonl"

    async def go():
        async with AsyncTypeSafeClient(api_key="test", model=MODEL, transport=t, retry=__import__("typesafe_sdk").RetryPolicy(max_retries=0)) as client:
            r = Runner(client, concurrency=4, rpm=6000, budget_usd=None, planned_calls=100)
            await r.run_cell(cell, _items(100), [0], out)
    with pytest.raises(FatalAPIError):
        asyncio.run(go())
    n_err = len(out.with_suffix(".errors.jsonl").read_text().splitlines())
    assert MAX_CONSECUTIVE_ERRORS <= n_err < 100
