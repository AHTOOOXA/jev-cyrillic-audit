"""Async runner: one item per call, two passes, resumable JSONL, model pinned and asserted.

Usage
    python -m jev_cyrillic_audit.run --smoke
    python -m jev_cyrillic_audit.run --cells xnli-ru --limit 5 --passes 1 --scratch      # dry run
    python -m jev_cyrillic_audit.run --run-id 20260920T120000Z                          # full MVP
Re-running with the same --run-id resumes: (item_id, pass) pairs already on disk are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import typesafe_sdk
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from .data import DATASETS, N_PER_DATASET, REVISIONS, SEED, join_items
from .questions import build_question, check_state, from_choice

MODEL = "jev-1.13.0"
PRICE_PER_MTOK_USD = 0.042
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
SCRATCH = ROOT / "scratch"
CELLS = ("xnli-en", "xnli-ru", "massive-en", "massive-ru")
ROW_FIELDS = (
    "item_id", "dataset", "lang", "instr_lang", "pass", "gold", "pred", "choice", "p_max", "confidence",
    "probs", "input_tokens", "latency_ms", "model", "request_id", "ts",
)


class ModelDrift(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_frozen() -> dict:
    """PREREG.md and prompts/*.json must match their committed SHA-256 files and be clean in git."""
    out = subprocess.run(["shasum", "-a", "256", "-c", "PREREG.sha256"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, f"PREREG.sha256 does not verify:\n{out.stdout}{out.stderr}"
    out = subprocess.run(["shasum", "-a", "256", "-c", "prompts.sha256"], cwd=ROOT / "prompts", capture_output=True, text=True)
    assert out.returncode == 0, f"prompts.sha256 does not verify:\n{out.stdout}{out.stderr}"
    paths = ["PREREG.md", "PREREG.sha256", "prompts", "data/items.parquet"]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, capture_output=True, text=True).stdout
    assert not dirty.strip(), f"frozen files are modified or untracked in git:\n{dirty}"
    tracked = subprocess.run(["git", "ls-files", "--", *paths], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert {"PREREG.md", "PREREG.sha256", "prompts/prompts.sha256"} <= set(tracked), "freeze files are not committed"
    return {
        "prereg_sha256": (ROOT / "PREREG.sha256").read_text().split()[0],
        "prompts_sha256": {l.split()[1]: l.split()[0] for l in (ROOT / "prompts/prompts.sha256").read_text().splitlines() if l.strip()},
    }


class Pacer:
    """Client-side request pacer: at most `rpm` request starts per minute, evenly spaced."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = asyncio.get_running_loop().time()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)


def done_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["item_id"], r["pass"]) for r in map(json.loads, f) if r.get("model")}


class Runner:
    def __init__(self, client: AsyncTypeSafeClient, *, concurrency: int, rpm: float, budget_usd: float | None,
                 planned_calls: int) -> None:
        self.client = client
        self.sem = asyncio.Semaphore(concurrency)
        self.pacer = Pacer(rpm)
        self.budget_usd = budget_usd
        self.planned_calls = planned_calls
        self.calls = 0
        self.tokens = 0
        self.latencies: list[float] = []
        self.errors = 0
        self.abort: BaseException | None = None

    def projected_cost(self) -> float:
        if not self.calls:
            return 0.0
        return self.tokens / self.calls * self.planned_calls * PRICE_PER_MTOK_USD / 1e6

    async def one(self, cell: dict, item: pd.Series, p: int, out, err) -> None:
        if self.abort is not None:
            return
        state = check_state(cell["dataset"], cell["instr_lang"], dict(item[f"state_{cell['lang']}"]))
        question = build_question(cell["dataset"], cell["instr_lang"])
        async with self.sem:
            if self.abort is not None:
                return
            await self.pacer.wait()
            t0 = time.perf_counter()
            try:
                r = await self.client.system_one(state, {"q": question}, model=MODEL)
            except TypeSafeError as e:
                self.errors += 1
                err.write(json.dumps({"item_id": item["item_id"], "pass": p, "error": type(e).__name__,
                                      "message": str(e)[:500], "ts": utc_now()}, ensure_ascii=False) + "\n")
                err.flush()
                return
            latency_ms = (time.perf_counter() - t0) * 1000
        if r.model != MODEL:
            self.abort = ModelDrift(f"response.model={r.model!r} != {MODEL!r} (request_id={r.request_id})")
            return
        ans = from_choice(r.choices["q"])
        row = {
            "item_id": item["item_id"], "dataset": cell["dataset"], "lang": cell["lang"],
            "instr_lang": cell["instr_lang"], "pass": p, "gold": item["gold"],
            **{k: ans[k] for k in ("pred", "choice", "p_max", "confidence", "probs")},
            "input_tokens": r.usage.input_tokens, "latency_ms": round(latency_ms, 1),
            "model": r.model, "request_id": r.request_id, "ts": utc_now(),
        }
        assert tuple(row) == ROW_FIELDS
        out.write(json.dumps(row, ensure_ascii=False) + "\n")   # no await between write and flush
        out.flush()
        self.calls += 1
        self.tokens += r.usage.input_tokens or 0
        self.latencies.append(latency_ms)
        if self.budget_usd is not None and self.calls % 50 == 0 and self.projected_cost() > self.budget_usd:
            self.abort = BudgetExceeded(f"projected ${self.projected_cost():.3f} > budget ${self.budget_usd:.2f}")

    async def run_cell(self, cell: dict, items: pd.DataFrame, passes: list[int], out_path: Path) -> None:
        done = done_pairs(out_path)
        todo = [(p, it) for p in passes for _, it in items.iterrows() if (it["item_id"], p) not in done]
        print(f"[{cell['name']}] {len(done)} rows on disk, {len(todo)} calls to make -> {out_path.name}", flush=True)
        if not todo:
            return
        with out_path.open("a", encoding="utf-8") as out, out_path.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as err:
            t0 = time.perf_counter()
            await asyncio.gather(*(self.one(cell, it, p, out, err) for p, it in todo))
            dt = time.perf_counter() - t0
        if self.abort is not None:
            raise self.abort
        print(f"[{cell['name']}] done in {dt:.0f}s; cumulative calls={self.calls} tokens={self.tokens} errors={self.errors}", flush=True)


def parse_cell(name: str, instr_lang: str) -> dict:
    dataset, lang = name.split("-")
    assert dataset in DATASETS and lang in ("en", "ru"), name
    return {"name": name, "dataset": dataset, "lang": lang, "instr_lang": instr_lang}


def write_manifest(run_id: str, cells: list[dict], passes: int, n: int, frozen: dict, out_dir: Path) -> dict:
    files = sorted(out_dir.glob(f"{run_id}-*.jsonl"))
    rows = [json.loads(l) for f in files if not f.name.endswith(".errors.jsonl") for l in f.open(encoding="utf-8")]
    errs = sum(1 for f in files if f.name.endswith(".errors.jsonl") for _ in f.open(encoding="utf-8"))
    per_cell = {}
    for c in cells:
        rs = [r for r in rows if r["dataset"] == c["dataset"] and r["lang"] == c["lang"]]
        lat = sorted(r["latency_ms"] for r in rs)
        per_cell[c["name"]] = {
            "file": f"{run_id}-{c['name']}.jsonl", "rows": len(rs),
            "expected_rows": n * passes,
            "input_tokens": sum(r["input_tokens"] or 0 for r in rs),
            "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rs), 1) if rs else None,
            "latency_p50_ms": lat[len(lat) // 2] if lat else None,
            "latency_p95_ms": lat[int(len(lat) * 0.95)] if lat else None,
            "models": sorted({r["model"] for r in rs}),
        }
    total_tokens = sum(r["input_tokens"] or 0 for r in rows)
    m = {
        "run_id": run_id, "model": MODEL, "sdk_version": typesafe_sdk.__version__, "python": sys.version.split()[0],
        "seed": SEED, "n_per_cell": n, "passes": passes, "instr_lang": cells[0]["instr_lang"],
        "concurrency_rpm_cap": None,
        "run_started_utc": min(r["ts"] for r in rows) if rows else None,
        "run_finished_utc": max(r["ts"] for r in rows) if rows else None,
        "total_calls": len(rows), "total_errors_logged": errs, "total_input_tokens": total_tokens,
        "total_cost_usd": round(total_tokens * PRICE_PER_MTOK_USD / 1e6, 4),
        "dataset_revisions": REVISIONS, "items_parquet_sha256": sha256_file(ROOT / "data/items.parquet"),
        **frozen, "cells": per_cell,
    }
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


async def smoke() -> None:
    async with AsyncTypeSafeClient(model=MODEL) as client:
        r = await client.system_one(
            {"utterance": "поставь будильник на семь утра"},
            {"q": build_question("massive", "en")}, model=MODEL,
        )
    a = from_choice(r.choices["q"])
    print(f"model={r.model} request_id={r.request_id} input_tokens={r.usage.input_tokens}")
    print(f"pred={a['pred']} p_max={a['p_max']:.3f} confidence={a['confidence']:.3f} choice={a['choice']}")
    assert r.model == MODEL, r.model
    print("OK: response.model == jev-1.13.0")


async def main_async(a: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    if a.smoke:
        await smoke()
        return
    cells = [parse_cell(c, a.instr_lang) for c in a.cells]
    out_dir = SCRATCH if a.scratch else RUNS
    out_dir.mkdir(exist_ok=True)
    frozen = {"prereg_sha256": None, "prompts_sha256": None} if a.scratch else check_frozen()
    items = pd.read_parquet(ROOT / "data/items.parquet")
    n = a.limit or N_PER_DATASET
    per_dataset = {ds: join_items(items, ds).head(n) for ds in sorted({c["dataset"] for c in cells})}
    planned = len(cells) * n * a.passes
    print(f"run_id={a.run_id} cells={[c['name'] for c in cells]} n={n} passes={a.passes} planned_calls={planned} "
          f"out={out_dir.relative_to(ROOT)}/ budget=${a.budget_usd}", flush=True)
    retry = RetryPolicy(max_retries=6, backoff_initial=1.0, backoff_max=20.0, timeout=180.0)
    async with AsyncTypeSafeClient(model=MODEL, retry=retry, timeout=60.0) as client:
        runner = Runner(client, concurrency=a.concurrency, rpm=a.rpm, budget_usd=a.budget_usd, planned_calls=planned)
        for p in range(a.passes):                      # all cells at pass 0, then all cells at pass 1
            for c in cells:
                await runner.run_cell(c, per_dataset[c["dataset"]], [p], out_dir / f"{a.run_id}-{c['name']}.jsonl")
    m = write_manifest(a.run_id, cells, a.passes, n, frozen, out_dir)
    m["concurrency_rpm_cap"] = [a.concurrency, a.rpm]
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    lat = sorted(runner.latencies)
    print(f"\nthis invocation: calls={runner.calls} errors={runner.errors} tokens={runner.tokens} "
          f"p50={lat[len(lat)//2]:.0f}ms" if lat else "\nthis invocation: nothing to do")
    print(f"manifest: rows={m['total_calls']} tokens={m['total_input_tokens']} cost=${m['total_cost_usd']:.4f} "
          f"models={sorted({mm for c in m['cells'].values() for mm in c['models']})}")
    if runner.calls:
        full = 2 * N_PER_DATASET * len(CELLS)
        print(f"projection at {full} calls (MVP): ${runner.tokens / runner.calls * full * PRICE_PER_MTOK_USD / 1e6:.3f}")
    short = {k: v for k, v in m["cells"].items() if v["rows"] != v["expected_rows"]}
    if short:
        print(f"INCOMPLETE cells (re-run with the same --run-id to resume): {short}")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="one call, print and assert the model id")
    ap.add_argument("--cells", nargs="+", default=list(CELLS), choices=CELLS)
    ap.add_argument("--instr-lang", default="en", choices=("en", "ru"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N items per dataset (dry runs)")
    ap.add_argument("--scratch", action="store_true", help="write to scratch/ (gitignored) and skip the freeze check")
    ap.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=1150.0)
    ap.add_argument("--budget-usd", type=float, default=1.0, help="abort if the projected run cost exceeds this")
    a = ap.parse_args()
    if a.limit and not a.scratch:
        ap.error("--limit is for dry runs; add --scratch")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
