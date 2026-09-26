"""Study 4: translate XNLI test pairs EN->RU two ways with one translator.

  sep   - premise and hypothesis each in its own request (fresh context). A premise shared by several
          pairs is translated once (keyed by its text hash) and reused: without context it is the same input.
  joint - both texts in one request, JSON with two keys

Everything the translator sees is in `prompts/translate.en-ru.json`. The raw log is append-only and
resumable: one row per request, keyed by (item_id, condition, field) where field is "premise",
"hypothesis" (sep) or "pair" (joint). `assemble` turns the log into one file per condition.

Usage:
  uv run python -m jev_cyrillic_audit.translate run --out runs/study4/translation [--limit 20]
  uv run python -m jev_cyrillic_audit.translate assemble --out runs/study4/translation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import XNLI_MISALIGNED_ROWS, load_xnli_lang

PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "translate.en-ru.json"
RAW = "raw.jsonl"
_CYR = re.compile(r"[Ѐ-ӿ]")
_LETTER = re.compile(r"[^\W\d_]")


def load_prompts() -> tuple[dict, str]:
    raw = PROMPTS.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def items() -> pd.DataFrame:
    """All XNLI test items except the rows misaligned in `ru` (dropped from every Study-4 arm)."""
    en = load_xnli_lang("en")
    return en[~en["item_id"].astype(int).isin(XNLI_MISALIGNED_ROWS)].reset_index(drop=True)


def cyrillic_share(text: str) -> float:
    letters = _LETTER.findall(text)
    return sum(bool(_CYR.match(c)) for c in letters) / len(letters) if letters else 0.0


def valid(texts: list[str]) -> bool:
    # Non-empty and at least one Cyrillic letter. Names, titles and URLs legitimately stay in Latin script
    # (e.g. "Carrer dels Banys Nous назван в честь бань."), so no minimum Cyrillic share: an earlier 0.5
    # threshold rejected 12 such correct translations in the first full run.
    return all(isinstance(t, str) and t.strip() and _CYR.search(t) for t in texts)


def row_ok(r: dict) -> bool:
    """Validity is recomputed from the stored output, so a change to `valid` applies to old log rows."""
    return isinstance(r.get("output"), dict) and valid(list(r["output"].values()))


def _schema(keys: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "translation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


def premise_key(premise: str) -> str:
    return "P" + hashlib.sha256(premise.encode("utf-8")).hexdigest()[:16]


def jobs_for(df: pd.DataFrame) -> list[tuple[str, str, str, dict]]:
    """(key, condition, field, texts) for every request of the study; sep premises are deduplicated."""
    out, seen = [], set()
    for iid, st in zip(df["item_id"], df["state"]):
        pk = premise_key(st["premise"])
        if pk not in seen:
            seen.add(pk)
            out.append((pk, "sep", "premise", {"text_1": st["premise"]}))
        out.append((iid, "sep", "hypothesis", {"text_1": st["hypothesis"]}))
        out.append((iid, "joint", "pair", {"text_1": st["premise"], "text_2": st["hypothesis"]}))
    return out


def done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if row_ok(r):
            keys.add((r["item_id"], r["condition"], r["field"]))
    return keys


class Pacer:
    """Spaces request starts to stay under a requests-per-minute limit (shared by all threads)."""

    def __init__(self, rpm: int):
        self.gap, self.next, self.lock = 60.0 / rpm, time.monotonic(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def run(out: Path, limit: int | None, concurrency: int, max_tokens_total: int, data_sharing: str, rpm: int) -> None:
    from openai import APIError, OpenAI

    p, phash = load_prompts()
    client = OpenAI(max_retries=6)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / RAW
    df = items()
    if limit:
        df = df.iloc[:limit]
    done = done_keys(raw_path)
    todo = [j for j in jobs_for(df) if j[:3] not in done]
    print(f"{len(df)} items, {len(todo)} requests to do")

    lock, used = threading.Lock(), {"tok": 0, "n": 0, "bad": 0, "err": 0, "err_run": 0}
    pacer = Pacer(rpm)

    def one(job):
        iid, cond, field, texts = job
        keys = list(texts)
        prompt = p["separate" if cond == "sep" else "joint"].format(**texts)
        for attempt in (1, 2):
            pacer.wait()
            r = client.chat.completions.create(
                model=p["model"],
                messages=[{"role": "system", "content": p["system"]}, {"role": "user", "content": prompt}],
                temperature=p["temperature"],
                seed=p["seed"],
                response_format=_schema(keys),
            )
            msg = r.choices[0].message
            try:
                o = json.loads(msg.content)
                ok = valid([o[k] for k in keys])
            except (json.JSONDecodeError, KeyError, TypeError):
                o, ok = None, False
            if ok:
                break
        return {
            "item_id": iid, "condition": cond, "field": field, "attempts": attempt, "valid": ok, "output": o,
            "refusal": getattr(msg, "refusal", None),
            "request_id": r.id, "model": r.model, "system_fingerprint": r.system_fingerprint,
            "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "prompt_sha256": phash,
        }

    t0 = time.time()
    with raw_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(concurrency) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except APIError as e:  # left undone; the next run resumes it
                with lock:
                    used["err"] += 1
                    used["err_run"] += 1
                    if used["err_run"] >= 20:
                        raise SystemExit(f"20 consecutive API errors, last: {e}") from e
                continue
            used["err_run"] = 0
            assert row["model"].startswith(p["model"].rsplit("-", 3)[0]), f"model drift: {row['model']}"
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                used["tok"] += row["input_tokens"] + row["output_tokens"]
                used["n"] += 1
                used["bad"] += not row["valid"]
                if used["n"] % 500 == 0:
                    print(f"  {used['n']}/{len(todo)}  tokens {used['tok']:,}  invalid {used['bad']}  {time.time()-t0:.0f}s")
                if used["tok"] > max_tokens_total:
                    raise SystemExit(f"token guard hit at {used['tok']:,}; resume later")
    manifest = {
        "study": 4, "step": "translation", "prompts_file": str(PROMPTS.name), "prompts_sha256": phash,
        "translator": p["model"], "temperature": p["temperature"], "seed": p["seed"],
        "items": len(df), "requests_this_run": used["n"], "invalid_this_run": used["bad"], "api_errors_this_run": used["err"],
        "tokens_this_run": used["tok"], "openai_data_sharing": data_sharing,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / f"manifest-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def assemble(out: Path) -> None:
    """raw.jsonl -> ru_sep.jsonl, ru_joint.jsonl (item_id, gold, premise, hypothesis); lists exclusions."""
    df = items().set_index("item_id")
    rows = [json.loads(l) for l in (out / RAW).open(encoding="utf-8")]
    last = {}
    for r in rows:  # a later valid row supersedes an earlier invalid one
        r["valid"] = row_ok(r)
        k = (r["item_id"], r["condition"], r["field"])
        if r["valid"] or k not in last:
            last[k] = r
    sep, joint, excluded, missing = {}, {}, [], 0
    for iid in df.index:
        pk = premise_key(df.loc[iid, "state"]["premise"])
        a, b, c = (last.get((pk, "sep", "premise")), last.get((iid, "sep", "hypothesis")), last.get((iid, "joint", "pair")))
        if not (b or c):
            missing += 1  # not requested (yet), e.g. a --limit run
            continue
        if not (a and b and c and a["valid"] and b["valid"] and c["valid"]):
            excluded.append(iid)
            continue
        sep[iid] = (a["output"]["text_1"], b["output"]["text_1"])
        joint[iid] = (c["output"]["text_1"], c["output"]["text_2"])
    for name, d in (("ru_sep", sep), ("ru_joint", joint)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for iid, (prem, hyp) in d.items():
                f.write(json.dumps({"item_id": iid, "gold": df.loc[iid, "gold"], "premise": prem, "hypothesis": hyp},
                                   ensure_ascii=False) + "\n")
    (out / "excluded.json").write_text(json.dumps(sorted(excluded, key=int)))
    same = sum(sep[i] == joint[i] for i in sep)
    print(f"assembled {len(sep)} items; excluded (invalid/incomplete) {len(excluded)}; not requested {missing}; identical sep/joint pairs {same} ({same/max(len(sep),1):.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assemble"])
    ap.add_argument("--out", type=Path, default=Path("runs/study4/translation"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rpm", type=int, default=400, help="requests per minute (org limit for gpt-4.1-mini was 500)")
    ap.add_argument("--max-tokens-total", type=int, default=8_000_000)
    ap.add_argument("--data-sharing", default="on", help="OpenAI org data-sharing setting, recorded in the manifest")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.out, a.limit, a.concurrency, a.max_tokens_total, a.data_sharing, a.rpm)
    else:
        assemble(a.out)


if __name__ == "__main__":
    main()
