"""Parallel RU/EN slices: pinned loads, join assertions, seeded stratified sample.

Nothing in this module writes source text to disk. `data/items.parquet` holds ids, gold labels,
join keys and SHA-256 hashes of the state; the runner re-joins against the Hugging Face cache.

Study 2 ("Panorama") extends the same contract to 15 XNLI languages and Belebele en/ru:
`data/items2.parquet` is LONG — one row per (dataset, item_id, lang) — and again holds only ids,
gold and hashes. The XNLI items are exactly the 600 frozen for Study 1; nothing is resampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

SEED = 20260919
N_PER_DATASET = 600
LANGS = ("en", "ru")

# HF commit hashes queried 2026-09-20 (xnli, massive) and 2026-09-21 (belebele); also recorded in
# PREREG.md / PREREG2.md and manifest.json.
REVISIONS = {
    "facebook/xnli": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
    "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
}

XNLI_MISALIGNED_ROWS = {2805, 2806}

DATASETS = {
    "xnli": {"repo": "facebook/xnli", "split": "test", "n_source": 5010},
    "massive": {"repo": "mteb/amazon_massive_intent", "split": "test", "n_source": 2974},
}

# --- Study 2 ("Panorama") -----------------------------------------------------------------------
# The 15 XNLI configs; every one is parallel to `en` by row index in the pinned test split.
XNLI_LANGS = ("ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh")
# Belebele: 900 four-way reading-comprehension questions per language, parallel by (link, question_number).
BELEBELE_LANGS = {"en": "eng_Latn", "ru": "rus_Cyrl"}
BELEBELE_CHOICES = {"1": "A", "2": "B", "3": "C", "4": "D"}

DATASETS2 = {
    "xnli": {"repo": "facebook/xnli", "split": "test", "n_source": 5010},
    "belebele": {"repo": "facebook/belebele", "split": "test", "n_source": 900},
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# One representative block per non-Latin script; a text "contains the script" if any char matches.
# Latin-script languages (de, en, es, fr, sw, tr, vi) have no cheap discriminating check and are skipped.
_SCRIPTS = {
    "bg": _CYRILLIC,
    "ru": _CYRILLIC,
    "ar": re.compile(r"[\u0600-\u06FF]"),  # Arabic
    "ur": re.compile(r"[\u0600-\u06FF]"),  # Arabic (Urdu uses the same block)
    "hi": re.compile(r"[\u0900-\u097F]"),  # Devanagari
    "th": re.compile(r"[\u0E00-\u0E7F]"),  # Thai
    "el": re.compile(r"[\u0370-\u03FF]"),  # Greek and Coptic
    "zh": re.compile(r"[\u4E00-\u9FFF]"),  # CJK Unified Ideographs
}


def state_hash(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load(repo: str, config: str, split: str):
    return load_dataset(repo, config, split=split, revision=REVISIONS[repo])


def load_parallel(dataset: str) -> pd.DataFrame:
    """One row per item with `item_id`, `gold`, and `state_<lang>` dicts for both languages.

    Asserts the parallel join (same n, same gold on every item) before returning — the whole
    paired analysis rides on it.
    """
    spec = DATASETS[dataset]
    if dataset == "xnli":
        en = _load(spec["repo"], "en", spec["split"])
        ru = _load(spec["repo"], "ru", spec["split"])
        assert len(en) == len(ru) == spec["n_source"], (len(en), len(ru))
        names = en.features["label"].names
        lab_en, lab_ru = np.asarray(en["label"]), np.asarray(ru["label"])
        # Parallel by row index — except two rows whose hypotheses are swapped within their
        # premise triple in the ru file (each language's label matches its own text). They are
        # excluded from the pool; the set is asserted so a revision change cannot pass silently.
        misaligned = set(np.where(lab_en != lab_ru)[0].tolist())
        assert misaligned == XNLI_MISALIGNED_ROWS, f"XNLI misaligned rows changed: {sorted(misaligned)}"
        keep = [i for i in range(len(en)) if i not in misaligned]
        df = pd.DataFrame(
            {
                "item_id": [str(i) for i in keep],  # row index in the pinned test split
                "gold": [names[lab_en[i]] for i in keep],
                "state_en": [{"premise": en[i]["premise"], "hypothesis": en[i]["hypothesis"]} for i in keep],
                "state_ru": [{"premise": ru[i]["premise"], "hypothesis": ru[i]["hypothesis"]} for i in keep],
            }
        )
    elif dataset == "massive":
        en = _load(spec["repo"], "en", spec["split"]).to_pandas()
        ru = _load(spec["repo"], "ru", spec["split"]).to_pandas()
        m = pd.merge(en, ru, on="id", suffixes=("_en", "_ru"), how="inner", validate="one_to_one")
        assert len(m) == len(en) == len(ru) == spec["n_source"], (len(m), len(en), len(ru))
        assert (m["label_en"] == m["label_ru"]).all(), "MASSIVE: gold differs between en and ru rows"
        df = pd.DataFrame(
            {
                "item_id": m["id"].astype(str),
                "gold": m["label_en"],
                "state_en": [{"utterance": t} for t in m["text_en"]],
                "state_ru": [{"utterance": t} for t in m["text_ru"]],
            }
        )
    else:
        raise KeyError(dataset)

    # The ru config really is Russian, and the two sides really differ.
    ru_texts = df["state_ru"].map(lambda s: " ".join(s.values()))
    assert (ru_texts.map(lambda t: bool(_CYRILLIC.search(t)))).mean() > 0.95, "ru slice is not Cyrillic"
    # A handful of untranslatable utterances (brand names) are identical across languages; fine.
    assert (df["state_en"] != df["state_ru"]).mean() > 0.99
    df.insert(0, "dataset", dataset)
    return df.sort_values("item_id", kind="stable").reset_index(drop=True)


def label_set(df: pd.DataFrame) -> list[str]:
    """Labels present in the source split, sorted — the Choice option space."""
    return sorted(df["gold"].unique().tolist())


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation by gold label (largest remainder), then uniform within class."""
    rng = np.random.default_rng(seed)
    counts = df["gold"].value_counts().sort_index()
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    short = n - int(alloc.sum())
    for lab in (exact - alloc).sort_values(ascending=False).index[:short]:
        alloc[lab] += 1
    assert alloc.sum() == n and (alloc <= counts).all()
    parts = []
    for lab in counts.index:  # sorted label order -> deterministic given the seed
        k = int(alloc[lab])
        if k == 0:
            continue
        pool = df.index[df["gold"] == lab].to_numpy()
        parts.append(np.sort(rng.choice(pool, size=k, replace=False)))
    idx = np.concatenate(parts)
    return df.loc[idx].sort_values("item_id", kind="stable").reset_index(drop=True)


def items_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The publishable frame: ids, gold, hashes. No text."""
    return pd.DataFrame(
        {
            "dataset": df["dataset"],
            "item_id": df["item_id"],
            "gold": df["gold"],
            "state_sha256_en": df["state_en"].map(state_hash),
            "state_sha256_ru": df["state_ru"].map(state_hash),
        }
    )


def build_items(n: int = N_PER_DATASET, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    frames, meta = [], {}
    for name in DATASETS:
        full = load_parallel(name)
        sample = stratified_sample(full, n, seed)
        frames.append(items_frame(sample))
        meta[name] = {
            "repo": DATASETS[name]["repo"],
            "revision": REVISIONS[DATASETS[name]["repo"]],
            "split": DATASETS[name]["split"],
            "n_source": DATASETS[name]["n_source"],
            "n_pool": len(full),
            "n_sampled": len(sample),
            "labels_in_split": label_set(full),
            "labels_in_sample": label_set(sample),
        }
    return pd.concat(frames, ignore_index=True), meta


def join_items(items: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Re-attach source text to the frozen sample and verify every hash."""
    full = load_parallel(dataset).set_index("item_id")
    sub = items[items["dataset"] == dataset]
    out = full.loc[sub["item_id"].to_numpy()].reset_index()
    for lang in LANGS:
        got = out[f"state_{lang}"].map(state_hash).to_numpy()
        want = sub[f"state_sha256_{lang}"].to_numpy()
        assert (got == want).all(), f"{dataset}/{lang}: state hash mismatch — HF revision drift?"
    assert (out["gold"].to_numpy() == sub["gold"].to_numpy()).all()
    return out


# --- Study 2 ("Panorama") -----------------------------------------------------------------------


def _script_share(texts: pd.Series, lang: str) -> float | None:
    """Share of texts containing the language's script; None when no cheap check exists (Latin)."""
    rx = _SCRIPTS.get(lang)
    if rx is None:
        return None
    return float(texts.map(lambda t: bool(rx.search(t))).mean())


def load_xnli_lang(lang: str) -> pd.DataFrame:
    """One XNLI language: `item_id` (row index), `gold` (label name), `state` ({premise, hypothesis}).

    Aligned against the `en` config by row index. Rows whose label differs from `en` are misaligned
    (in `ru` exactly XNLI_MISALIGNED_ROWS); they are dropped and the full set is recorded in
    `df.attrs["misaligned_rows"]` (sorted list). Non-Latin languages are also script-checked.
    """
    assert lang in XNLI_LANGS, lang
    spec = DATASETS2["xnli"]
    en = _load(spec["repo"], "en", spec["split"])
    d = en if lang == "en" else _load(spec["repo"], lang, spec["split"])
    assert len(en) == len(d) == spec["n_source"], (len(en), len(d))
    names = en.features["label"].names
    assert d.features["label"].names == names, f"xnli/{lang}: label names differ from en"
    lab_en, lab = np.asarray(en["label"]), np.asarray(d["label"])
    misaligned = set(np.where(lab != lab_en)[0].tolist())
    if lang == "ru":
        assert misaligned == XNLI_MISALIGNED_ROWS, f"XNLI misaligned rows changed: {sorted(misaligned)}"
    keep = [i for i in range(len(d)) if i not in misaligned]
    prem, hyp = d["premise"], d["hypothesis"]
    df = pd.DataFrame(
        {
            "item_id": [str(i) for i in keep],  # row index in the pinned test split
            "gold": [names[lab[i]] for i in keep],
            "state": [{"premise": prem[i], "hypothesis": hyp[i]} for i in keep],
        }
    )
    texts = df["state"].map(lambda st: " ".join(st.values()))
    share = _script_share(texts, lang)
    assert share is None or share > 0.95, f"xnli/{lang}: expected script in only {share:.3f} of texts"
    if lang != "en":
        # The config really is a translation, not a copy of en.
        en_states = [{"premise": en[i]["premise"], "hypothesis": en[i]["hypothesis"]} for i in keep]
        assert (df["state"] != pd.Series(en_states)).mean() > 0.99, f"xnli/{lang}: texts identical to en"
    df = df.sort_values("item_id", kind="stable").reset_index(drop=True)
    df.attrs["misaligned_rows"] = sorted(misaligned)
    df.attrs["script_share"] = share
    return df


def load_belebele(lang: str) -> pd.DataFrame:
    """One Belebele language: `item_id` (link#question_number), `gold` (A-D), `state`, `criteria`.

    `state` is {passage, question}; `criteria` maps A-D to the four answer texts, all in `lang`.
    Aligned against `eng_Latn` on item_id: the id sets must be equal (900) and gold identical per item.
    """
    spec = DATASETS2["belebele"]
    en = _load(spec["repo"], BELEBELE_LANGS["en"], spec["split"]).to_pandas()
    d = en if lang == "en" else _load(spec["repo"], BELEBELE_LANGS[lang], spec["split"]).to_pandas()
    assert len(en) == len(d) == spec["n_source"], (len(en), len(d))

    def frame(t: pd.DataFrame) -> pd.DataFrame:
        gold = t["correct_answer_num"].map(BELEBELE_CHOICES)
        assert gold.notna().all(), "belebele: correct_answer_num outside 1..4"
        out = pd.DataFrame(
            {
                "item_id": t["link"].astype(str) + "#" + t["question_number"].astype(str),
                "gold": gold,
                "state": [{"passage": p, "question": q} for p, q in zip(t["flores_passage"], t["question"])],
                "criteria": [
                    {"A": a, "B": b, "C": c, "D": e}
                    for a, b, c, e in zip(t["mc_answer1"], t["mc_answer2"], t["mc_answer3"], t["mc_answer4"])
                ],
            }
        )
        assert out["item_id"].is_unique, "belebele: duplicate (link, question_number)"
        return out.sort_values("item_id", kind="stable").reset_index(drop=True)

    df_en, df = frame(en), frame(d)
    assert set(df_en["item_id"]) == set(df["item_id"]) and len(df) == spec["n_source"], (
        f"belebele/{lang}: item ids differ from en"
    )
    assert (df_en["gold"].to_numpy() == df["gold"].to_numpy()).all(), f"belebele/{lang}: gold differs from en"
    texts = df["state"].map(lambda st: " ".join(st.values())) + " " + df["criteria"].map(lambda c: " ".join(c.values()))
    share = _script_share(texts, lang)
    assert share is None or share > 0.95, f"belebele/{lang}: expected script in only {share:.3f} of texts"
    if lang != "en":
        assert (df["state"] != df_en["state"]).mean() > 0.99, f"belebele/{lang}: texts identical to en"
    df.attrs["script_share"] = share
    return df


def build_items2(items_path: str | Path = "data/items.parquet") -> tuple[pd.DataFrame, dict]:
    """LONG frame (dataset, item_id, lang, gold, state_sha256, criteria_sha256) + meta. No text.

    XNLI rows are exactly the 600 item_ids frozen in `items_path` (Study 1), per language, minus the
    (item, lang) pairs misaligned in that language. Belebele is all 900 items in en and ru.
    """
    items = pd.read_parquet(items_path)
    frozen = items[items["dataset"] == "xnli"].reset_index(drop=True)
    xnli_ids = frozen["item_id"].tolist()
    assert len(xnli_ids) == N_PER_DATASET and frozen["item_id"].is_unique, len(xnli_ids)
    frozen = frozen.set_index("item_id")

    frames: list[pd.DataFrame] = []
    meta: dict = {
        "items_parquet": str(items_path),
        "items_parquet_sha256": hashlib.sha256(Path(items_path).read_bytes()).hexdigest(),
        "datasets": {},
    }

    xspec = DATASETS2["xnli"]
    xmeta = {
        "repo": xspec["repo"], "revision": REVISIONS[xspec["repo"]], "split": xspec["split"],
        "n_source": xspec["n_source"], "n_frozen_ids": len(xnli_ids), "langs": list(XNLI_LANGS), "per_lang": {},
    }
    for lang in XNLI_LANGS:
        full = load_xnli_lang(lang)
        idx = full.set_index("item_id")
        present = [i for i in xnli_ids if i in idx.index]  # frozen order, sans misaligned
        missing = [i for i in xnli_ids if i not in idx.index]
        sub = idx.loc[present]
        # Gold must match the Study 1 freeze item by item; en/ru hashes must match the Study 1 hashes.
        assert (sub["gold"].to_numpy() == frozen.loc[present, "gold"].to_numpy()).all(), f"xnli/{lang}: gold != Study 1"
        hashes = sub["state"].map(state_hash).to_numpy()
        if lang in LANGS:
            assert (hashes == frozen.loc[present, f"state_sha256_{lang}"].to_numpy()).all(), f"xnli/{lang}: hash != Study 1"
        frames.append(
            pd.DataFrame(
                {
                    "dataset": "xnli", "item_id": present, "lang": lang, "gold": sub["gold"].to_numpy(),
                    "state_sha256": hashes, "criteria_sha256": "",
                }
            )
        )
        mis_full = full.attrs["misaligned_rows"]
        xmeta["per_lang"][lang] = {
            "n_rows": len(present),
            "n_missing_from_sample": len(missing),
            "misaligned_rows_full": mis_full,
            "misaligned_rows_in_sample": sorted(int(i) for i in missing),
            "script_checked": lang in _SCRIPTS,
            "script_share": full.attrs["script_share"],
        }
        assert set(missing) == {str(i) for i in mis_full} & set(xnli_ids)
    meta["datasets"]["xnli"] = xmeta

    bspec = DATASETS2["belebele"]
    bmeta = {
        "repo": bspec["repo"], "revision": REVISIONS[bspec["repo"]], "split": bspec["split"],
        "n_source": bspec["n_source"], "configs": dict(BELEBELE_LANGS), "per_lang": {}, "checks": {},
    }
    golds = {}
    for lang in BELEBELE_LANGS:
        full = load_belebele(lang)
        frames.append(
            pd.DataFrame(
                {
                    "dataset": "belebele", "item_id": full["item_id"], "lang": lang, "gold": full["gold"],
                    "state_sha256": full["state"].map(state_hash), "criteria_sha256": full["criteria"].map(state_hash),
                }
            )
        )
        golds[lang] = full.set_index("item_id")["gold"]
        bmeta["per_lang"][lang] = {
            "n_rows": len(full), "script_checked": lang in _SCRIPTS, "script_share": full.attrs["script_share"],
            "gold_counts": full["gold"].value_counts().sort_index().to_dict(),
        }
    ids_equal = set(golds["en"].index) == set(golds["ru"].index)
    gold_equal = bool((golds["en"] == golds["ru"].reindex(golds["en"].index)).all())
    assert ids_equal and gold_equal and len(golds["en"]) == bspec["n_source"]
    bmeta["checks"] = {"en_ru_id_sets_equal": ids_equal, "n_items": len(golds["en"]), "gold_identical_en_ru": gold_equal}
    meta["datasets"]["belebele"] = bmeta

    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(["dataset", "item_id", "lang"]).any()
    return out, meta


def join_items2(items2: pd.DataFrame, dataset: str, lang: str) -> pd.DataFrame:
    """Re-attach source text for one (dataset, lang) of the Study 2 freeze and verify every hash.

    Returns `item_id, gold, state, criteria` in the frozen row order; `criteria` is None for xnli.
    """
    sub = items2[(items2["dataset"] == dataset) & (items2["lang"] == lang)]
    assert len(sub) > 0, f"{dataset}/{lang}: not in items2"
    if dataset == "xnli":
        full = load_xnli_lang(lang).set_index("item_id")
    elif dataset == "belebele":
        full = load_belebele(lang).set_index("item_id")
    else:
        raise KeyError(dataset)
    out = full.loc[sub["item_id"].to_numpy()].reset_index()
    got = out["state"].map(state_hash).to_numpy()
    assert (got == sub["state_sha256"].to_numpy()).all(), f"{dataset}/{lang}: state hash mismatch — HF revision drift?"
    if dataset == "belebele":
        got_c = out["criteria"].map(state_hash).to_numpy()
        assert (got_c == sub["criteria_sha256"].to_numpy()).all(), f"{dataset}/{lang}: criteria hash mismatch — HF revision drift?"
    else:
        assert (sub["criteria_sha256"] == "").all()
        out["criteria"] = None
    assert (out["gold"].to_numpy() == sub["gold"].to_numpy()).all()
    return out[["item_id", "gold", "state", "criteria"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the item sample (no text) to data/items.parquet")
    ap.add_argument("--out", default=None, help="default data/items.parquet, or data/items2.parquet with --study2")
    ap.add_argument("--n", type=int, default=N_PER_DATASET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--study2", action="store_true", help="build the Study 2 long frame from the frozen Study 1 ids")
    ap.add_argument("--items", default="data/items.parquet", help="Study 1 freeze to take the XNLI ids from (--study2)")
    a = ap.parse_args()
    if a.study2:
        out = a.out or "data/items2.parquet"
        items2, meta = build_items2(a.items)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        items2.to_parquet(out, index=False)
        Path(out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        print(items2.groupby(["dataset", "lang"], sort=False).size().to_string())
        for lang, m in meta["datasets"]["xnli"]["per_lang"].items():
            if m["n_missing_from_sample"]:
                print(f"xnli/{lang}: {m['n_missing_from_sample']} of {N_PER_DATASET} missing (misaligned rows {m['misaligned_rows_in_sample']})")
        return
    out = a.out or "data/items.parquet"
    items, meta = build_items(a.n, a.seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(out, index=False)
    Path(out).with_suffix(".meta.json").write_text(
        json.dumps({"seed": a.seed, "n_per_dataset": a.n, "datasets": meta}, indent=2, ensure_ascii=False) + "\n"
    )
    print(items.groupby("dataset").size().to_string())
    for name, m in meta.items():
        print(f"{name}: {m['n_sampled']} items, {len(m['labels_in_sample'])}/{len(m['labels_in_split'])} labels")


if __name__ == "__main__":
    main()
