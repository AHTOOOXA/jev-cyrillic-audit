"""Parallel RU/EN slices: pinned loads, join assertions, seeded stratified sample.

Nothing in this module writes source text to disk. `data/items.parquet` holds ids, gold labels,
join keys and SHA-256 hashes of the state; the runner re-joins against the Hugging Face cache.
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

# HF commit hashes queried 2026-09-20; also recorded in PREREG.md and manifest.json.
REVISIONS = {
    "facebook/xnli": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
    "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
}

XNLI_MISALIGNED_ROWS = {2805, 2806}

DATASETS = {
    "xnli": {"repo": "facebook/xnli", "split": "test", "n_source": 5010},
    "massive": {"repo": "mteb/amazon_massive_intent", "split": "test", "n_source": 2974},
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the item sample (no text) to data/items.parquet")
    ap.add_argument("--out", default="data/items.parquet")
    ap.add_argument("--n", type=int, default=N_PER_DATASET)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    items, meta = build_items(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(a.out, index=False)
    Path(a.out).with_suffix(".meta.json").write_text(
        json.dumps({"seed": a.seed, "n_per_dataset": a.n, "datasets": meta}, indent=2, ensure_ascii=False) + "\n"
    )
    print(items.groupby("dataset").size().to_string())
    for name, m in meta.items():
        print(f"{name}: {m['n_sampled']} items, {len(m['labels_in_sample'])}/{len(m['labels_in_split'])} labels")


if __name__ == "__main__":
    main()
