"""Frozen prompts -> typesafe_sdk questions. Option keys are English identifiers in every cell."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from typesafe_sdk import Choice

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=None)
def load_prompt(dataset: str, instr_lang: str) -> dict:
    p = json.loads((PROMPTS_DIR / f"{dataset}.{instr_lang}.json").read_text())
    assert p["dataset"] == dataset and p["instr_lang"] == instr_lang
    return p


def build_question(dataset: str, instr_lang: str) -> Choice:
    p = load_prompt(dataset, instr_lang)
    return Choice(instructions=p["instructions"], criteria=p["criteria"])


def check_state(dataset: str, instr_lang: str, state: dict) -> dict:
    """The state dict must carry exactly the frozen keys, in the frozen order."""
    keys = load_prompt(dataset, instr_lang)["state_schema"]
    assert list(state) == keys, (list(state), keys)
    return state


def from_choice(ans) -> dict:
    """ChoiceAnswer -> derived fields. p_max drives calibration; confidence is characterised separately."""
    p = {k: float(v) for k, v in ans.probabilities.items()}
    top = max(p, key=p.get)
    return {"pred": top, "p_max": p[top], "confidence": float(ans.confidence), "probs": p, "choice": ans.choice}
