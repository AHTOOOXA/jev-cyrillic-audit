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


def build_question(dataset: str, instr_lang: str, criteria: dict | None = None) -> Choice:
    """Frozen instructions; frozen criteria, or per-item criteria (Belebele: the item's four answers
    under the frozen keys A-D) when the prompt file declares placeholders."""
    p = load_prompt(dataset, instr_lang)
    if criteria is None:
        assert "criteria_note" not in p, f"{dataset} needs per-item criteria"
        return Choice(instructions=p["instructions"], criteria=p["criteria"])
    assert list(criteria) == list(p["criteria"]), (list(criteria), list(p["criteria"]))
    return Choice(instructions=p["instructions"], criteria={k: str(v) for k, v in criteria.items()})


def check_state(dataset: str, instr_lang: str, state: dict) -> dict:
    """The state dict must carry exactly the frozen keys, in the frozen order; values as plain str."""
    keys = load_prompt(dataset, instr_lang)["state_schema"]
    assert list(state) == keys, (list(state), keys)
    return {k: str(v) for k, v in state.items()}


def from_choice(ans) -> dict:
    """ChoiceAnswer -> derived fields. p_max drives calibration; confidence is characterised separately."""
    p = {k: float(v) for k, v in ans.probabilities.items()}
    top = max(p, key=p.get)
    return {"pred": top, "p_max": p[top], "confidence": float(ans.confidence), "probs": p, "choice": ans.choice}
