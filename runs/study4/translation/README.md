# Study 4 — machine translations of XNLI test pairs (EN→RU), separate vs joint

- `ru_sep.jsonl`, `ru_joint.jsonl`: 5,006 XNLI test items (`item_id` = row index in `facebook/xnli` test,
  revision `b8dd5d7a…`), `gold`, Russian `premise`, `hypothesis`.
  - **sep**: premise and hypothesis translated in separate requests (a premise shared by several pairs was
    translated once and reused).
  - **joint**: both texts translated in one request.
- `raw.jsonl`: every request (ids, served model, fingerprint, tokens, refusal text). `excluded.json`: items
  dropped from every Study-4 arm — rows 2805/2806 (misaligned in XNLI `ru`) and 1272/2050 (the translator
  refused the hypothesis when given without context: "I'm sorry, but I can't assist with that request.").
- Translator: `gpt-4.1-mini-2025-04-14`, temperature 0, seed 20260926, prompts in `prompts/translate.en-ru.json`
  (sha256 `c818af5a…`), code `src/jev_cyrillic_audit/translate.py`. OpenAI organisation data sharing was ON.

**Licence.** These translations are adaptations of XNLI (Conneau et al., 2018), which is licensed
[CC BY-NC 4.0](https://github.com/facebookresearch/XNLI/blob/main/LICENSE); they are distributed under the same
CC BY-NC 4.0 licence (non-commercial use, attribution to XNLI). The MIT licence of this repository does not apply
to this directory.
