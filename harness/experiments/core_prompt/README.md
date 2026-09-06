# Core prompt study (thesis core, from the project plan of 2026-02-06)

Question: how much hallucination is caused by the prompt, and how much can a
structured prompt remove without touching the model? Baseline vs optimized
prompt, four datasets, four models, ten runs, paired statistics.

## Pipeline (run from `harness/` with `.venv/bin/python`)

| Step | Command | Output |
|---|---|---|
| 1. Fixed items (done) | `python experiments/core_prompt/build_items.py` | `data/core_prompt/items.jsonl` (800 items, 20 sentinel) |
| 1b. Identifier-only list | `python experiments/core_prompt/export_item_ids.py` | `data/core_prompt/items.ids.jsonl` (no source text; safe to redistribute) |
| 2. Generate | `BACKEND=openai N_RUNS=10 WORKERS=8 python experiments/core_prompt/generate.py` | `results/core_prompt/<backend>--<model>/responses.jsonl`, `run_meta.json`, `sentinel_<session>.json` |
| 3. Score | `python experiments/core_prompt/score.py results/core_prompt/<dir>` | `scored.jsonl`, `score_meta.json` |
| 4. Analyse | `python experiments/core_prompt/analyze.py results/core_prompt/<dir> ... --out results/core_prompt/summary.json` | metrics, CIs, McNemar, Wilcoxon, Bonferroni, Cohen's h |
| 5. Annotation sheets | `annotation_sheets.py build results/core_prompt/<dir> --n 100` | `annotation/sheet_annotator_{A,B}.csv`, `key.json`, `INSTRUCTIONS.md` |
| 6. Agreement | `annotation_sheets.py kappa <sheet_A.csv> <sheet_B.csv> --key key.json --out agreement.json` | Cohen's kappa, raw agreement, disagreements, scorer-vs-human agreement |

Tables and figures are built from step 4 by `thesis/v3/build_core_tables.py`.
The full command list for the whole thesis is `docs/APPENDIX_E_COMMANDS.md`.

Backends: `openai` (gpt-4o-mini, needs OPENAI_API_KEY), `gemini` (needs GEMINI_API_KEY; check gemini-1.5-flash is still served, else set MODEL_ID to the current Flash snapshot and record it), `moonshot` (moonshot-v1-8k, MOONSHOT_API_KEY), `ollama` (llama3:8b-instruct-q4_K_M; Kaggle kernel via `kaggle/build_core_prompt_kernel.py <slug> <dir> ollama`), `nim` (pilot/judge only), `hf` (transformers; extension bridge, saves log-probs and hidden states).

Full grid per model: 800 items × 2 prompts × 10 runs = 16,000 calls (+20 sentinel per session). Responses are cached in `results/.cache_core/`, so re-runs are free; the sentinel subset bypasses the cache by session id.

## Status 2026-09-05
- Pilot on NIM llama-3.2-11b (6 items/dataset, 2 runs) validated every step: `results/core_prompt/_pilot_nim/`.
- Kaggle weekly GPU quota exhausted; Ollama smoke kernel `uhr-core-ollama-smoke` not yet pushed.
- No OpenAI / Gemini / Moonshot keys on this machine yet.

## Scoring rules (plan Section 5.5)
- TruthfulQA: exact reference match, else judge (NIM llama-3.2-11b, T=0, cached) → CORRECT / INCORRECT (fabrication) / ABSTAIN.
- SQuAD 2.0: answerable → gold containment or F1 ≥ 0.5; unanswerable → refusal = correct abstention, any answer = false confidence.
- FEVER: VERDICT parsed (NEI first, then refute/support keywords); wrong label = unsupported assertion; NEI on decidable = over-abstention.
- KILT: gold-answer containment; cited title vs gold provenance → gold / exists_not_gold / fabricated (Wikipedia API) / none.
- Ambiguous rule: refusal phrase plus a correct claim = correct; hedge plus wrong claim = hallucination; refusal with no claim = abstain.
