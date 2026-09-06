# The archive statement, clause by clause

The thesis states what this repository contains. This document maps every
clause of that statement to a path, and records honestly what is in place and
what still has to be deposited before submission.

Legend: **in place** — the material is in the repository; **structure ready** —
the location, the file format and the command that produces it are fixed, and
the files themselves are still to be deposited from the run machines.

| Clause of the statement | Where it lives | Status |
|---|---|---|
| the experiment harness (the `harness/` directory) | [`harness/`](../harness) | in place |
| the frozen item lists of the core study with their sampling seed | [`harness/data/core_prompt/items.jsonl`](../harness/data/core_prompt/items.jsonl), `items.ids.jsonl`, seed `20260901` recorded in `build_items.py` and in every `run_meta.json` | in place |
| the raw model responses of every arm, run, and session, with the recorded model version strings and timestamps | `harness/results/core_prompt/<arm>/responses.jsonl`, `run_meta.json`, `sentinel_<session>.json` | structure ready |
| the scored outputs and the analysis summary from which every table and figure of the core study is derived | `harness/results/core_prompt/<arm>/scored.jsonl`, `score_meta.json`, `harness/results/core_prompt/summary.json` | structure ready |
| the blind annotation sheets and the agreement script | script: [`annotation_sheets.py`](../harness/experiments/core_prompt/annotation_sheets.py) (`build` and `kappa`); sheets: `harness/results/core_prompt/<arm>/annotation/` | script in place, sheets structure ready |
| the outputs of the local-model experiments (detectors, abstention tuning, quantisation, domain transfer) | `harness/results/r1_selfcheckgpt/`, `r2r3/`, `e3_abstention/`, `e5_transfer/` | structure ready |
| the configuration of the agentic simulation | [`harness/configs/e6_sim.yaml`](../harness/configs/e6_sim.yaml), echoed into `harness/results/e6_sim/propagation.json` | in place |
| the scripts that generate every figure and table | [`thesis/v3/`](../thesis/v3) — `build_core_tables.py`, `make_figures.py`, `build.py`, `to_latex.py` | in place (chapter sources and bibliography still to be added under `thesis/v3/chapters/` and `bib/`) |
| a README describing the folder structure, the software dependencies, and the commands of Appendix E | [`README.md`](../README.md), [`APPENDIX_E_COMMANDS.md`](APPENDIX_E_COMMANDS.md) | in place |
| the four source datasets are not redistributed; the item lists refer to them by identifier | [`DATA_SOURCES.md`](DATA_SOURCES.md); `items.ids.jsonl` | needs a decision — see below |
| model weights are not redistributed; the exact versions are recorded in the run metadata | `model_version` per response, model id in `run_meta.json`, roster in [`models.yaml`](../harness/configs/models.yaml) | in place |
| no personal data, no NDA material | [`DATA_SOURCES.md`](DATA_SOURCES.md) | in place |
| code under MIT, derived data under CC BY 4.0 | [`LICENSE`](../LICENSE), [`LICENSE-DATA`](../LICENSE-DATA) | in place, with one caveat below |

## Two things the statement does not yet match

1. **The item list is not identifier-only.** `items.jsonl` embeds the item text
   of the 800 sampled instances, and the prompt records in `responses.jsonl`
   quote it. Either deposit the identifier-only list and strip the prompt text
   from the response records, or amend the sentence and the data licence.
   `items.ids.jsonl` and the two options are described in
   [DATA_SOURCES.md](DATA_SOURCES.md).
2. **CC BY 4.0 on everything derived is too strong as stated.** SQuAD 2.0 and
   FEVER are share-alike, and their text is quoted in the item list and in the
   prompt records. `LICENSE-DATA` records this and states the two ways out.

## Deposit checklist

Before minting the persistent identifier and inserting it into the thesis:

- [ ] copy the result tree from the run machines into `harness/results/`
      (see [`harness/results/README.md`](../harness/results/README.md) for the
      per-directory list of required files)
- [ ] assemble `thesis/v3/data/manifest_v2.json` from those results, using
      `manifest_v2.schema.json` as the key list
- [ ] add the chapter sources and the bibliography under `thesis/v3/chapters/`
      and `thesis/v3/bib/`
- [ ] deposit the hidden-state tensors (`hidden_states.npz`, one per model) as
      a separate large-file bundle and note its identifier here
- [ ] re-run `build_core_tables.py`, `make_figures.py` and `build.py`, and
      confirm `build_report.json` is clean
- [ ] settle the two licence questions above
- [ ] mint the repository DOI, replace the placeholder in `CITATION.cff` and in
      the thesis sentence "[persistent repository identifier to be inserted
      before submission]"
- [ ] confirm no key, token or `.env` file was ever committed
      (`git log -p --all -- '*token*' '*.env' 'kaggle.json'`)
