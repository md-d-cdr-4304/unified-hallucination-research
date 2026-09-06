# Unified hallucination research

Experiment harness, data, and analysis for a master's thesis on hallucination
in large language models. The study asks one question at three levels:

1. **Prompt level** — how much hallucination is caused by the prompt, and how
   much can a structured prompt remove without touching the model?
2. **Agent level** — what happens to the errors that remain when the model is
   one step inside a chain of calls?
3. **Local-model level** — on a small, quantised, offline model: can the error
   be *detected* from the model's own signals, and can the model be *trained*
   to abstain instead?

This repository is the material behind the chain of evidence from the results,
through the analysis, to the conclusions of the thesis. `docs/ARCHIVE.md` maps
every clause of the thesis archive statement to a path in this tree and lists
what still has to be deposited before submission.

---

## Repository layout

```
README.md                      this file
LICENSE                        MIT, for the code
LICENSE-DATA                   CC BY 4.0, for the derived data, with the share-alike caveat
CITATION.cff                   how to cite the repository (DOI placeholder until deposit)
requirements.txt               CPU environment: APIs, scoring, statistics, figures, thesis build
requirements-gpu.txt           adds torch / transformers / peft / bitsandbytes for the GPU hosts

docs/
  ARCHIVE.md                   the archive statement, clause by clause, plus the deposit checklist
  APPENDIX_E_COMMANDS.md       every command that produces a result, in order (Appendix E)
  DATA_SOURCES.md              the four datasets, their licences, and what is not redistributed
  MODEL_ROSTER_AND_COMPUTE.md  why these models, on which hardware, and the lessons learned

harness/
  configs/
    models.yaml                the model roster: local ladder, cross-family, NIM judges
    e6_sim.yaml                the configuration of the agentic simulation
  data/core_prompt/
    items.jsonl                the frozen 800-item list of the core study (seed 20260901)
    items.ids.jsonl            the identifier-only view of the same list
    README.md                  how the list was drawn, and the redistribution caveat
  experiments/
    core_prompt/               the core study: build items, generate, score, analyse, annotate
    r1_selfcheckgpt/           R1: SelfCheckGPT reproduction on WikiBio
    r3_probes/                 R3: hidden-state probes and the quantisation ablation
    e3_abstention/             E3: probe-gated abstention and bootstrap CIs
    e5_transfer/               E5: cross-domain transfer of the probes
    e6_sim/                    E6-sim: Monte-Carlo propagation in an agent pipeline
  kaggle/                      script kernels, the job queue, and the collect/dispatch helpers
  modal/                       the same canonical jobs on a rented GPU
  src/halluc/                  shared library: chat client, run log, SelfCheck, WikiBio
  results/                     the evidence trail (tracked; see results/README.md)

thesis/v3/
  build_core_tables.py         Tables 5.1-5.3 and two core figures, from summary.json
  make_figures.py              every remaining figure, from the numbers manifest
  build.py                     chapters + bibliography + figures -> HTML/PDF
  to_latex.py                  the same sources -> a LaTeX project (BTH template)
  chapters/, bib/, data/       chapter sources, bibliography, and the numbers manifest
```

## The study

### Core study — the prompt effect

Only the prompt formulation varies; decoding is held constant at T = 0,
top_p = 1, fixed `max_tokens`. Four datasets, one hallucination-prone failure
mode each; 200 items per dataset, drawn once with seed `20260901` and frozen;
two prompt classes; ten runs per condition.

| Dataset | Failure mode | Items |
|---|---|---|
| TruthfulQA | factual fabrication | 200 |
| SQuAD 2.0 | false confidence on unanswerable items | 200, half unanswerable |
| FEVER | unsupported assertion | 200, balanced 3-way |
| KILT | citation grounding | 200 |

The optimized prompt class follows one uniform four-element framework:
a task boundary definition; an explicit abstention condition with a
standardised refusal phrase; a structured output requirement; and a
prohibition of unsupported inference.

Per model that is 800 items × 2 prompt classes × 10 runs = 16,000 calls, plus
20 sentinel calls per session.

### What is measured

Per model, dataset and prompt class: fabrication rate, incorrect-answer rate,
false-positive rate (answering when abstention is correct), correct-abstention
rate, over-abstention rate (the cost of the optimized prompt), overall
hallucination rate, and — for KILT — the unsupported-citation rate, split into
*not the gold page* and *the page does not exist*.

Each rate is computed per item as the fraction of the ten runs, then averaged
over items, with 95% bootstrap confidence intervals (10,000 resamples over
items, seed 20260901).

Baseline against optimized, paired at the item level: McNemar's exact test on
per-item majority-over-runs outcomes, Wilcoxon signed-rank on the per-item run
fractions, Cohen's h and matched-pairs rank-biserial r as effect sizes,
Bonferroni correction over the four task categories (α = 0.05/4), the relative
reduction in percent, and a 10-percentage-point practical-significance flag.

### Scoring rules

Automatic scoring is string matching (SQuAD-style normalisation) plus, where
the references are sentences, a fixed deterministic judge model, cached, whose
identity is recorded in `score_meta.json`.

- **TruthfulQA** — exact reference match, otherwise the judge decides
  CORRECT / INCORRECT (fabrication) / ABSTAIN.
- **SQuAD 2.0** — answerable: gold containment or F1 ≥ 0.5; unanswerable:
  a refusal is a correct abstention, any answer is false confidence.
- **FEVER** — the verdict is parsed (NEI first, then refute/support keywords);
  a wrong label is an unsupported assertion, NEI on a decidable claim is
  over-abstention.
- **KILT** — gold-answer containment, and the cited title against the gold
  provenance: gold / exists-but-not-gold / fabricated (checked once against the
  MediaWiki API and cached) / none.
- **Ambiguous cases** — a refusal phrase next to a correct claim counts as
  correct; a hedge in front of a wrong claim is still a hallucination; a refusal
  with no claim is an abstention.

### Human validation

Both researchers independently label a stratified random sample of 100 outputs
per model on blind sheets. Cohen's kappa and raw agreement are reported,
disagreements go to a consensus meeting, and cases that stay unresolved are
excluded from the metrics and reported as ambiguous. `annotation_sheets.py`
generates the blind sheets and computes the agreement; it also reports how
often the automatic scorer agreed with the humans.

### Guarding against silent model updates

API models move underneath a study. Three defences are built in: the model
version string returned by the provider is stored with **every** response,
next to the endpoint and a UTC timestamp; a 20-item sentinel subset is re-run
at the start of every session into `sentinel_<session>.json`, bypassing the
cache; and `analyze.py` runs a McNemar test between the first session and each
later one on those items.

## Software dependencies

Python 3.12. The pinned versions in `requirements.txt` are the ones the study
was run with: `openai`, `requests`, `datasets`, `numpy`, `pandas`, `pyarrow`,
`scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `PyYAML`, plus `weasyprint` and
`pypdf` for the thesis build. `requirements-gpu.txt` adds `torch`,
`transformers`, `accelerate`, `bitsandbytes` and `peft` for the GPU hosts;
on Kaggle most of those are preinstalled and the kernel headers pin only what
has to change.

```bash
python -m venv harness/.venv
harness/.venv/bin/pip install -r requirements.txt        # laptop
harness/.venv/bin/pip install -r requirements-gpu.txt    # GPU host
```

Credentials are read from the environment and never stored in the repository:
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, `NVIDIA_API_KEY`,
`HF_TOKEN`, and `~/.kaggle/kaggle.json` for the Kaggle CLI.

## Reproducing the study

The full command list, with every environment knob, is in
[docs/APPENDIX_E_COMMANDS.md](docs/APPENDIX_E_COMMANDS.md). The short path
through the core study:

```bash
PY=harness/.venv/bin/python                                   # Scripts/python.exe on Windows

$PY harness/experiments/core_prompt/build_items.py            # or use the archived items.jsonl
cd harness
BACKEND=openai N_RUNS=10 WORKERS=8 $PY experiments/core_prompt/generate.py
$PY experiments/core_prompt/score.py results/core_prompt/<arm>
$PY experiments/core_prompt/analyze.py results/core_prompt/<arm> ... --out results/core_prompt/summary.json
$PY experiments/core_prompt/annotation_sheets.py build results/core_prompt/<arm> --n 100
$PY experiments/core_prompt/annotation_sheets.py kappa <sheet_A.csv> <sheet_B.csv> --key <key.json>
cd ..
$PY thesis/v3/build_core_tables.py harness/results/core_prompt/summary.json
$PY thesis/v3/make_figures.py
$PY thesis/v3/build.py --out THESIS.pdf
```

Three properties make re-running cheap and honest: every response is cached by
a key that includes the model, the prompt and the decoding parameters, so a
re-run costs nothing and cannot silently change; the sentinel subset bypasses
that cache deliberately; and every seed in the study — sampling, bootstrap,
simulation, and the per-run decoding seed offset — is `20260901`.

The GPU work (R2/R3 generation, abstention tuning, domain transfer) runs as
Kaggle script kernels or Modal jobs over the same canonical scripts, and the
outputs are collected back into `harness/results/`. `docs/MODEL_ROSTER_AND_COMPUTE.md`
explains the roster, the platforms, and the hardware constraints that shaped
the runs — in particular why the 14B ladder point is 4-bit everywhere, and why
there is a quantisation ablation to bound that confound.

## The evidence trail

`harness/results/` is tracked in git rather than ignored, because it is the
part of the archive that connects a number in the thesis to the response that
produced it: raw responses with model versions and timestamps → scored
outputs → the analysis summary → the tables and figures.
[`harness/results/README.md`](harness/results/README.md) documents every
directory and file, and lists the deliberate exclusions: the response cache,
GPU checkpoints, the hidden-state tensors (deposited as a separate large-file
bundle) and model weights.

## Licences and redistribution

- Code: MIT (`LICENSE`).
- Derived data: CC BY 4.0 (`LICENSE-DATA`) — with one caveat that must be
  settled before deposit, because the item list quotes text from two
  share-alike datasets. `LICENSE-DATA` and `docs/DATA_SOURCES.md` set out the
  two ways to resolve it.
- The four source datasets are not redistributed as datasets; they are
  downloaded from the Hugging Face Hub by `build_items.py`, and the archived
  identifier-only list refers to the instances by their own identifiers.
- Model weights are not redistributed; the exact version used is recorded in
  the run metadata of every run.
- No personal data was collected, and no material here is subject to a
  non-disclosure agreement.

## Citation

See `CITATION.cff`. The persistent identifier of the deposited archive is
minted at submission time; until then the placeholder in that file and in the
thesis sentence is the only thing standing in for it.
