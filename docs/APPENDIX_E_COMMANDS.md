# Appendix E — the commands

Every result in the thesis is produced by one of the commands below, in this
order. Paths are relative to the repository root unless a command says
otherwise. `PY` stands for the interpreter of the project virtual environment
(`harness/.venv/bin/python`, or `harness/.venv/Scripts/python.exe` on Windows).

## E.1 Environment

```bash
python -m venv harness/.venv
harness/.venv/bin/pip install -r requirements.txt          # laptop: APIs, scoring, statistics, figures
harness/.venv/bin/pip install -r requirements-gpu.txt      # GPU host: adds torch, transformers, peft, bitsandbytes
```

API keys are read from the environment, never from a file in the repository:
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `MOONSHOT_API_KEY`, `NVIDIA_API_KEY`
(NIM judge), `HF_TOKEN` (gated weights), and `~/.kaggle/kaggle.json` for the
Kaggle CLI.

## E.2 The frozen item list

```bash
$PY harness/experiments/core_prompt/build_items.py        # 800 items, seed 20260901 -> data/core_prompt/items.jsonl
$PY harness/experiments/core_prompt/export_item_ids.py    # identifier-only view -> items.ids.jsonl
```

The archived `items.jsonl` is authoritative; re-running is a check.

## E.3 Core study — prompt effect

Run once per arm. `BACKEND` selects the provider; `MODEL_ID` overrides its
default model; the output directory defaults to
`harness/results/core_prompt/<backend>--<model>`.

```bash
cd harness
BACKEND=openai   N_RUNS=10 WORKERS=8 $PY experiments/core_prompt/generate.py
BACKEND=gemini   N_RUNS=10 WORKERS=8 $PY experiments/core_prompt/generate.py
BACKEND=moonshot N_RUNS=10 WORKERS=8 $PY experiments/core_prompt/generate.py
BACKEND=ollama   N_RUNS=10 WORKERS=4 $PY experiments/core_prompt/generate.py   # llama3:8b-instruct-q4_K_M
```

Knobs: `MODEL_ID`, `N_RUNS` (10), `PROMPTS` (`baseline,optimized`), `DATASETS`
(all four), `N_ITEMS_PER_DATASET` (0 = all), `MAX_TOKENS` (256), `WORKERS` (8),
`OUT_DIR`, `ITEMS_PATH`, `SESSION_ID` (default: UTC start time),
`SKIP_SENTINEL`, `EXTRA_BODY`, `LOAD_4BIT` (hf backend).
Decoding is fixed at T = 0, top_p = 1. Responses are cached in
`results/.cache_core/`, so a re-run costs nothing; the sentinel subset bypasses
the cache by session id. Full grid per model: 800 items × 2 prompt classes ×
10 runs = 16,000 calls, plus 20 sentinel calls per session.

Score, then analyse all arms together:

```bash
$PY experiments/core_prompt/score.py results/core_prompt/<arm>            # -> scored.jsonl, score_meta.json
$PY experiments/core_prompt/analyze.py results/core_prompt/<arm> ... \
    --out results/core_prompt/summary.json
```

`score.py` takes `--judge-backend` (default `nim`) and `--judge-model`
(default `meta/llama-3.2-11b-vision-instruct`); the judge runs at T = 0 and is
cached, and its identity is recorded in `score_meta.json`.
`analyze.py` produces every rate, 95% bootstrap CIs (10,000 resamples over
items, seed 20260901), McNemar's exact test and Wilcoxon signed-rank on
item-paired outcomes, Cohen's h, the Bonferroni correction over the four task
categories, and the sentinel-drift test between sessions.

### Blind dual annotation and agreement

```bash
$PY experiments/core_prompt/annotation_sheets.py build results/core_prompt/<arm> --n 100 --seed 20260901
# both annotators fill sheet_annotator_A.csv and sheet_annotator_B.csv independently
$PY experiments/core_prompt/annotation_sheets.py kappa \
    results/core_prompt/<arm>/annotation/sheet_annotator_A.csv \
    results/core_prompt/<arm>/annotation/sheet_annotator_B.csv \
    --key results/core_prompt/<arm>/annotation/key.json \
    --out results/core_prompt/<arm>/annotation/agreement.json
```

`build` draws a stratified sample (equal share per dataset × prompt class) and
writes two identical blind sheets plus `key.json` and `INSTRUCTIONS.md`.
`kappa` reports Cohen's kappa, raw agreement, the disagreement list for the
consensus meeting, and how often the automatic scorer agrees with the humans.
A filled `consensus` column becomes the final human label; rows resolved as
`AMBIGUOUS` are counted and excluded.

## E.4 Local-model experiments

### R1 — SelfCheckGPT reproduction

```bash
$PY harness/experiments/r1_selfcheckgpt/run_r1.py --passages 238 --samples 20 --judge google/gemma-3-4b-it
```

(`--passages 10 --samples 5` is the smoke test.)

### R2/R3 — generation, semantic entropy, hidden states

GPU work runs on Kaggle or Modal. Kaggle script kernels are built and pushed
from the queue directory (the account slug in the helper scripts is `mastaan`;
change it for another account):

```bash
kaggle kernels push -p harness/kaggle/queue/<kernel-dir>
bash harness/kaggle/dispatch_queue.sh <kernel-dir>...   # keeps both GPU slots busy
bash harness/kaggle/collect_pending.sh <kernel-slug>    # downloads into harness/results/<dest>
bash harness/kaggle/watch_core.sh                       # watches the two core-prompt kernels
```

Kernel builders:

```bash
$PY harness/kaggle/build_core_prompt_kernel.py <slug> <out_dir> ollama
$PY harness/kaggle/build_core_prompt_kernel.py <slug> <out_dir> hf MODEL_ID=Qwen/Qwen3-4B [LOAD_4BIT=1]
$PY harness/kaggle/build_pair_kernel.py <slug> <out_dir> \
    e5-1p7b:e5_transfer_generate.py:0:MODEL_ID=Qwen/Qwen3-1.7B \
    e5-4b:e5_transfer_generate.py:1:MODEL_ID=Qwen/Qwen3-4B
```

The same canonical scripts run on Modal:

```bash
modal run harness/modal/run_job.py --script r2r3_generate.py \
    --name r2r3-qwen3-14b-fp16 --gpu A100-40GB --env MODEL_ID=Qwen/Qwen3-14B,LOAD_4BIT=0
modal volume get uhr-results /<name> harness/results/_modal/<name>     # fetch later
```

Canonical GPU scripts and their environment knobs:
`r2r3_generate.py` (`MODEL_ID N_QUESTIONS K_SAMPLES LOAD_4BIT`),
`r1_full_generate.py`, `e3_abstention.py`
(`MODEL_ID LOAD_4BIT DEVICE_MAP TRAIN_BATCH`),
`e5_transfer_generate.py` (`MODEL_ID N_PER_DOMAIN K_SAMPLES LOAD_4BIT DOMAINS`).

### R3 — probes and the quantisation ablation

```bash
$PY harness/experiments/r3_probes/train_probe.py harness/results/r2r3/<model>/hidden_states.npz
$PY harness/experiments/r3_probes/quant_ablation.py harness/results/r2r3/<model> harness/results/r2r3/<model>-4bit
```

### E3 — abstention tuning

```bash
# LoRA training + four-arm evaluation run on the GPU host (kaggle/e3_abstention.py)
$PY harness/experiments/e3_abstention/probe_gated.py harness/results/r2r3/<model>/hidden_states.npz --match-abstain 0.72 0.95
$PY harness/experiments/e3_abstention/bootstrap_ci.py harness/results/e3_abstention --B 10000
```

### E5 — domain transfer

```bash
$PY harness/experiments/e5_transfer/transfer_eval.py \
    harness/results/r2r3/<model> harness/results/e5_transfer/<model>
```

## E.5 Agentic simulation

```bash
$PY harness/experiments/e6_sim/propagate.py                       # uses harness/configs/e6_sim.yaml
$PY harness/experiments/e6_sim/propagate.py --config harness/configs/e6_sim.yaml \
    --manifest thesis/v3/data/manifest_v2.json \
    --out harness/results/e6_sim/propagation.json --runs 20000 --seed 20260901
```

The simulation is calibrated on measured quantities (per-step error =
1 − greedy accuracy; detector quality = probe AUROC) read from the manifest.
The configuration actually used is echoed into the `config` key of
`propagation.json`.

## E.6 Figures, tables, and the thesis

```bash
$PY thesis/v3/build_core_tables.py harness/results/core_prompt/summary.json   # Tables 5.1-5.3 + two figures
$PY thesis/v3/make_figures.py                                                 # all remaining figures
$PY thesis/v3/build.py --out THESIS.pdf                                       # HTML/PDF build + build_report.json
$PY thesis/v3/to_latex.py                                                     # LaTeX project under thesis/v3/latex/
cd thesis/v3/latex && pdflatex thesis && bibtex thesis && pdflatex thesis && pdflatex thesis
```

`build_report.json` must list no unknown citation keys, no missing figures and
no remaining `DATA-NEEDED` / `CITE-NEEDED` markers before submission.
