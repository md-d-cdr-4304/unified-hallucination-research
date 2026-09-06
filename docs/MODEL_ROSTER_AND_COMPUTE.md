# Model roster and compute

Referenced from `harness/configs/models.yaml`, which is the machine-readable
form of this document.

## Why the roster looks like this

The study needs a **size ladder within one family** (so that size is the only
variable) and a **cross-family check** (so that a finding is not an artefact of
one training recipe).

- Primary ladder: Qwen3 1.7B / 4B / 8B / 14B — Apache-2.0, ungated, four sizes
  with the same tokenizer and recipe. The 14B point runs 4-bit only on the
  free-tier GPUs, which is why the quantisation ablation exists: it bounds the
  precision confound on that ladder point.
- Cross-family: Llama 3.2 1B, Llama 3.1 8B, Mistral 7B Instruct v0.3,
  Gemma 3 4B — all gated on the Hub, so they need `HF_TOKEN` and an accepted
  licence per model.

Exact identifiers and parameter counts are in `harness/configs/models.yaml`.
Weights are never redistributed; the version actually loaded in a run is
recorded in that run's `run_meta.json`.

## API models

- Core study arms: `gpt-4o-mini` (OpenAI), a current Gemini Flash snapshot,
  `moonshot-v1-8k`. Provider-side snapshots move, so the model version string
  returned by the provider is stored with **every** response, and the 20-item
  sentinel subset is re-run at the start of every session to test for silent
  updates (McNemar in `analyze.py`).
- NIM catalogue: probed on 2026-09-01, only 14 of the 83 listed models were
  invocable on this account, and none of the 1–14B roster models were served.
  NIM is therefore used only for the judge
  (`meta/llama-3.2-11b-vision-instruct`, non-reasoning, clean Yes/No) and for
  frontier reference points (`openai/gpt-oss-120b`); the roster runs locally.

## Compute platforms

| Platform | Used for | Notes |
|---|---|---|
| Laptop (CPU) | API generation, scoring, statistics, probes, figures, thesis build | probes train on CPU in seconds; `requirements.txt` is enough |
| Kaggle (T4 ×2, 30 GPU-hours/week) | R2/R3 generation, E3 abstention tuning, E5 transfer, local core-prompt arms | script kernels pushed from `harness/kaggle/queue/`; two concurrent sessions, so `build_pair_kernel.py` packs two jobs onto the two T4s |
| Modal (rented GPU) | the same canonical scripts, in parallel with Kaggle | `harness/modal/run_job.py`; outputs land in the `uhr-results` volume and are copied to `harness/results/_modal/<name>/` |

Indicative Kaggle cost of the primary ladder (300 questions × 11 generations):
~0.3 h for a 1.7B smoke run of 50 questions, ~1.5 h for 1.7B full, ~2.5 h for
4B, ~4 h for 8B, ~6 h for 14B in 4-bit — the whole ladder fits inside one
week's free quota.

## Hardware lessons that shaped the runs

- **Qwen3-14B in fp16 does not fit 2×T4** (29.6 GB of weights against 29 GiB of
  VRAM). `accelerate` silently CPU-offloads `lm_head`, which runs about 2.5×
  slower and then OOMs late in the job. Use 4-bit on Kaggle, or rent an A100.
  This is the reason the 14B ladder point is 4-bit everywhere and appears as
  `qwen3-14b-4bit` in the E3 and E5 result keys.
- With 4-bit plus PEFT, do **not** call `prepare_model_for_kbit_training`: it
  fp32-casts the 152k-vocab embeddings and OOMs.
- The E3 kernel header must `pip uninstall torchao` before training.
- Device maps that worked: 8B with `DEVICE_MAP=auto`; 14B with `LOAD_4BIT=1
  DEVICE_MAP=auto`.
- Long GPU jobs checkpoint to `*.partial.*` every 20 questions, so a killed
  session resumes instead of restarting.

## Reference values used as sanity checks

- `auroc_semantic_entropy` around 0.75–0.83 for 7B-class models on short-form
  QA in the literature; lower at 1.7B — which is itself the RQ1 result, not a
  bug.
- `auroc_neg_logprob` should trail semantic entropy; if it does not, inspect
  the run before believing it.
- Greedy accuracy on TriviaQA roughly 0.2–0.6 across the 1.7B → 14B ladder.
