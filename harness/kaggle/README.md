# Kaggle runs — R2 (semantic entropy) + R3 (probe data)

One notebook, one roster model per run. Output artifact feeds R2 metrics directly
and provides training data for R3 probes (trained locally, CPU is enough).

## One-time setup (after Kaggle phone verification)

1. kaggle.com → **Create → Notebook** → File → **Import Notebook** → upload
   `r2r3_kaggle.ipynb` from this folder.
2. **Add-ons → Secrets** → Add secret: name `HF_TOKEN`, value = your HF read token.
3. Right panel → **Session options → Accelerator → GPU T4 x2**. Internet ON.

## Per run

1. In the config cell set `MODEL_ID` (roster order: `Qwen/Qwen3-1.7B`,
   `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B` with `LOAD_4BIT=1`,
   then cross-family models once HF access is granted).
2. **Run All**. Expected wall time at defaults (300 q × 11 generations):
   ~1–2 h for 1.7B on a T4; scale accordingly. First run: try `N_QUESTIONS=50`
   as a smoke test.
3. When done, download `/kaggle/working/out/` (Output panel) and place it at
   `harness/results/r2r3/<model-short-name>/` locally.

## What "good" looks like (reference values)

- `auroc_semantic_entropy`: literature reports ~0.75–0.83 on short-form QA for
  7B-class models; expect lower for 1.7B (that *is* the RQ1 result).
- `auroc_neg_logprob` should trail semantic entropy; if it doesn't, inspect.
- `greedy_accuracy` on TriviaQA: roughly 0.2–0.6 across 1.7B→14B.

## Weekly GPU budget (30 h)

| Run | Est. hours |
|-----|-----------|
| Qwen3-1.7B smoke (50 q) | 0.3 |
| Qwen3-1.7B full (300 q) | ~1.5 |
| Qwen3-4B | ~2.5 |
| Qwen3-8B | ~4 |
| Qwen3-14B (4-bit) | ~6 |

The whole primary ladder fits in one week's free quota.

## Phase 2 scripts (added 2026-09-02 → 2026-09-04)

All are pushed as *script kernels* via the CLI (`kaggle kernels push -p <dir>`,
`machine_shape: NvidiaTeslaT4`); a kernel dir = `kernel-metadata.json` + one `.py`
that is a small env-setting header concatenated with the canonical script here.

| Script | Purpose | Knobs (env) | Notes |
|---|---|---|---|
| `r2r3_generate.py` | R2/R3 generation + SE (TriviaQA) | `MODEL_ID N_QUESTIONS K_SAMPLES LOAD_4BIT` | checkpoints `*.partial.*` every 20 q |
| `r1_full_generate.py` | R1 SelfCheckGPT full run, local judge | — | |
| `e3_abstention.py` | E3 R-Tuning LoRA + 4-arm eval | `MODEL_ID LOAD_4BIT DEVICE_MAP TRAIN_BATCH` | header must `pip uninstall torchao`; 8B: `DEVICE_MAP=auto`; 14B: `LOAD_4BIT=1 DEVICE_MAP=auto` |
| `e5_transfer_generate.py` | E5 domain-shift (MMLU medical / legal MC) | `MODEL_ID N_PER_DOMAIN K_SAMPLES LOAD_4BIT DOMAINS` | same output schema as r2r3 → `experiments/e5_transfer/transfer_eval.py` |

Hardware lessons: 14B **fp16 does not fit** 2×T4 (29.6 GB weights vs 29 GiB) —
accelerate silently CPU-offloads `lm_head`, runs 2.5× slower and OOMs late;
use 4-bit on Kaggle or rent an A100. With 4-bit + peft, do **not** call
`prepare_model_for_kbit_training` (fp32-casts the 152k-vocab embeddings → OOM).

`collect_pending.sh` downloads finished kernels into `harness/results/<dest>` by
slug→dir map; edit the `DEST` table when adding kernels.
