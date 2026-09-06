# The numbers manifest

`manifest_v2.json` is the single place where the measured quantities of the
local-model experiments are collected, so that a figure never re-reads a raw
result file and never contains a hand-typed number. It is consumed by
`thesis/v3/make_figures.py` and by `harness/experiments/e6_sim/propagate.py`,
which calibrates the agentic simulation on the measured accuracy and detector
AUROC.

`manifest_v2.schema.json` records the required keys with `null` in place of
every value. It is documentation, not data: fill a copy from the result files.

## Where each block comes from

| Block | Source |
|---|---|
| `r2r3.<model>` | `harness/results/r2r3/<model>/` — `metrics.json` (`accuracy`, `auroc_se`, `auroc_neg_logprob`, `auroc_lexical`) and `probe_results.json` (`probe_best`, from `train_probe.py`) |
| `e3.<model>` | `harness/results/e3_abstention/<model>/run_meta.json` — the four arms A–D |
| `e3._bootstrap_ci` | `bootstrap_ci.py` over the same tree, plus `probe_gated_abstention.json` from `probe_gated.py` |
| `quant.<model>-4bit` | `quant_ablation.json` from `quant_ablation.py <fp16_dir> <4bit_dir>` |
| `e5.v2.<model>` | `harness/results/e5_transfer/<model>/transfer_results.json` from `transfer_eval.py` |

Model keys are the short names used throughout the repository: `qwen3-1p7b`,
`qwen3-4b`, `qwen3-8b`, `qwen3-14b` — `qwen3-14b-4bit` in the E3 and E5 blocks,
which ran 4-bit only — `llama32-1b`, `llama31-8b`, `mistral-7b`, `gemma3-4b`.
Keys beginning with `_` are metadata and are skipped when the size ladder is
plotted.

The core prompt study does **not** pass through this manifest: its tables and
figures are built straight from `harness/results/core_prompt/summary.json` by
`build_core_tables.py`.
