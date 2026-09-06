# harness/results — the evidence trail

This tree is the part of the archive that carries the chain of evidence from
the raw model responses, through the scored outputs, to the numbers in the
thesis. It is **tracked in git** (the ignore rules exclude only caches,
in-flight checkpoints and large tensors — see the bottom of this file).

Every directory below is created by a command listed in
[docs/APPENDIX_E_COMMANDS.md](../../docs/APPENDIX_E_COMMANDS.md); nothing here
is edited by hand.

## Layout

```
results/
  core_prompt/                    core study: prompt effect, 4 datasets x 4 models x 10 runs
    <backend>--<model>/           one arm, e.g. ollama--llama3-8b-instruct-q4_k_m
      responses.jsonl             raw responses: item id, prompt class, run index,
                                  full prompt, full response, model version string
                                  returned by the provider, endpoint, UTC timestamp
      run_meta.json               backend, model id, decoding parameters, seeds,
                                  session id, item list hash, counts, wall time
      sentinel_<session>.json     the 20-item sentinel subset, re-run at every
                                  session start (silent model-update check)
      scored.jsonl                per response: correct / answered / abstained /
                                  hallucinated / hall_type, and the evidence used
      score_meta.json             judge model and version, cache hits, rule counts
      annotation/                 blind dual-annotator material
        sheet_annotator_A.csv     blind sheet (no automatic label)
        sheet_annotator_B.csv     identical blind sheet
        key.json                  automatic labels + sample metadata (n, seed, strata)
        INSTRUCTIONS.md           operational definition and label codes
        agreement.json            Cohen's kappa, raw agreement, disagreement list,
                                  scorer-vs-human agreement
    summary.json                  the analysis summary: every rate, bootstrap CI,
                                  McNemar, Wilcoxon, Cohen's h, Bonferroni result
                                  and sentinel-drift test behind Chapter 5
  r1_selfcheckgpt/                R1: SelfCheckGPT reproduction on WikiBio
  r2r3/<model>/                   R2/R3: generation, semantic entropy, detector AUROCs
      metrics.json                accuracy, auroc_se, auroc_neg_logprob, auroc_lexical
      probe_results.json          hidden-state probe AUROCs (train_probe.py)
      quant_ablation.json         fp16 vs 4-bit comparison (written into the 4-bit dir)
      hidden_states.npz           large tensor, deposited separately (see below)
  e3_abstention/<model>/          abstention tuning: four arms A-D + probe gating
      run_meta.json               arm results, training configuration, seed
      probe_gated_abstention.json probe-gated arm at matched abstention rates
      bootstrap_ci.json           95% CIs on the abstention-aware score s
  e5_transfer/<model>/<domain>/   domain transfer (MMLU medical / legal)
      transfer_results.json       in-domain OOF vs transferred probe AUROC, SE, gap
  e6_sim/
      propagation.json            agentic simulation output; the configuration that
                                  produced it is echoed in its "config" key and
                                  lives in ../configs/e6_sim.yaml
  _modal/<name>/                  raw job output fetched from the Modal volume
  .cache_core/                    response and judge cache (not tracked)
```

## What must be present before the archive is deposited

- [ ] `core_prompt/<arm>/` for every arm of the core study, each with
      `responses.jsonl`, `run_meta.json`, `sentinel_*.json`, `scored.jsonl`,
      `score_meta.json`
- [ ] `core_prompt/summary.json`
- [ ] `core_prompt/<arm>/annotation/` with both blind sheets, `key.json`,
      `INSTRUCTIONS.md` and `agreement.json`
- [ ] `r1_selfcheckgpt/`, `r2r3/<model>/`, `e3_abstention/<model>/`,
      `e5_transfer/<model>/` for every model in the roster
- [ ] `e6_sim/propagation.json`
- [ ] `thesis/v3/data/manifest_v2.json`, assembled from the above

`docs/ARCHIVE.md` maps each of these back to the sentence of the thesis
archive statement that promises it.

## Deliberate exclusions

| Excluded | Why | How to get it |
|---|---|---|
| `.cache_core/` | response/judge cache, purely derivable | rebuilt on the next run |
| `*.partial.*` | GPU-kernel checkpoints written every 20 questions | superseded by the final file |
| `hidden_states.npz` | hundreds of MB per model; git is the wrong medium | regenerate with the R2/R3 kernel, or take the large-file bundle deposited alongside the repository |
| model weights | licensed by their publishers, never redistributed | Hugging Face, at the exact version recorded in `run_meta.json` |
