"""Quantization ablation: same model, same 300 questions, fp16 vs 4-bit.

Compares accuracy, detector AUROCs (SE, -logprob, lexical, probes) and
per-question agreement of greedy answers between two r2r3 run dirs.
Bounds the precision confound on the 14B (4-bit only) ladder point.

Usage: quant_ablation.py <fp16_dir> <4bit_dir>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from train_probe import cv_auroc, REPRESENTATIONS


def summarize(d: Path):
    meta = json.loads((d / "run_meta.json").read_text())
    data = np.load(d / "hidden_states.npz")
    y = 1 - data["label_correct"]
    probes = {}
    for rep in REPRESENTATIONS:
        X = np.nan_to_num(data[rep].astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
        probes[rep] = cv_auroc(X, y)["auroc"]
    g = pd.read_parquet(d / "generations.parquet").query("greedy").sort_values("q_idx")
    return {"dir": str(d), "load_4bit": meta.get("load_4bit"), "accuracy": meta["greedy_accuracy"],
            "auroc_se": meta["auroc_semantic_entropy"], "auroc_logprob": meta["auroc_neg_logprob"],
            "auroc_lexical": meta["auroc_lexical_diversity"], "probe": probes,
            "gpu_seconds": meta["wall_seconds"]}, g


def main():
    a, ga = summarize(Path(sys.argv[1])); b, gb = summarize(Path(sys.argv[2]))
    same_text = (ga.text.str.strip().str.lower().values == gb.text.str.strip().str.lower().values)
    same_label = (ga.correct.values == gb.correct.values)
    out = {"fp16": a, "quant": b,
           "delta_quant_minus_fp16": {
               "accuracy": b["accuracy"] - a["accuracy"], "auroc_se": b["auroc_se"] - a["auroc_se"],
               "auroc_logprob": b["auroc_logprob"] - a["auroc_logprob"],
               "probe_best": max(b["probe"].values()) - max(a["probe"].values()),
               "probe_final_tok": b["probe"]["last_layer_final"] - a["probe"]["last_layer_final"]},
           "greedy_agreement": {"identical_text_rate": float(same_text.mean()),
                                "same_correctness_rate": float(same_label.mean()),
                                "n": int(len(same_text))}}
    Path(sys.argv[2], "quant_ablation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
