"""R3: hidden-state probes for hallucination detection (SEP-style).

Consumes the hidden_states.npz produced by the Kaggle R2/R3 run: trains a
logistic-regression probe per representation (last-layer mean / final token /
penultimate-layer mean) to predict greedy-answer correctness, with question-level
cross-validation. This is the Layer 1 "single-pass risk signal": ~zero inference
cost because the states are free byproducts of generation.

Reference: Semantic Entropy Probes (Kossen et al. 2024) report probes recover
most of sampling-based SE's detection power at a fraction of the cost.

Usage: python train_probe.py harness/results/r2r3/<model>/hidden_states.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPRESENTATIONS = ["last_layer_mean", "last_layer_final", "penult_layer_mean"]


def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(scaler.transform(X[tr]), y[tr])
        scores[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return {"auroc": float(roc_auc_score(y, scores)), "folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    args = ap.parse_args()
    path = Path(args.npz_path)
    data = np.load(path)
    y = 1 - data["label_correct"]  # 1 = incorrect/hallucinated

    results = {"source": str(path), "n": int(len(y)), "prevalence_incorrect": float(y.mean())}
    for rep in REPRESENTATIONS:
        X = data[rep].astype(np.float32)
        # fp16 storage can overflow to +/-inf for large-activation models (e.g. Gemma)
        X = np.nan_to_num(X, nan=0.0, posinf=65504.0, neginf=-65504.0)
        results[rep] = cv_auroc(X, y)
        print(f"{rep}: AUROC {results[rep]['auroc']:.3f}")

    out = path.parent / "probe_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
