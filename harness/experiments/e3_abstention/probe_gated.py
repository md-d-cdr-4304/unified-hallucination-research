"""E3 comparison arm: probe-gated abstention (no weight updates).

Uses the Phase 1 hidden states: out-of-fold probe risk scores on the fixed
300-question set; abstain on the top-q fraction by risk. Reports the same
metrics as the E3 tuning arms at (a) the LoRA arm's abstention rate and
(b) a small coverage sweep.

Usage: probe_gated.py <hidden_states.npz> [--match-abstain 0.72 0.95]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def oof_scores(X, y, seed=0, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return scores


def arm_metrics(correct, abstain):
    n = len(correct)
    attempted = ~abstain
    n_c = int((correct & attempted).sum())
    n_w = int((~correct & attempted).sum())
    return {
        "abstain_rate": float(abstain.mean()),
        "accuracy_overall": n_c / n,
        "accuracy_on_attempted": n_c / max(1, attempted.sum()),
        "wrong_rate": n_w / n,
        "abstention_aware_score": (n_c - n_w) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--match-abstain", type=float, nargs="+", default=[0.72])
    ap.add_argument("--rep", default="last_layer_final")
    args = ap.parse_args()

    data = np.load(args.npz_path)
    correct = data["label_correct"].astype(bool)
    y = (~correct).astype(int)
    X = np.nan_to_num(data[args.rep].astype(np.float32), nan=0.0,
                      posinf=65504.0, neginf=-65504.0)
    risk = oof_scores(X, y)

    out = {"source": args.npz_path, "representation": args.rep,
           "base_accuracy": float(correct.mean()), "arms": {}}
    for q in sorted(set([0.3, 0.5, 0.85] + list(args.match_abstain))):
        thr = np.quantile(risk, 1 - q)
        abstain = risk >= thr
        out["arms"][f"abstain_{q:.2f}"] = arm_metrics(correct, abstain)

    path = Path(args.npz_path).parent / "probe_gated_abstention.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
