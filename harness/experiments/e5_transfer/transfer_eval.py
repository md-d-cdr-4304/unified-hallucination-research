"""E5: cross-domain transfer of hidden-state probes (RQ5).

For one model: train a probe on ALL 300 TriviaQA questions (source), apply it
unchanged to each domain-shifted set (medical / legal) and report AUROC. Compare
against (a) an in-domain out-of-fold probe (ceiling) and (b) the calibration-free
signals already computed on the domain (semantic entropy, -logprob).
Transfer gap = in-domain OOF AUROC - transferred AUROC.

Usage: transfer_eval.py <r2r3_model_dir> <e5_model_dir>
  e.g. transfer_eval.py results/r2r3/qwen3-4b results/e5_transfer/qwen3-4b
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd


def lenient_letter(t):
    """Answer letter from free text. Strips an echoed 'A:' answer prefix (the prompt
    ends in 'A:', which collides with option A), then looks for 'answer is C',
    '**B.', or a bare letter."""
    u = str(t).strip().upper()
    u = re.sub(r"^(?:A\s*:\s*)+", "", u)          # echoed prompt suffix, possibly repeated
    m = (re.search(r"(?:ANSWER|OPTION|DIAGNOSIS)\s*(?:IS|:)?\s*\**\s*\(?([ABCD])\b", u)
         or re.search(r"\*\*\s*([ABCD])\b", u) or re.search(r"\b([ABCD])\b", u))
    return m.group(1) if m else ""


def entropy_over_letters(samples):
    vals, counts = np.unique([lenient_letter(x) or "?" for x in samples], return_counts=True)
    ps = counts / counts.sum()
    return float(-(ps * np.log(ps)).sum()), int(len(vals))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPS = ["last_layer_mean", "last_layer_final", "penult_layer_mean"]


def load(npz):
    d = np.load(npz)
    y = 1 - d["label_correct"].astype(int)
    X = {r: np.nan_to_num(d[r].astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0) for r in REPS}
    return X, y


def fit(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X), y)


def oof(X, y, seed=0, folds=5):
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc, clf = fit(X[tr], y[tr]); s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src_dir"); ap.add_argument("e5_dir")
    ap.add_argument("--lenient", action="store_true",
                    help="re-derive correctness from generations.parquet with a lenient letter parser "
                         "(for models that explain before answering)")
    a = ap.parse_args()
    Xs, ys = load(Path(a.src_dir) / "hidden_states.npz")
    probes = {r: fit(Xs[r], ys) for r in REPS}
    out = {"source": a.src_dir, "domains": {}}
    for dom_dir in sorted(p for p in Path(a.e5_dir).iterdir() if (p / "hidden_states.npz").exists()):
        Xd, yd = load(dom_dir / "hidden_states.npz")
        meta = json.loads((dom_dir / "run_meta.json").read_text())
        res = {"n": int(len(yd)), "prevalence_incorrect": float(yd.mean()),
               "accuracy": meta["greedy_accuracy"], "unparsed_greedy_rate": meta.get("unparsed_greedy_rate"),
               "auroc_semantic_entropy": meta["auroc_semantic_entropy"],
               "auroc_neg_logprob": meta["auroc_neg_logprob"], "probe": {}}
        if a.lenient:
            allg = pd.read_parquet(dom_dir / "generations.parquet")
            g = allg.query("greedy").sort_values("q_idx")
            pred = g.text.map(lenient_letter)
            yd = (pred.values != g.gold.values).astype(int)
            # recompute SE over letter clusters from the raw samples (runtime SE used the buggy parser)
            se_vals = [entropy_over_letters(allg[(allg.q_idx == q) & (~allg.greedy)].text.tolist())[0] for q in g.q_idx]
            res.update({"relabelled": "lenient", "unparsed_greedy_rate_lenient": float((pred == "").mean()),
                        "pred_letter_dist": {k: int(v) for k, v in pred.value_counts().items()},
                        "accuracy": float(1 - yd.mean()), "prevalence_incorrect": float(yd.mean()),
                        "auroc_semantic_entropy": float(roc_auc_score(yd, se_vals)),
                        "auroc_neg_logprob": float(roc_auc_score(yd, -g.mean_logprob.values))})
        for r in REPS:
            sc, clf = probes[r]
            transferred = float(roc_auc_score(yd, clf.predict_proba(sc.transform(Xd[r]))[:, 1]))
            indomain = oof(Xd[r], yd)
            res["probe"][r] = {"transfer_auroc": transferred, "indomain_oof_auroc": indomain,
                               "transfer_gap": indomain - transferred}
        out["domains"][dom_dir.name] = res
        print(dom_dir.name, json.dumps(res, indent=1))
    (Path(a.e5_dir) / "transfer_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
