"""R1: SelfCheckGPT reproduction (Prompt variant) on WikiBio hallucination set.

Reference numbers (Manakul et al. 2023, Table 2, GPT-3.5 judge, 20 samples):
  NonFact AUC-PR 93.42 | Factual AUC-PR 67.09 | random baseline NonFact ~72.96

Usage:
  python run_r1.py --passages 10 --samples 5 --judge google/gemma-3-4b-it
Full reproduction: --passages 238 --samples 20 (approx 45k judge calls).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from halluc.client import NimClient
from halluc.runlog import ResultCard, RESULTS_DIR
from halluc.selfcheck import score_sentence
from halluc.wikibio import load_wikibio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--judge", default="google/gemma-3-4b-it")
    args = ap.parse_args()

    card = ResultCard(
        "r1_selfcheckgpt",
        {"passages": args.passages, "samples": args.samples, "judge": args.judge,
         "variant": "prompt", "dataset": "potsawee/wiki_bio_gpt3_hallucination"},
    )

    passages = load_wikibio(n_passages=args.passages, n_samples=args.samples)
    client = NimClient()

    rows = []
    for p in tqdm(passages, desc="passages"):
        for si, (sent, label, raw) in enumerate(zip(p.sentences, p.labels, p.raw_labels)):
            score, per_sample = score_sentence(client, args.judge, sent, p.samples)
            rows.append({
                "passage": p.idx, "sent_idx": si, "sentence": sent,
                "label": label, "raw_label": raw,
                "score": score, "per_sample": per_sample,
            })

    out_dir = RESULTS_DIR / "r1_selfcheckgpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sentence_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows])
    metrics = {
        "n_sentences": len(rows),
        "prevalence_nonfactual": float(y.mean()),
        "auc_pr_nonfactual": float(average_precision_score(y, s)),
        "auc_pr_factual": float(average_precision_score(1 - y, 1 - s)),
        "auroc": float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None,
        "client": client.stats(),
    }
    path = card.finish(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"result card: {path}")


if __name__ == "__main__":
    main()
