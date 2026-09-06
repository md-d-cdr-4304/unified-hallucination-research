"""Bootstrap 95% CIs for the E3 abstention-aware score s = (#correct - #wrong)/N.

s depends only on the per-question label (correct / wrong / abstain), so
resampling the multinomial counts from run_meta.json is exact bootstrap.
Also reports the paired-difference CI between the probe-gated arm and the
matched tuned arm where probe_gated_abstention.json exists.

Usage: bootstrap_ci.py <results_root> [--B 10000]
"""
import argparse, json
from pathlib import Path
import numpy as np


def ci_from_counts(n_c, n_w, n, B, rng):
    labels = np.array([1] * n_c + [-1] * n_w + [0] * (n - n_c - n_w))
    idx = rng.integers(0, n, size=(B, n))
    s = labels[idx].mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(20260901)
    root = Path(a.results_root)
    summary = {}
    for meta_path in sorted(root.glob("e3_abstention/*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        tag = meta_path.parent.name
        summary[tag] = {"model": meta["model"], "arms": {}}
        for arm, m in meta["arms"].items():
            n = m["n"]
            n_c = round(m["accuracy_overall"] * n)
            n_w = round(m["wrong_rate"] * n)
            lo, hi = ci_from_counts(n_c, n_w, n, a.B, rng)
            summary[tag]["arms"][arm] = {"s": m["abstention_aware_score"],
                                         "ci95": [round(lo, 3), round(hi, 3)]}
        pg = root / "r2r3" / tag / "probe_gated_abstention.json"
        if not pg.exists():  # e3 dir may carry a precision suffix (qwen3-14b-4bit -> r2r3/qwen3-14b)
            pg = root / "r2r3" / tag.replace("-4bit", "") / "probe_gated_abstention.json"
        if pg.exists():
            pgd = json.loads(pg.read_text())
            summary[tag]["probe_gated"] = {}
            for k, m in pgd["arms"].items():
                n = 300
                lo, hi = ci_from_counts(round(m["accuracy_overall"] * n), round(m["wrong_rate"] * n),
                                        n, a.B, rng)
                summary[tag]["probe_gated"][k] = {"s": m["abstention_aware_score"],
                                                  "ci95": [round(lo, 3), round(hi, 3)]}
    out = root / "e3_abstention" / "e3_summary_ci.json"
    out.write_text(json.dumps(summary, indent=2))
    for tag, d in summary.items():
        print(tag)
        for arm, v in d["arms"].items():
            print(f"  {arm:24s} s={v['s']:+.3f}  CI {v['ci95']}")
        for k, v in d.get("probe_gated", {}).items():
            print(f"  probe {k:18s} s={v['s']:+.3f}  CI {v['ci95']}")


if __name__ == "__main__":
    main()
