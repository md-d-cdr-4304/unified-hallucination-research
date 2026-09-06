"""Core prompt study analysis (project plan Sections 5.5-5.6).

Per (model, dataset, prompt class):
  fabrication_rate        outputs containing information not supported by context / ground truth
  incorrect_answer_rate   wrong answer when a correct answer exists (answerable items)
  false_positive_rate     answer given when the correct behaviour is to abstain (unanswerable items)
  correct_abstention_rate correct "insufficient information" on unanswerable items
  over_abstention_rate    abstention on answerable items (cost of the optimized prompt)
  hallucination_rate      any of the three operational conditions
  unsupported_citation_rate (KILT only; strict = not the gold page, fabricated = page does not exist)
Each rate is computed per item as the fraction of the N runs, then averaged over
items; 95% bootstrap CIs (10,000 resamples over items, seed 20260901).

Baseline vs optimized, paired at the item level:
  McNemar (exact, binomial) on per-item majority-over-runs binary outcomes
  Wilcoxon signed-rank on per-item run-fractions (continuous rates)
  Cohen's h (binary rates) and matched-pairs rank-biserial r (Wilcoxon)
  Bonferroni over the four task categories (alpha 0.05 / 4)
  relative reduction (%) and the 10-percentage-point practical-significance flag
Sentinel drift: McNemar between the first and each later session on the 20 sentinel items.

Usage: python analyze.py <result_dir> [<result_dir> ...] [--out summary.json]
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

SEED = 20260901
ALPHA = 0.05
N_CATEGORIES = 4
PRACTICAL_PP = 10.0

METRICS = {  # metric -> (item filter, per-response indicator)
    "hallucination_rate":        (lambda it: True,                 lambda s: s["hallucinated"]),
    "fabrication_rate":          (lambda it: True,                 lambda s: s["hallucinated"] and s["hall_type"] in ("fabrication", "false_confidence", "unsupported_assertion", "unsupported_citation")),
    "incorrect_answer_rate":     (lambda it: it["answerable"],     lambda s: s["answered"] and not s["correct"]),
    "false_positive_rate":       (lambda it: not it["answerable"], lambda s: s["answered"]),
    "correct_abstention_rate":   (lambda it: not it["answerable"], lambda s: s["abstained"]),
    "over_abstention_rate":      (lambda it: it["answerable"],     lambda s: s["abstained"]),
    "format_compliance":         (lambda it: True,                 lambda s: s["structured"]),
    "accuracy":                  (lambda it: it["answerable"],     lambda s: s["correct"]),
    "unsupported_citation_rate": (lambda it: True,                 lambda s: s.get("source_status") in ("fabricated", "exists_not_gold")),
    "fabricated_citation_rate":  (lambda it: True,                 lambda s: s.get("source_status") == "fabricated"),
    "structured_output_rate":    (lambda it: True,                 lambda s: s["structured"]),
}
LOWER_IS_BETTER = {"hallucination_rate", "fabrication_rate", "incorrect_answer_rate", "false_positive_rate", "over_abstention_rate",
                   "unsupported_citation_rate", "fabricated_citation_rate"}


def cohen_h(p1, p2):
    f = lambda p: 2 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return f(p1) - f(p2)


def mcnemar_exact(b, c):
    """b = baseline-only positives, c = optimized-only positives (discordant pairs)."""
    n = b + c
    if n == 0: return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def wilcoxon_rb(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) < 5: return None, None
    try:
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None, None
    r = stats.rankdata(np.abs(d)); rpos = r[d > 0].sum(); rneg = r[d < 0].sum()
    rb = (rpos - rneg) / (rpos + rneg) if (rpos + rneg) else 0.0
    return float(res.pvalue), float(rb)


def bootstrap_ci(vals, n=10000, seed=SEED):
    vals = np.asarray(vals, float)
    if len(vals) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def per_item_fractions(scored, items, dataset, pc, metric):
    filt, ind = METRICS[metric]
    acc = defaultdict(list)
    for s in scored:
        if s["dataset"] != dataset or s["prompt_class"] != pc or s.get("ambiguous"): continue  # ambiguous rows are excluded from every rate
        it = items[s["item_id"]]
        if not filt(it): continue
        acc[s["item_id"]].append(1.0 if ind(s) else 0.0)
    ids = sorted(acc)
    return ids, np.array([np.mean(acc[i]) for i in ids])


def analyze_dir(rd: Path, items: dict) -> dict:
    scored = [json.loads(l) for l in open(rd / "scored.jsonl")]
    meta = json.loads((rd / "run_meta.json").read_text())
    out = {"model": meta["model"], "backend": meta["backend"], "n_runs": meta["n_runs"], "n_responses": len(scored),
           "model_versions": sorted({s["model_version"] for s in scored}), "datasets": {}}
    datasets = sorted({s["dataset"] for s in scored})
    for ds in datasets:
        d = {"n_items": len({s["item_id"] for s in scored if s["dataset"] == ds}), "metrics": {}}
        for metric in METRICS:
            if metric.endswith("citation_rate") and ds != "kilt": continue
            ids_b, fb = per_item_fractions(scored, items, ds, "baseline", metric)
            ids_o, fo = per_item_fractions(scored, items, ds, "optimized", metric)
            if len(ids_b) == 0 and len(ids_o) == 0: continue
            m = {}
            for pc, ids, f in (("baseline", ids_b, fb), ("optimized", ids_o, fo)):
                if len(ids):
                    lo, hi = bootstrap_ci(f)
                    # majority vote over the runs (plan Section 5.4, self-consistency signal): item counts if >= half its runs show the indicator
                    m[pc] = {"rate": float(f.mean()), "ci95": [lo, hi], "n_items": int(len(ids)), "majority_vote_rate": float(np.mean(f >= 0.5))}
            if "baseline" in m and "optimized" in m and ids_b == ids_o and len(ids_b) >= 2:
                pb, po = m["baseline"]["rate"], m["optimized"]["rate"]
                delta_pp = (po - pb) * 100
                rel = ((po - pb) / pb * 100) if pb > 0 else None
                # paired binary outcome: per-item majority over runs
                mb, mo = fb >= 0.5, fo >= 0.5
                b_only, o_only = int(np.sum(mb & ~mo)), int(np.sum(~mb & mo))
                p_mc = mcnemar_exact(b_only, o_only)
                p_w, rb = wilcoxon_rb(fb, fo)
                diff_ci = bootstrap_ci(fo - fb)
                m["comparison"] = {
                    "delta_pp": delta_pp, "delta_ci95_pp": [diff_ci[0] * 100, diff_ci[1] * 100], "relative_change_pct": rel,
                    "cohen_h": cohen_h(po, pb), "mcnemar": {"b_only": b_only, "o_only": o_only, "p": p_mc, "p_bonferroni": min(1.0, p_mc * N_CATEGORIES)},
                    "wilcoxon": {"p": p_w, "p_bonferroni": (min(1.0, p_w * N_CATEGORIES) if p_w is not None else None), "rank_biserial": rb},
                    "significant_bonferroni": bool(p_mc * N_CATEGORIES < ALPHA),
                    "improved": (po < pb) if metric in LOWER_IS_BETTER else (po > pb),
                    "practically_significant": bool(abs(delta_pp) >= PRACTICAL_PP and (p_mc * N_CATEGORIES < ALPHA)),
                }
            d["metrics"][metric] = m
        out["datasets"][ds] = d
    # sentinel drift between sessions (baseline prompt, raw text equality + hallucination label if scored)
    sess = sorted(p for p in rd.glob("sentinel_*.json") if not p.name.startswith("sentinel_scored_"))
    if len(sess) >= 2:
        first = {r["item_id"]: r["text"] for r in json.loads(sess[0].read_text())}
        drift = []
        for p in sess[1:]:
            cur = {r["item_id"]: r["text"] for r in json.loads(p.read_text())}
            same = [first[i].strip() == cur[i].strip() for i in first if i in cur]
            drift.append({"session": p.stem.replace("sentinel_", ""), "n": len(same), "identical_fraction": float(np.mean(same)) if same else None})
        out["sentinel_drift_vs_first"] = {"first_session": sess[0].stem.replace("sentinel_", ""), "later": drift}
        scored_sess = sorted(rd.glob("sentinel_scored_*.json"))
        if len(scored_sess) >= 2:
            lab = lambda p: {r["item_id"]: bool(r["hallucinated"]) for r in json.loads(p.read_text())}
            first = lab(scored_sess[0]); res = []
            for p in scored_sess[1:]:
                cur = lab(p); ids = [i for i in first if i in cur]
                b_only = sum(first[i] and not cur[i] for i in ids); o_only = sum(cur[i] and not first[i] for i in ids)
                res.append({"session": p.stem.replace("sentinel_scored_", ""), "n": len(ids), "hallucinated_first": sum(first[i] for i in ids), "hallucinated_later": sum(cur[i] for i in ids),
                            "label_agreement": float(np.mean([first[i] == cur[i] for i in ids])), "mcnemar_p": mcnemar_exact(b_only, o_only), "model_change_detected": mcnemar_exact(b_only, o_only) < 0.05})
            out["sentinel_drift_vs_first"]["labels"] = res
    out["n_ambiguous_excluded"] = sum(1 for s in scored if s.get("ambiguous"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out"); ap.add_argument("--items")
    a = ap.parse_args()
    items_path = Path(a.items) if a.items else Path(__file__).resolve().parents[2] / "data" / "core_prompt" / "items.jsonl"
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path)}
    res = {}
    for d in a.dirs:
        r = analyze_dir(Path(d), items); res[Path(d).name] = r
        print(f"\n### {r['backend']}/{r['model']}  runs={r['n_runs']}  versions={r['model_versions']}")
        for ds, dd in r["datasets"].items():
            for metric in ("hallucination_rate", "false_positive_rate", "correct_abstention_rate", "incorrect_answer_rate", "over_abstention_rate", "unsupported_citation_rate"):
                m = dd["metrics"].get(metric)
                if not m or "baseline" not in m or "optimized" not in m: continue
                c = m.get("comparison", {})
                print(f"  {ds:10s} {metric:26s} base {m['baseline']['rate']:.3f}  opt {m['optimized']['rate']:.3f}  "
                      f"Δ {c.get('delta_pp', float('nan')):+6.1f}pp  h {c.get('cohen_h', float('nan')):+.2f}  "
                      f"McNemar p {c.get('mcnemar', {}).get('p', float('nan')):.3f}  Wilcoxon p {c.get('wilcoxon', {}).get('p') if c.get('wilcoxon', {}).get('p') is not None else float('nan'):.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
