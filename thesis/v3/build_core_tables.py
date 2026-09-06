"""Turn the core-study analysis (analyze.py summary.json) into the Chapter 5
tables and figure. Writes:
  chapters/_core_tables.html   HTML fragments (Table 5.1, 5.2, 5.3) to paste into ch05
  figures/f_core_deltas.png    reduction in percentage points per dataset and model, with CIs
  figures/f_core_rates.png     baseline vs optimized hallucination rate per dataset and model
Usage: harness/.venv/bin/python thesis/v3/build_core_tables.py harness/results/core_prompt/summary.json
Every number comes from summary.json; nothing is typed by hand.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads(Path(sys.argv[1]).read_text())
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.linewidth": 0.6, "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
                     "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})
P = {"blue": "#6b8cae", "orange": "#c9a27a", "green": "#8fa88f", "mauve": "#a48fa6", "grey": "#9a9a9a", "dark": "#4a4a4a", "fill": "#eef1f5"}
DS = ["truthfulqa", "squad2", "fever", "kilt"]; DSL = {"truthfulqa": "TruthfulQA", "squad2": "SQuAD 2.0", "fever": "FEVER", "kilt": "KILT"}
NAME = {"ollama--llama3-8b-instruct-q4_k_m": "Llama 3 8B (4-bit, local)", "hf--google-gemma-3-4b-it": "Gemma 3 4B (local)", "nim--openai-gpt-oss-20b": "gpt-oss-20b (cloud)"}
models = [m for m in S if m in NAME] or list(S)
label = lambda m: NAME.get(m, m)
COL = [P["orange"], P["mauve"], P["blue"], P["green"]]


def get(m, ds, metric):
    return S[m]["datasets"].get(ds, {}).get("metrics", {}).get(metric)


def fmt(x, d=1): return "" if x is None else f"{x:.{d}f}"


def pct(v): return f"{100 * v:.1f}"


rows = []
# ---- Table 5.1: hallucination rate per dataset and model, both arms, with delta and tests ----
h = '<table class="data"><caption><span class="tabnum">Table 5.1:</span> Hallucination rate under the baseline and the optimized prompt, per dataset and model, over all items and runs. Rates are percentages with 95 percent bootstrap intervals. The reduction is optimized minus baseline in percentage points; p is McNemar\'s exact test after Bonferroni correction over the four categories; h is Cohen\'s h.</caption>\n'
h += '<thead><tr><th>Dataset</th><th>Model</th><th>Baseline</th><th>Optimized</th><th>Reduction (pp)</th><th>p (corrected)</th><th>h</th></tr></thead>\n<tbody>\n'
for ds in DS:
    first = True
    for m in models:
        x = get(m, ds, "hallucination_rate")
        if not x or "comparison" not in x: continue
        b, o, c = x["baseline"], x["optimized"], x["comparison"]
        mark = "*" if c["significant_bonferroni"] else ""
        h += f'<tr>{"<td rowspan=\"%d\">%s</td>" % (sum(1 for mm in models if get(mm, ds, "hallucination_rate") and "comparison" in get(mm, ds, "hallucination_rate")), DSL[ds]) if first else ""}<td>{label(m)}</td>'
        h += f'<td>{pct(b["rate"])} [{pct(b["ci95"][0])}, {pct(b["ci95"][1])}]</td><td>{pct(o["rate"])} [{pct(o["ci95"][0])}, {pct(o["ci95"][1])}]</td>'
        h += f'<td>{c["delta_pp"]:+.1f} [{c["delta_ci95_pp"][0]:+.1f}, {c["delta_ci95_pp"][1]:+.1f}]</td><td>{c["mcnemar"]["p_bonferroni"]:.3g}{mark}</td><td>{c["cohen_h"]:+.2f}</td></tr>\n'
        first = False
h += '</tbody></table>\n'
rows.append(h)

# ---- Table 5.2: the four plan metrics per dataset and model ----
METS = [("fabrication_rate", "Fabrication"), ("incorrect_answer_rate", "Incorrect answer"), ("false_positive_rate", "False positive"), ("correct_abstention_rate", "Correct abstention"), ("over_abstention_rate", "Over-abstention"), ("format_compliance", "Format compliance")]
h = '<table class="data"><caption><span class="tabnum">Table 5.2:</span> The plan\'s metrics per dataset and model, baseline → optimized, in percent. False positive and correct abstention are defined on unanswerable items only; incorrect answer and over-abstention on answerable items only.</caption>\n'
h += '<thead><tr><th>Dataset</th><th>Model</th>' + "".join(f"<th>{n}</th>" for _, n in METS) + '</tr></thead>\n<tbody>\n'
for ds in DS:
    for m in models:
        if not get(m, ds, "hallucination_rate"): continue
        h += f'<tr><td>{DSL[ds]}</td><td>{label(m)}</td>'
        for k, _ in METS:
            x = get(m, ds, k)
            h += f'<td>{pct(x["baseline"]["rate"])} → {pct(x["optimized"]["rate"])}</td>' if x and "baseline" in x and "optimized" in x else "<td></td>"
        h += "</tr>\n"
h += '</tbody></table>\n'
rows.append(h)

# ---- Table 5.3: hypotheses ----
h = '<table class="data"><caption><span class="tabnum">Table 5.3:</span> Verdicts on H1 (reduction in every category) and H2 (largest reduction on the abstention and citation categories), per model. A category counts as reduced when the corrected McNemar p is below 0.05 and the reduction is at least 10 points.</caption>\n'
h += '<thead><tr><th>Model</th><th>Categories reduced (significant and ≥ 10 pp)</th><th>H1</th><th>Largest reduction</th><th>H2</th></tr></thead>\n<tbody>\n'
for m in models:
    red = []; best = (None, 0.0)
    for ds in DS:
        x = get(m, ds, "hallucination_rate")
        if not x or "comparison" not in x: continue
        c = x["comparison"]
        if c["practically_significant"] and c["improved"]: red.append(DSL[ds])
        if c["delta_pp"] < best[1]: best = (DSL[ds], c["delta_pp"])
    h += f'<tr><td>{label(m)}</td><td>{", ".join(red) or "none"}</td><td>{"supported" if len(red) == 4 else "not supported"}</td><td>{best[0] or ""} ({best[1]:+.1f} pp)</td><td>{"supported" if best[0] in ("SQuAD 2.0", "KILT") else "not supported"}</td></tr>\n'
h += '</tbody></table>\n'
rows.append(h)
(HERE / "chapters" / "_core_tables.html").write_text("\n".join(rows)); print("wrote chapters/_core_tables.html")

# ---- Figure: reductions with CIs ----
fig, ax = plt.subplots(figsize=(6.2, 2.9)); w = 0.8 / max(len(models), 1); x = np.arange(len(DS))
for j, m in enumerate(models):
    d = []; lo = []; hi = []
    for ds in DS:
        c = (get(m, ds, "hallucination_rate") or {}).get("comparison")
        d.append(c["delta_pp"] if c else np.nan); lo.append(c["delta_ci95_pp"][0] if c else np.nan); hi.append(c["delta_ci95_pp"][1] if c else np.nan)
    d, lo, hi = map(np.array, (d, lo, hi))
    ax.bar(x + (j - (len(models) - 1) / 2) * w, d, w, color=COL[j % 4], label=label(m), edgecolor="none")
    ax.errorbar(x + (j - (len(models) - 1) / 2) * w, d, yerr=[d - lo, hi - d], fmt="none", ecolor=P["dark"], elinewidth=0.6, capsize=1.5)
ax.axhline(0, color="#888888", lw=0.6); ax.axhline(-10, color="#bbbbbb", lw=0.6, linestyle=":")
ax.set_xticks(x); ax.set_xticklabels([DSL[d] for d in DS]); ax.set_ylabel("Change in hallucination rate (pp)")
ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.savefig(HERE / "figures" / "f_core_deltas.png"); plt.close(); print("fig f_core_deltas")

# ---- Figure: rates both arms ----
fig, axes = plt.subplots(1, len(DS), figsize=(7.0, 2.4), sharey=True)
for ax, ds in zip(axes, DS):
    b = [ (get(m, ds, "hallucination_rate") or {}).get("baseline", {}).get("rate", np.nan) * 100 for m in models]
    o = [ (get(m, ds, "hallucination_rate") or {}).get("optimized", {}).get("rate", np.nan) * 100 for m in models]
    xx = np.arange(len(models)); ax.bar(xx - 0.19, b, 0.38, color=P["grey"], label="baseline"); ax.bar(xx + 0.19, o, 0.38, color=P["blue"], label="optimized")
    ax.set_xticks(xx); ax.set_xticklabels([label(m).split(" (")[0] for m in models], rotation=25, ha="right", fontsize=7.5); ax.set_title(DSL[ds], fontsize=9, color=P["dark"])
axes[0].set_ylabel("Hallucination rate (%)"); axes[0].legend(fontsize=7)
plt.savefig(HERE / "figures" / "f_core_rates.png"); plt.close(); print("fig f_core_rates")
