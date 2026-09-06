"""Figures for thesis v3. Plain, muted style: serif text, faded colours, thin
lines, light grid, no titles inside the image (captions carry the message).
Every number is read from the v2 manifest or from result files. Re-runnable:
    harness/.venv/bin/python thesis/v3/make_figures.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; R = ROOT / "harness" / "results"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
M = json.loads((HERE / "data" / "manifest_v2.json").read_text())
SIM = json.loads((R / "e6_sim" / "propagation.json").read_text())

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5, "grid.color": "#777777",
    "xtick.color": "#333333", "ytick.color": "#333333", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "lines.linewidth": 1.2, "lines.markersize": 4.5,
})
# muted palette
P = {"blue": "#6b8cae", "orange": "#c9a27a", "green": "#8fa88f", "mauve": "#a48fa6", "grey": "#9a9a9a",
     "dark": "#4a4a4a", "light": "#d9d9d9", "fill": "#eef1f5"}
FAMC = {"qwen": P["blue"], "llama": P["orange"], "mistral": P["green"], "gemma": P["mauve"]}
FAM = {"qwen3-1p7b": "qwen", "qwen3-4b": "qwen", "qwen3-8b": "qwen", "qwen3-14b": "qwen",
       "llama32-1b": "llama", "llama31-8b": "llama", "mistral-7b": "mistral", "gemma3-4b": "gemma"}
LABEL = {"qwen3-1p7b": "Qwen3 1.7B", "qwen3-4b": "Qwen3 4B", "qwen3-8b": "Qwen3 8B", "qwen3-14b": "Qwen3 14B (4-bit)",
         "llama32-1b": "Llama 3.2 1B", "llama31-8b": "Llama 3.1 8B", "mistral-7b": "Mistral 7B", "gemma3-4b": "Gemma 3 4B"}
ORDER = ["qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "llama32-1b", "llama31-8b", "mistral-7b", "gemma3-4b"]
QK = ["qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"]; SIZES = [1.7, 4, 8, 14]
E3K = {"qwen3-1p7b": "qwen3-1p7b", "qwen3-4b": "qwen3-4b", "qwen3-8b": "qwen3-8b", "qwen3-14b": "qwen3-14b-4bit"}
r2 = {k: v for k, v in M["r2r3"].items() if not k.startswith("_")}
e3 = M["e3"]; ci = e3["_bootstrap_ci"]


def save(name):
    plt.savefig(FIG / f"{name}.png"); plt.close(); print("fig", name)


def box(ax, x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec="#777777", lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=P["dark"], wrap=True)


# ---------- three levels diagram ----------
fig, ax = plt.subplots(figsize=(8.0, 2.9)); ax.set_xlim(0, 13.0); ax.set_ylim(0, 4.2); ax.axis("off"); ax.grid(False)
bw = 3.6; gap = 0.95; x0s = [0.2, 0.2 + bw + gap, 0.2 + 2 * (bw + gap)]
texts = ["Level 1: prompt\n\nbaseline vs optimized prompt,\nthree models, four failure modes",
         "Level 2: agent loop\n\nremaining errors inside\na chain of calls with tools",
         "Level 3: local model\n\nsmall, quantized, offline:\ndetect, and train to abstain"]
subs = ["prompting framework", "agentic framework", "detection and training framework"]
for x0, t, sub in zip(x0s, texts, subs):
    box(ax, x0, 1.35, bw, 1.75, t, P["fill"], fs=7.2)
    ax.text(x0 + bw / 2, 0.85, sub, ha="center", fontsize=8, color="#555555", style="italic")
for x0 in x0s[:2]:
    xa = x0 + bw + 0.12; xb = x0 + bw + gap - 0.12
    ax.annotate("", xy=(xb, 2.1), xytext=(xa, 2.1), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8))
    ax.text((xa + xb) / 2, 2.32, "residual", ha="center", fontsize=7, color="#666666")
ax.text(6.5, 3.75, "what the prompt cannot remove is what the later levels must catch", ha="center", fontsize=8, color="#555555")
save("d_three_levels")

# ---------- accuracy ladder ----------
fig, ax = plt.subplots(figsize=(6.2, 2.7))
acc = [r2[k]["accuracy"] for k in ORDER]
ax.bar(range(8), acc, color=[FAMC[FAM[k]] for k in ORDER], width=0.62, edgecolor="none")
for i, v in enumerate(acc): ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=P["dark"])
ax.set_xticks(range(8)); ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=28, ha="right")
ax.set_ylabel("Accuracy on 300 TriviaQA questions"); ax.set_ylim(0, 0.9)
save("f_accuracy_ladder")

# ---------- detector table as a light heatmap ----------
dets = ["auroc_se", "auroc_neg_logprob", "auroc_lexical", "probe_best"]; dl = ["Semantic entropy", "Log-probability", "Lexical diversity", "Probe (best)"]
mat = np.array([[r2[k][d] for d in dets] for k in ORDER])
fig, ax = plt.subplots(figsize=(5.0, 3.4)); ax.grid(False)
im = ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("m", ["#f7f7f7", "#8fa4bd"]), vmin=0.55, vmax=0.9, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(dl, rotation=22, ha="right"); ax.set_yticks(range(8)); ax.set_yticklabels([LABEL[k] for k in ORDER])
for i in range(8):
    for j in range(4):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=P["dark"], fontweight="bold" if mat[i, j] == mat[i].max() else "normal")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
save("f_detector_heatmap")

# ---------- Qwen3 scaling ----------
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for d, lab, c, mk in zip(dets, dl, [P["blue"], P["orange"], P["green"], P["mauve"]], ["o", "s", "^", "D"]):
    ax.plot(SIZES, [r2[k][d] for k in QK], marker=mk, label=lab, color=c)
ax.plot(SIZES, [r2[k]["accuracy"] for k in QK], marker="x", linestyle="--", color=P["grey"], label="Accuracy")
ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(["1.7B", "4B", "8B", "14B (4-bit)"]); ax.minorticks_off()
ax.set_xlabel("Qwen3 model size"); ax.set_ylabel("AUROC (detectors) / accuracy"); ax.set_ylim(0.2, 0.9)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
save("f_qwen_scaling")

# ---------- E3 arms: abstention-aware score by size ----------
arms = ["A_base_plain", "B_base_abstainprompt", "C_tuned_plain", "D_tuned_abstainprompt"]
al = ["A  base, plain prompt", "B  base, abstain prompt", "C  tuned, plain prompt", "D  tuned, abstain prompt"]
ac = [P["grey"], P["blue"], P["orange"], P["mauve"]]
fig, ax = plt.subplots(figsize=(5.6, 3.0)); w = 0.19; x = np.arange(4)
for i, (a, lab, c) in enumerate(zip(arms, al, ac)):
    s = [ci[E3K[k]]["arms"][a]["s"] for k in QK]; lo = [ci[E3K[k]]["arms"][a]["ci95"][0] for k in QK]; hi = [ci[E3K[k]]["arms"][a]["ci95"][1] for k in QK]
    ax.bar(x + (i - 1.5) * w, s, w, color=c, label=lab, edgecolor="none")
    ax.errorbar(x + (i - 1.5) * w, s, yerr=[np.array(s) - np.array(lo), np.array(hi) - np.array(s)], fmt="none", ecolor=P["dark"], elinewidth=0.6, capsize=1.5)
ax.axhline(0, color="#888888", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(["1.7B", "4B", "8B", "14B (4-bit)"])
ax.set_ylabel("Score s = (correct − wrong) / 300"); ax.legend(ncol=2, loc="upper left"); ax.set_ylim(-0.6, 0.8)
save("f_e3_arms")

# ---------- prompt-only abstention (arm B vs A): wrong rate and abstain rate ----------
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
wa = [e3[E3K[k]]["arms"]["A_base_plain"]["wrong_rate"] for k in QK]; wb = [e3[E3K[k]]["arms"]["B_base_abstainprompt"]["wrong_rate"] for k in QK]
ab = [e3[E3K[k]]["arms"]["B_base_abstainprompt"]["abstain_rate"] for k in QK]
aa = [e3[E3K[k]]["arms"]["A_base_plain"]["accuracy_overall"] for k in QK]; ab_acc = [e3[E3K[k]]["arms"]["B_base_abstainprompt"]["accuracy_overall"] for k in QK]
x = np.arange(4); w = 0.36
axes[0].bar(x - w / 2, wa, w, color=P["grey"], label="plain prompt"); axes[0].bar(x + w / 2, wb, w, color=P["blue"], label="abstain prompt")
axes[0].set_xticks(x); axes[0].set_xticklabels(["1.7B", "4B", "8B", "14B*"]); axes[0].set_ylabel("Wrong answers (share of 300)"); axes[0].legend(); axes[0].set_ylim(0, 0.8)
axes[1].bar(x - w / 2, aa, w, color=P["grey"], label="correct, plain"); axes[1].bar(x + w / 2, ab_acc, w, color=P["blue"], label="correct, abstain prompt")
axes[1].bar(x + w / 2, ab, w, bottom=ab_acc, color=P["fill"], edgecolor=P["blue"], lw=0.6, label="abstained")
axes[1].set_xticks(x); axes[1].set_xticklabels(["1.7B", "4B", "8B", "14B*"]); axes[1].set_ylabel("Share of 300 questions"); axes[1].legend(fontsize=7, loc="upper left", ncol=1); axes[1].set_ylim(0, 1.3)
save("f_prompt_abstain_ladder")

# ---------- probe gate vs tuning ----------
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for k, c, mk in zip(QK, [P["blue"], P["orange"], P["green"], P["mauve"]], ["o", "s", "^", "D"]):
    pg = ci[E3K[k]]["probe_gated"]; xs = sorted(float(a.split("_")[1]) for a in pg)
    ax.plot(xs, [pg[f"abstain_{a:.2f}"]["s"] for a in xs], marker=mk, color=c, label=LABEL[k])
    ax.fill_between(xs, [pg[f"abstain_{a:.2f}"]["ci95"][0] for a in xs], [pg[f"abstain_{a:.2f}"]["ci95"][1] for a in xs], color=c, alpha=0.12, lw=0)
    for arm, m in (("C_tuned_plain", "*"),):
        ax.plot(e3[E3K[k]]["arms"][arm]["abstain_rate"], ci[E3K[k]]["arms"][arm]["s"], marker=m, color=c, markersize=11, markeredgecolor=P["dark"], markeredgewidth=0.5, linestyle="none")
ax.plot([], [], marker="*", color="#bbbbbb", markersize=10, linestyle="none", label="LoRA tuned (arm C)")
ax.axhline(0, color="#888888", lw=0.6); ax.set_xlabel("Abstention rate"); ax.set_ylabel("Score s"); ax.set_ylim(-0.35, 0.5)
ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save("f_probe_gate")

# ---------- quantization ablation ----------
q = M["quant"]["qwen3-8b-4bit"]; keys = ["accuracy", "auroc_se", "auroc_logprob", "auroc_lexical"]; kl = ["Accuracy", "Semantic entropy", "Log-probability", "Lexical diversity", "Probe (final token)"]
fp = [q["fp16"][k] for k in keys] + [q["fp16"]["probe"]["last_layer_final"]]; qq = [q["quant"][k] for k in keys] + [q["quant"]["probe"]["last_layer_final"]]
fig, ax = plt.subplots(figsize=(5.4, 2.6)); x = np.arange(5); w = 0.36
ax.bar(x - w / 2, fp, w, color=P["blue"], label="16-bit"); ax.bar(x + w / 2, qq, w, color=P["orange"], label="4-bit")
for i in range(5): ax.text(i, max(fp[i], qq[i]) + 0.012, f"{qq[i] - fp[i]:+.3f}", ha="center", fontsize=7.5, color=P["dark"])
ax.set_xticks(x); ax.set_xticklabels(kl, rotation=20, ha="right"); ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy / AUROC, Qwen3 8B"); ax.legend()
save("f_quant_ablation")

# ---------- domain transfer ----------
v2 = M["e5"]["v2"]; fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
for ax, dom in zip(axes, ["medical", "legal"]):
    src = [r2[k]["probe_best"] for k in QK]
    ind = []; tr = []; se = []
    for k in QK:
        d = v2[E3K[k]]["transfer"]["domains"][dom]; pr = d["probe"]
        best = max(pr.values(), key=lambda v: v["transfer_auroc"]); ind.append(max(v["indomain_oof_auroc"] for v in pr.values())); tr.append(best["transfer_auroc"]); se.append(d["auroc_semantic_entropy"])
    ax.plot(SIZES, src, marker="o", color=P["grey"], linestyle="--", label="probe, TriviaQA (source)")
    ax.plot(SIZES, ind, marker="s", color=P["blue"], label="probe, trained in the domain")
    ax.plot(SIZES, tr, marker="^", color=P["orange"], label="probe, transferred")
    ax.plot(SIZES, se, marker="D", color=P["green"], label="semantic entropy")
    ax.axhline(0.5, color="#aaaaaa", lw=0.6, linestyle=":"); ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(["1.7B", "4B", "8B", "14B*"]); ax.minorticks_off()
    ax.set_title(f"{dom} (MMLU)", fontsize=9, color=P["dark"]); ax.set_ylim(0.4, 0.9)
axes[0].set_ylabel("AUROC"); h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.02))
save("f_transfer")

# ---------- simulation: levers at p=0.10, n=10 ----------
lev = SIM["experiments"]["p0.1"]["n10"]; names = ["none", "detector+verifier", "predicate", "abstention", "full-stack"]
nl = ["no control", "detector + verifier", "predicate check", "abstention gate", "full stack"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
sf = [lev[n]["none"]["silent_failure_rate_given_answered"] for n in names]; cov = [lev[n]["none"]["coverage"] for n in names]
axes[0].barh(range(5), sf, color=[P["grey"], P["blue"], P["green"], P["orange"], P["mauve"]], height=0.6)
for i, v in enumerate(sf): axes[0].text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.5, color=P["dark"])
axes[0].set_yticks(range(5)); axes[0].set_yticklabels(nl); axes[0].invert_yaxis(); axes[0].set_xlabel("Silent failure rate among answered runs"); axes[0].set_xlim(0, 0.42)
axes[1].barh(range(5), cov, color=[P["grey"], P["blue"], P["green"], P["orange"], P["mauve"]], height=0.6)
for i, v in enumerate(cov): axes[1].text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=7.5, color=P["dark"])
axes[1].set_yticks(range(5)); axes[1].set_yticklabels([]); axes[1].invert_yaxis(); axes[1].set_xlabel("Coverage (share of runs that answer)"); axes[1].set_xlim(0, 1.15)
save("f_sim_levers")

# ---------- simulation: where the error is injected ----------
lv = SIM["late_vs_early"]; ps = [0.02, 0.05, 0.1, 0.2, 0.3]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(ps, [lv[f"p{p}"]["inject_first"] for p in ps], marker="o", color=P["blue"], label="error injected at the first step")
ax.plot(ps, [lv[f"p{p}"]["inject_last"] for p in ps], marker="s", color=P["orange"], label="error injected at the last step")
ax.set_xlabel("Per-step error probability"); ax.set_ylabel("Probability the final answer is wrong"); ax.set_ylim(0, 1.08)
ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
save("f_sim_late_vs_early")
print("done")
