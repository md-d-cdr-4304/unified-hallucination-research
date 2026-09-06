"""E6-sim: hallucination propagation in multi-step agentic pipelines (SIMULATION).

This is a Monte-Carlo model, NOT an empirical agent run. Its parameters are
calibrated on measured Phase-1/2 quantities from the manifest:
  - per-step hallucination probability  p  = 1 - greedy accuracy of the model
  - detector quality: AUROC A  -> TPR at a chosen FPR under the binormal model
  - abstention: the probe-gated coverage/wrong-rate trade-off (measured)
Model of an n-step pipeline (plan -> retrieve -> extract -> synthesize ...):
  * At every step an unforced hallucination occurs w.p. p_s (p_s = p by default).
  * A fault may be INJECTED at step s (controlled fault injection, E6 design).
  * A hallucination at step t is *consumed* by each later step; with probability
    rho (dependence) a downstream step built on a corrupted input is itself wrong
    (propagation); otherwise the step "recovers" (e.g. re-retrieves).
  * Containment levers, applied at every step:
      L0 abstention: the step abstains (defers) when its risk score exceeds a
         threshold -> the pipeline halts with an explicit "cannot answer" (safe).
      L1 detector: flags a hallucinated step with prob TPR, false-alarms with FPR;
         a flag triggers one verifier retry that fixes the step w.p. v.
      L3 provenance predicate: deterministic check catching a fraction pi of
         hallucinations that violate a checkable constraint (e.g. citation exists).
  * Outcome per run: final answer correct / wrong (silent failure) / abstained.
Metrics: P(wrong final | injection step s, n), silent-failure rate, coverage,
blast radius (number of downstream steps contaminated), cost (verifier calls).
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import norm


def tpr_at_fpr(auroc: float, fpr: float) -> float:
    """Binormal ROC with equal variances: d' = sqrt(2)*Phi^-1(AUROC)."""
    if auroc <= 0.5: return fpr
    d = np.sqrt(2) * norm.ppf(auroc)
    return float(norm.cdf(d + norm.ppf(fpr)))


def simulate(n_steps, p, rho, auroc, fpr, verifier_fix, predicate_catch, abstain_fpr, inject_at, n_runs, rng):
    """One configuration. Returns outcome rates.
    p            per-step unforced hallucination probability
    rho          propagation probability: a step built on a corrupted state stays
                 corrupted w.p. rho, repairs it (re-derives from source) w.p. 1-rho
    auroc        detector AUROC (binormal ROC -> TPR at the chosen FPR)
    fpr          detector false-alarm rate per step (0 = detector off); flag -> verifier retry
    verifier_fix probability a verifier retry fixes a hallucinated step
    predicate_catch  fraction of hallucinations violating a checkable constraint (caught deterministically)
    abstain_fpr  per-step abstention-gate false-alarm rate (0 = no gate); gate TPR from AUROC
    inject_at    step index of a forced fault (None = no injection)
    """
    tpr = tpr_at_fpr(auroc, fpr) if fpr > 0 else 0.0
    tpr_gate = tpr_at_fpr(auroc, abstain_fpr) if abstain_fpr > 0 else 0.0
    wrong = abst = correct = 0; blast = []; verifier_calls = 0
    for _ in range(n_runs):
        corrupted = False; contaminated = 0; halted = False
        for t in range(n_steps):
            if inject_at is not None and t == inject_at:
                h = True
            else:
                h = rng.random() < p
            if corrupted and not h:
                # downstream of a corrupted state: propagate or repair
                if rng.random() < rho: contaminated += 1
                else: corrupted = False
            if h and predicate_catch > 0 and rng.random() < predicate_catch:
                verifier_calls += 1; h = False          # L3: constraint violation caught, regenerated
            if abstain_fpr > 0:
                risky = h or corrupted
                if (risky and rng.random() < tpr_gate) or ((not risky) and rng.random() < abstain_fpr):
                    halted = True; break                # L0: defer
            if fpr > 0:
                flagged = (h and rng.random() < tpr) or ((not h) and rng.random() < fpr)
                if flagged:
                    verifier_calls += 1
                    if h and rng.random() < verifier_fix: h = False   # L1/L2: verifier repair
            if h:
                corrupted = True; contaminated += 1
        if halted: abst += 1
        elif corrupted: wrong += 1; blast.append(contaminated)
        else: correct += 1
    n = n_runs
    return {"p_wrong_final": wrong / n, "p_abstain": abst / n, "p_correct": correct / n,
            "coverage": 1 - abst / n, "silent_failure_rate_given_answered": wrong / max(1, wrong + correct),
            "mean_blast_radius": float(np.mean(blast)) if blast else 0.0,
            "verifier_calls_per_run": verifier_calls / n, "tpr_detector": tpr, "tpr_gate": tpr_gate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parents[3] / "thesis/v2/data/manifest.json"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results/e6_sim/propagation.json"))
    ap.add_argument("--runs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text()); rng = np.random.default_rng(a.seed)
    # calibration points from measured data (Qwen3 ladder, TriviaQA)
    cal = {}
    for k in ("qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"):
        e = man["r2r3"][k]; cal[k] = {"p": 1 - e["accuracy"], "auroc_best": e["probe_best"], "auroc_se": e["auroc_se"]}
    out = {"note": "SIMULATION calibrated on measured accuracy/AUROC; not an empirical agent run", "calibration": cal,
           "step_error_grid": [0.02, 0.05, 0.10, 0.20, 0.30], "experiments": {}}
    A4 = cal["qwen3-4b"]["auroc_best"]; A8 = cal["qwen3-8b"]["auroc_best"]
    LEVERS = {"none": dict(fpr=0, verifier_fix=0, predicate_catch=0, abstain_fpr=0),
              "detector+verifier": dict(fpr=0.10, verifier_fix=0.8, predicate_catch=0, abstain_fpr=0),
              "predicate": dict(fpr=0, verifier_fix=0, predicate_catch=0.5, abstain_fpr=0),
              "abstention": dict(fpr=0, verifier_fix=0, predicate_catch=0, abstain_fpr=0.05),
              "full-stack": dict(fpr=0.10, verifier_fix=0.8, predicate_catch=0.5, abstain_fpr=0.05)}
    R = a.runs
    # 1. propagation curves: P(wrong final | inject at s), n in {5,10,15}, p in grid, rho=0.8, detector=4B probe
    for p in out["step_error_grid"]:
        for n in (5, 10, 15):
            for lever, cfg in LEVERS.items():
                curve = {("none" if s is None else s): simulate(n, p, 0.8, A4, inject_at=s, n_runs=R // 4, rng=rng, **cfg)
                         for s in list(range(n)) + [None]}
                out["experiments"].setdefault(f"p{p}", {}).setdefault(f"n{n}", {})[lever] = curve
    # 2. measured-anchor pipelines: per-step error = measured 1-acc on TriviaQA (worst case: every step is an open-domain fact lookup)
    out["measured_anchor"] = {}
    for k in ("qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"):
        c = cal[k]
        out["measured_anchor"][k] = {n: {lever: simulate(n, c["p"], 0.8, c["auroc_best"], inject_at=None, n_runs=R // 4, rng=rng, **cfg)
                                          for lever, cfg in LEVERS.items()} for n in (1, 3, 5, 10)}
    # 3. sweeps at p=0.10, n=10, inject at step 2
    out["rho_sweep"] = {str(r): simulate(10, 0.10, r, A4, 0, 0, 0, 0, 2, R // 2, rng) for r in (0.0, 0.2, 0.5, 0.8, 0.9, 1.0)}
    out["auroc_sweep"] = {str(A): simulate(10, 0.10, 0.8, A, 0.10, 0.8, 0, 0, 2, R // 2, rng) for A in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)}
    out["gate_fpr_sweep"] = {str(g): simulate(10, 0.10, 0.8, A4, 0, 0, 0, g, None, R // 2, rng) for g in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)}
    out["verifier_fix_sweep"] = {str(v): simulate(10, 0.10, 0.8, A4, 0.10, v, 0, 0, 2, R // 2, rng) for v in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}
    out["late_vs_early"] = {f"p{p}": {"inject_first": simulate(10, p, 0.8, A4, 0, 0, 0, 0, 0, R // 2, rng)["p_wrong_final"],
                                      "inject_last": simulate(10, p, 0.8, A4, 0, 0, 0, 0, 9, R // 2, rng)["p_wrong_final"]} for p in out["step_error_grid"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    for lever, curve in out["experiments"]["p0.1"]["n10"].items():
        print(f"{lever:20s} p=0.10 n=10: P(wrong|inject@0)={curve[0]['p_wrong_final']:.3f} @5={curve[5]['p_wrong_final']:.3f} @9={curve[9]['p_wrong_final']:.3f} | no-inject wrong={curve['none']['p_wrong_final']:.3f} abstain={curve['none']['p_abstain']:.3f} blast={curve[0]['mean_blast_radius']:.2f} verifier/run={curve['none']['verifier_calls_per_run']:.2f}")
    print("late vs early:", out["late_vs_early"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
