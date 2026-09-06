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


DEFAULTS = {  # identical to harness/configs/e6_sim.yaml; used if the file is absent
    "seed": 20260901, "runs": 20000,
    "manifest": "thesis/v3/data/manifest_v2.json", "out": "harness/results/e6_sim/propagation.json",
    "calibration_models": ["qwen3-1p7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"], "detector_model": "qwen3-4b",
    "step_error_grid": [0.02, 0.05, 0.10, 0.20, 0.30], "rho": 0.8,
    "levers": {"none": dict(fpr=0, verifier_fix=0, predicate_catch=0, abstain_fpr=0),
               "detector+verifier": dict(fpr=0.10, verifier_fix=0.8, predicate_catch=0, abstain_fpr=0),
               "predicate": dict(fpr=0, verifier_fix=0, predicate_catch=0.5, abstain_fpr=0),
               "abstention": dict(fpr=0, verifier_fix=0, predicate_catch=0, abstain_fpr=0.05),
               "full-stack": dict(fpr=0.10, verifier_fix=0.8, predicate_catch=0.5, abstain_fpr=0.05)},
    "propagation_curves": {"pipeline_lengths": [5, 10, 15], "runs_divisor": 4},
    "measured_anchor": {"pipeline_lengths": [1, 3, 5, 10], "runs_divisor": 4},
    "sweeps": {"runs_divisor": 2, "p": 0.10, "n_steps": 10, "inject_at": 2,
               "rho": [0.0, 0.2, 0.5, 0.8, 0.9, 1.0],
               "auroc": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
               "gate_fpr": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
               "verifier_fix": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]},
    "late_vs_early": {"n_steps": 10, "runs_divisor": 2},
}
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "harness" / "configs" / "e6_sim.yaml"


def load_config(path: Path) -> dict:
    """Config file wins over DEFAULTS, key by key (one level deep)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if not path.exists():
        print(f"[e6-sim] {path} not found; using built-in defaults")
        return cfg
    import yaml
    user = yaml.safe_load(path.read_text()) or {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG), help="simulation configuration (YAML)")
    ap.add_argument("--manifest", help="numbers manifest with the measured calibration points")
    ap.add_argument("--out", help="where to write propagation.json")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    for k in ("manifest", "out", "runs", "seed"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    man = json.loads(resolve(cfg["manifest"]).read_text()); rng = np.random.default_rng(cfg["seed"])
    # calibration points from measured data (Qwen3 ladder, TriviaQA)
    cal = {}
    for k in cfg["calibration_models"]:
        e = man["r2r3"][k]; cal[k] = {"p": 1 - e["accuracy"], "auroc_best": e["probe_best"], "auroc_se": e["auroc_se"]}
    out = {"note": "SIMULATION calibrated on measured accuracy/AUROC; not an empirical agent run",
           "config": cfg, "calibration": cal,
           "step_error_grid": cfg["step_error_grid"], "experiments": {}}
    A4 = cal[cfg["detector_model"]]["auroc_best"]
    LEVERS = cfg["levers"]
    R = cfg["runs"]
    rho = cfg["rho"]; sw = cfg["sweeps"]
    # 1. propagation curves: P(wrong final | inject at s), n from the config, p in grid, detector = detector_model probe
    pc = cfg["propagation_curves"]
    for p in cfg["step_error_grid"]:
        for n in pc["pipeline_lengths"]:
            for lever, lcfg in LEVERS.items():
                curve = {("none" if s is None else s): simulate(n, p, rho, A4, inject_at=s, n_runs=R // pc["runs_divisor"], rng=rng, **lcfg)
                         for s in list(range(n)) + [None]}
                out["experiments"].setdefault(f"p{p}", {}).setdefault(f"n{n}", {})[lever] = curve
    # 2. measured-anchor pipelines: per-step error = measured 1-acc on TriviaQA (worst case: every step is an open-domain fact lookup)
    ma = cfg["measured_anchor"]
    out["measured_anchor"] = {}
    for k in cfg["calibration_models"]:
        c = cal[k]
        out["measured_anchor"][k] = {n: {lever: simulate(n, c["p"], rho, c["auroc_best"], inject_at=None, n_runs=R // ma["runs_divisor"], rng=rng, **lcfg)
                                         for lever, lcfg in LEVERS.items()} for n in ma["pipeline_lengths"]}
    # 3. one-factor sweeps at the configured operating point
    SR = R // sw["runs_divisor"]; sp, sn, si = sw["p"], sw["n_steps"], sw["inject_at"]
    out["rho_sweep"] = {str(r): simulate(sn, sp, r, A4, 0, 0, 0, 0, si, SR, rng) for r in sw["rho"]}
    out["auroc_sweep"] = {str(A): simulate(sn, sp, rho, A, 0.10, 0.8, 0, 0, si, SR, rng) for A in sw["auroc"]}
    out["gate_fpr_sweep"] = {str(g): simulate(sn, sp, rho, A4, 0, 0, 0, g, None, SR, rng) for g in sw["gate_fpr"]}
    out["verifier_fix_sweep"] = {str(v): simulate(sn, sp, rho, A4, 0.10, v, 0, 0, si, SR, rng) for v in sw["verifier_fix"]}
    # 4. early vs late injection in a pipeline of the configured length
    lv = cfg["late_vs_early"]; ln = lv["n_steps"]; LR = R // lv["runs_divisor"]
    out["late_vs_early"] = {f"p{p}": {"inject_first": simulate(ln, p, rho, A4, 0, 0, 0, 0, 0, LR, rng)["p_wrong_final"],
                                      "inject_last": simulate(ln, p, rho, A4, 0, 0, 0, 0, ln - 1, LR, rng)["p_wrong_final"]}
                            for p in cfg["step_error_grid"]}
    dest = resolve(cfg["out"])
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(out, indent=1))
    key_p, key_n = f"p{sp}", f"n{sn}"
    if key_p in out["experiments"] and key_n in out["experiments"][key_p]:
        for lever, curve in out["experiments"][key_p][key_n].items():
            print(f"{lever:20s} p={sp} n={sn}: P(wrong|inject@0)={curve[0]['p_wrong_final']:.3f} @{sn//2}={curve[sn//2]['p_wrong_final']:.3f} "
                  f"@{sn-1}={curve[sn-1]['p_wrong_final']:.3f} | no-inject wrong={curve['none']['p_wrong_final']:.3f} "
                  f"abstain={curve['none']['p_abstain']:.3f} blast={curve[0]['mean_blast_radius']:.2f} verifier/run={curve['none']['verifier_calls_per_run']:.2f}")
    print("late vs early:", out["late_vs_early"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
