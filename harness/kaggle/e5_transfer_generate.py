"""E5 cross-domain transfer (RQ5): same R2/R3 protocol on domain-shifted QA.

Domains (uniform 4-option multiple-choice, answer = letter, from MMLU [cais/mmlu]):
  medical : professional_medicine + clinical_knowledge (test splits)
  legal   : professional_law (test split)
For each domain, N_PER_DOMAIN questions are sampled with SEED. Per question:
  1 greedy answer (+ hidden states of the answer tokens), K stochastic samples,
  mean logprob of every generation, correctness = first A-D letter == gold.
Outputs (per domain, OUT_DIR/<domain>/): generations.parquet, hidden_states.npz,
semantic_entropy.parquet (SE over letter-answer clusters; exact-match clustering
is the correct NLI here), run_meta.json — identical schema to r2r3 runs so the
local probe trainer and transfer evaluator can consume them unchanged.
"""
import json, os, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
N_PER_DOMAIN = int(os.environ.get("N_PER_DOMAIN", 300))
K_SAMPLES = int(os.environ.get("K_SAMPLES", 10))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 8))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"
SEED = int(os.environ.get("SEED", 20260901))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/out"))
# Optional assistant-turn prefill so the letter is the first generated token
# (small models otherwise explain first and get truncated; e.g. Qwen3-1.7B).
# v2 protocol (2026-09-04): the v1 user prompt ended in "A:", which collided with
# option A and was echoed by the models ("A: **C**"), corrupting labels. v2 ends the
# user turn after the options and prefills the assistant turn with "Answer: " for
# every model, so the letter is the first generated token.
ANSWER_PREFIX = os.environ.get("ANSWER_PREFIX", "Answer: ")
PROMPT_VERSION = 2
DOMAINS = {"medical": ["professional_medicine", "clinical_knowledge"],
           "legal": ["professional_law"]}
DOMAINS = {k: v for k, v in DOMAINS.items()
           if k in os.environ.get("DOMAINS", "medical,legal").split(",")}
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cuda"

print(f"loading {MODEL_ID} (4bit={LOAD_4BIT})")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
              "device_map": "auto"}
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).eval()
LETTERS = "ABCD"


def build_prompt(q, choices):
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
    content = (f"Answer the multiple-choice question with the letter of the correct option only.\n"
               f"Question: {q}\n{opts}")
    extra = {"enable_thinking": False} if "qwen3" in MODEL_ID.lower() else {}
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, **extra) + ANSWER_PREFIX
    return tok(text, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(model.device)


@torch.no_grad()
def generate(input_ids, do_sample, output_hidden=False):
    out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample,
                         temperature=1.0 if do_sample else None, top_p=0.9 if do_sample else None,
                         return_dict_in_generate=True, output_scores=True,
                         output_hidden_states=output_hidden, pad_token_id=tok.eos_token_id)
    gen_ids = out.sequences[0, input_ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    logprobs = []
    for step, score in enumerate(out.scores):
        logprobs.append(torch.log_softmax(score[0].float(), dim=-1)[gen_ids[step]].item())
        if gen_ids[step].item() == tok.eos_token_id:
            break
    mean_lp = float(np.mean(logprobs)) if logprobs else 0.0
    hidden = None
    if output_hidden and out.hidden_states:
        h_last = [out.hidden_states[t][-1][0, -1, :].float().cpu() for t in range(len(logprobs))]
        h_pen = [out.hidden_states[t][-2][0, -1, :].float().cpu() for t in range(len(logprobs))]
        hidden = {"last_layer_mean": torch.stack(h_last).mean(0).numpy().astype(np.float16),
                  "last_layer_final": h_last[-1].numpy().astype(np.float16),
                  "penult_layer_mean": torch.stack(h_pen).mean(0).numpy().astype(np.float16)}
    return text, mean_lp, hidden


def parse_letter(text):
    u = re.sub(r"^(?:A\s*:\s*)+", "", text.strip().upper())  # strip echoed 'A:' prompt suffix
    m = (re.search(r"(?:ANSWER|OPTION|DIAGNOSIS)\s*(?:IS|:)?\s*\**\s*\(?([ABCD])\b", u)
         or re.search(r"\*\*([ABCD])[.)]", u) or re.search(r"\b([ABCD])\b", u[:20]))
    return m.group(1) if m else ""


t0 = time.time()
summary = {}
for domain, subsets in DOMAINS.items():
    ds = concatenate_datasets([load_dataset("cais/mmlu", s, split="test") for s in subsets])
    ds = ds.shuffle(seed=SEED).select(range(min(N_PER_DOMAIN, len(ds))))
    od = OUT_DIR / domain; od.mkdir(exist_ok=True)
    rows, hs_mean, hs_final, hs_pen, labels, se_rows = [], [], [], [], [], []
    for qi, ex in enumerate(ds):
        gold = LETTERS[ex["answer"]]
        ids = build_prompt(ex["question"], ex["choices"])
        g_text, g_lp, hidden = generate(ids, False, True)
        correct = parse_letter(g_text) == gold
        rows.append({"q_idx": qi, "question": ex["question"], "sample_idx": -1, "text": g_text,
                     "mean_logprob": g_lp, "greedy": True, "correct": correct, "gold": gold,
                     "subject": ex.get("subject", "")})
        hs_mean.append(hidden["last_layer_mean"]); hs_final.append(hidden["last_layer_final"])
        hs_pen.append(hidden["penult_layer_mean"]); labels.append(int(correct))
        samp = []
        for k in range(K_SAMPLES):
            torch.manual_seed(SEED + qi * 1000 + k)
            text, lp, _ = generate(ids, True)
            samp.append(parse_letter(text) or text[:10])
            rows.append({"q_idx": qi, "question": ex["question"], "sample_idx": k, "text": text,
                         "mean_logprob": lp, "greedy": False, "correct": parse_letter(text) == gold,
                         "gold": gold, "subject": ex.get("subject", "")})
        # SE over letter clusters (exact match is the right equivalence for MC)
        _, counts = np.unique(samp, return_counts=True)
        ps = counts / counts.sum()
        se_rows.append({"q_idx": qi, "semantic_entropy": float(-(ps * np.log(ps)).sum()),
                        "n_clusters": int(len(counts)), "lexical_diversity": float(len(counts) / len(samp)),
                        "greedy_mean_logprob": g_lp, "correct": bool(correct)})
        if (qi + 1) % 25 == 0:
            print(f"[{domain}] {qi+1}/{len(ds)} | {time.time()-t0:.0f}s | acc {np.mean(labels):.3f}")
            pd.DataFrame(rows).to_parquet(od / "generations.partial.parquet")
    df = pd.DataFrame(rows); df.to_parquet(od / "generations.parquet")
    (od / "generations.partial.parquet").unlink(missing_ok=True)
    np.savez_compressed(od / "hidden_states.npz", last_layer_mean=np.stack(hs_mean),
                        last_layer_final=np.stack(hs_final), penult_layer_mean=np.stack(hs_pen),
                        label_correct=np.array(labels))
    se_df = pd.DataFrame(se_rows); se_df.to_parquet(od / "semantic_entropy.parquet")
    from sklearn.metrics import roc_auc_score
    y = 1 - se_df.correct.astype(int)
    m = {"model": MODEL_ID, "domain": domain, "subsets": subsets, "n_questions": int(len(ds)),
         "k_samples": K_SAMPLES, "load_4bit": LOAD_4BIT, "answer_prefix": ANSWER_PREFIX, "prompt_version": PROMPT_VERSION, "max_new_tokens": MAX_NEW_TOKENS, "greedy_accuracy": float(se_df.correct.mean()),
         "auroc_semantic_entropy": float(roc_auc_score(y, se_df.semantic_entropy)) if 0 < y.sum() < len(y) else None,
         "auroc_neg_logprob": float(roc_auc_score(y, -se_df.greedy_mean_logprob)) if 0 < y.sum() < len(y) else None,
         "auroc_lexical_diversity": float(roc_auc_score(y, se_df.lexical_diversity)) if 0 < y.sum() < len(y) else None,
         "unparsed_greedy_rate": float((df[df.greedy].text.map(parse_letter) == "").mean()),
         "wall_seconds": round(time.time() - t0, 1), "seed": SEED}
    (od / "run_meta.json").write_text(json.dumps(m, indent=2)); summary[domain] = m
    print(json.dumps(m, indent=2))
(OUT_DIR / "run_meta.json").write_text(json.dumps({"model": MODEL_ID, "domains": summary,
                                                   "wall_seconds": round(time.time() - t0, 1)}, indent=2))
print("DONE")
