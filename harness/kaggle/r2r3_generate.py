"""R2/R3 data collection on Kaggle GPU: sampling + hidden states + semantic entropy.

One run = one model from the roster. Produces everything needed for:
  R2  Semantic entropy reproduction (Kuhn et al. 2023 / Farquhar et al. 2024)
      reference AUROC on short-form QA: ~0.75-0.83
  R3  Hidden-state probe training data (SEP-style)

Protocol per question (TriviaQA rc.nocontext, validation split):
  - 1 greedy answer (temperature 0) -> the "official" answer we evaluate
  - K stochastic samples (temperature 1.0, top_p 0.9) -> for entropy
  - sequence logprob of every generation
  - hidden states of the greedy answer (mean + last token, layers -1 and -2)
  - correctness label: alias match against TriviaQA gold answers

Outputs in OUT_DIR:
  generations.parquet   per-sample rows
  hidden_states.npz     float16 arrays [n_questions, hidden]
  semantic_entropy.parquet  per-question entropy + label
  run_meta.json         config + timing

Designed for Kaggle 2xT4. 1.7B-4B fp16 fit on one T4, 7-8B fp16 shard across
both, 14B needs 4-bit (LOAD_4BIT=1): 14B fp16 (~29.6 GB) exceeds 2x14.5 GiB, so
accelerate CPU-offloads lm_head, runs 2.5x slower and OOMs late (observed
2026-09-02, uhr-r2r3-qwen3-14b-fp16). fp16 14B needs an A100.
"""

import json
import os
import re
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------- config -----------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", 300))
K_SAMPLES = int(os.environ.get("K_SAMPLES", 10))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 48))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"
SEED = int(os.environ.get("SEED", 20260901))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/out"))
NLI_MODEL = os.environ.get("NLI_MODEL", "microsoft/deberta-large-mnli")

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- model ------------------------------------
print(f"loading {MODEL_ID} (4bit={LOAD_4BIT}) on {DEVICE}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
# device_map="auto" shards across both T4s when one 14.5GB card is not enough (8B fp16)
kwargs = {"torch_dtype": torch.float16, "device_map": "auto" if DEVICE == "cuda" else DEVICE}
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    kwargs = {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        ),
        "device_map": "auto",
    }
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
model.eval()


def build_prompt(question: str) -> torch.Tensor:
    msgs = [
        {
            "role": "user",
            "content": f"Answer the question as briefly as possible.\nQ: {question}\nA:",
        }
    ]
    extra = {}
    if "qwen3" in MODEL_ID.lower():
        extra["enable_thinking"] = False  # keep latency axis controlled
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, **extra
    )
    return tok(text, return_tensors="pt").input_ids.to(model.device)


@torch.no_grad()
def generate(input_ids, do_sample: bool, output_hidden: bool = False):
    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        temperature=1.0 if do_sample else None,
        top_p=0.9 if do_sample else None,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=output_hidden,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out.sequences[0, input_ids.shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    # mean token logprob of the generated sequence
    logprobs = []
    for step, score in enumerate(out.scores):
        lp = torch.log_softmax(score[0].float(), dim=-1)[gen_ids[step]].item()
        logprobs.append(lp)
        if gen_ids[step].item() == tok.eos_token_id:
            break
    mean_lp = float(np.mean(logprobs)) if logprobs else 0.0
    hidden = None
    if output_hidden and out.hidden_states:
        # per generated token: hidden_states[t][layer][0, -1, :]
        h_last, h_penult = [], []
        for t in range(len(logprobs)):
            h_last.append(out.hidden_states[t][-1][0, -1, :].float().cpu())
            h_penult.append(out.hidden_states[t][-2][0, -1, :].float().cpu())
        hidden = {
            "last_layer_mean": torch.stack(h_last).mean(0).numpy().astype(np.float16),
            "last_layer_final": h_last[-1].numpy().astype(np.float16),
            "penult_layer_mean": torch.stack(h_penult).mean(0).numpy().astype(np.float16),
        }
    return text, mean_lp, hidden


# --------------------------- correctness --------------------------------
def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred: str, aliases: list[str]) -> bool:
    p = normalize(pred)
    return any(normalize(a) in p or p in normalize(a) for a in aliases if a.strip())


# ----------------------------- data -------------------------------------
print("loading TriviaQA validation")
ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
ds = ds.shuffle(seed=SEED).select(range(N_QUESTIONS))

rows, hs_mean, hs_final, hs_penult, labels = [], [], [], [], []
t0 = time.time()
for qi, ex in enumerate(ds):
    q = ex["question"]
    aliases = list(ex["answer"]["aliases"]) + [ex["answer"]["value"]]
    input_ids = build_prompt(q)

    greedy_text, greedy_lp, hidden = generate(input_ids, do_sample=False, output_hidden=True)
    correct = is_correct(greedy_text, aliases)
    rows.append({"q_idx": qi, "question": q, "sample_idx": -1, "text": greedy_text,
                 "mean_logprob": greedy_lp, "greedy": True, "correct": correct,
                 "gold": ex["answer"]["value"]})
    hs_mean.append(hidden["last_layer_mean"])
    hs_final.append(hidden["last_layer_final"])
    hs_penult.append(hidden["penult_layer_mean"])
    labels.append(int(correct))

    for k in range(K_SAMPLES):
        torch.manual_seed(SEED + qi * 1000 + k)
        text, lp, _ = generate(input_ids, do_sample=True)
        rows.append({"q_idx": qi, "question": q, "sample_idx": k, "text": text,
                     "mean_logprob": lp, "greedy": False,
                     "correct": is_correct(text, aliases), "gold": ex["answer"]["value"]})
    if (qi + 1) % 20 == 0:
        el = time.time() - t0
        print(f"{qi+1}/{N_QUESTIONS} questions | {el:.0f}s | acc so far {np.mean(labels):.3f}")
        # checkpoint: a crash late in the run must not lose the generations
        pd.DataFrame(rows).to_parquet(OUT_DIR / "generations.partial.parquet")
        np.savez_compressed(
            OUT_DIR / "hidden_states.partial.npz",
            last_layer_mean=np.stack(hs_mean), last_layer_final=np.stack(hs_final),
            penult_layer_mean=np.stack(hs_penult), label_correct=np.array(labels),
        )
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_parquet(OUT_DIR / "generations.parquet")
np.savez_compressed(
    OUT_DIR / "hidden_states.npz",
    last_layer_mean=np.stack(hs_mean),
    last_layer_final=np.stack(hs_final),
    penult_layer_mean=np.stack(hs_penult),
    label_correct=np.array(labels),
)
print(f"generation done: {len(df)} rows, greedy accuracy {np.mean(labels):.3f}")
for _p in ("generations.partial.parquet", "hidden_states.partial.npz"):
    (OUT_DIR / _p).unlink(missing_ok=True)

# ---------------------- semantic entropy (R2) ---------------------------
# Bidirectional-entailment clustering of the K samples per question, then
# entropy over cluster mass (discrete SE, Farquhar et al. 2024).
del model  # free LM VRAM before loading the NLI model (14B fp16 fills both T4s)
torch.cuda.empty_cache()
print(f"loading NLI model {NLI_MODEL}")
from transformers import AutoModelForSequenceClassification

nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
# fp32: DeBERTa's disentangled attention errors out under fp16
nli = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL, torch_dtype=torch.float32, device_map=DEVICE
).eval()
ENTAIL_IDX = 2  # deberta-mnli: 0=contradiction 1=neutral 2=entailment


@torch.no_grad()
def entails(question: str, a: str, b: str) -> bool:
    def one_way(x, y):
        inp = nli_tok(f"{question} {x}", f"{question} {y}", return_tensors="pt",
                      truncation=True, max_length=256).to(nli.device)
        return nli(**inp).logits[0].argmax().item() == ENTAIL_IDX
    return one_way(a, b) and one_way(b, a)


se_rows = []
for qi in range(N_QUESTIONS):
    g = df[(df.q_idx == qi) & (~df.greedy)]
    answers = g.text.tolist()
    lps = g.mean_logprob.tolist()
    question = g.question.iloc[0]
    clusters: list[list[int]] = []
    for i, a in enumerate(answers):
        placed = False
        for cl in clusters:
            if entails(question, a, answers[cl[0]]):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    # discrete SE: cluster probability = fraction of samples in cluster
    ps = np.array([len(cl) for cl in clusters], dtype=float) / len(answers)
    se = float(-(ps * np.log(ps)).sum())
    # naive baselines for comparison
    lex_unique = len(set(normalize(a) for a in answers)) / len(answers)
    se_rows.append({"q_idx": qi, "semantic_entropy": se, "n_clusters": len(clusters),
                    "lexical_diversity": lex_unique,
                    "greedy_mean_logprob": float(df[(df.q_idx == qi) & df.greedy].mean_logprob.iloc[0]),
                    "correct": bool(labels[qi])})
    if (qi + 1) % 50 == 0:
        print(f"SE {qi+1}/{N_QUESTIONS}")

se_df = pd.DataFrame(se_rows)
se_df.to_parquet(OUT_DIR / "semantic_entropy.parquet")

from sklearn.metrics import roc_auc_score

y = 1 - se_df.correct.astype(int)  # 1 = hallucination/incorrect
metrics = {
    "model": MODEL_ID,
    "load_4bit": LOAD_4BIT,
    "n_questions": N_QUESTIONS,
    "k_samples": K_SAMPLES,
    "greedy_accuracy": float(se_df.correct.mean()),
    "auroc_semantic_entropy": float(roc_auc_score(y, se_df.semantic_entropy)),
    "auroc_neg_logprob": float(roc_auc_score(y, -se_df.greedy_mean_logprob)),
    "auroc_lexical_diversity": float(roc_auc_score(y, se_df.lexical_diversity)),
    "wall_seconds": round(time.time() - t0, 1),
    "seed": SEED,
}
(OUT_DIR / "run_meta.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print("DONE - download the out/ directory as the run artifact")
