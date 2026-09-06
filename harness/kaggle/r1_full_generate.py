"""R1 full-scale: SelfCheckGPT-Prompt on WikiBio with a local constrained judge.

All 238 annotated passages, all 20 samples per sentence, judge = local small
model with batched greedy decoding. Reference (Manakul et al. 2023, GPT-3.5
judge): AUC-PR nonfactual 93.42, factual 67.09.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_ID = os.environ.get("JUDGE_ID", "Qwen/Qwen3-4B")
N_PASSAGES = int(os.environ.get("N_PASSAGES", 238))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 20))
BATCH = int(os.environ.get("BATCH", 48))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}
PROMPT = ("Context: {context}\n\nSentence: {sentence}\n\n"
          "Is the sentence supported by the context above? Answer Yes or No.\n\nAnswer:")

print(f"loading judge {JUDGE_ID}")
tok = AutoTokenizer.from_pretrained(JUDGE_ID, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")

tasks = []  # (passage, sent_idx, prompt)
meta = []   # (passage, sent_idx, label, raw_label)
for pi, row in enumerate(ds):
    if pi >= N_PASSAGES:
        break
    samples = row["gpt3_text_samples"][:N_SAMPLES]
    for si, (sent, ann) in enumerate(zip(row["gpt3_sentences"], row["annotation"])):
        meta.append((pi, si, LABEL_MAP[ann], ann))
        for sample in samples:
            tasks.append((pi, si, PROMPT.format(context=sample.strip(), sentence=sent.strip())))
print(f"{len(meta)} sentences, {len(tasks)} judge calls")


def to_chat(p):
    extra = {"enable_thinking": False} if "qwen3" in JUDGE_ID.lower() else {}
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True, **extra)


scores_per_sent = {}
t0 = time.time()
# sort by length for efficient batching, remember original order
order = sorted(range(len(tasks)), key=lambda i: len(tasks[i][2]))
results = [0.5] * len(tasks)
with torch.no_grad():
    for b in range(0, len(order), BATCH):
        idxs = order[b:b + BATCH]
        prompts = [to_chat(tasks[i][2]) for i in idxs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            text = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            t = text.strip().lower()
            results[i] = 0.0 if t.startswith("yes") else 1.0 if t.startswith("no") else 0.5
        if (b // BATCH) % 50 == 0:
            done = b + len(idxs)
            el = time.time() - t0
            print(f"{done}/{len(tasks)} calls | {el:.0f}s | eta {el/max(done,1)*(len(tasks)-done):.0f}s")

for (pi, si, _), score in zip(tasks, results):
    scores_per_sent.setdefault((pi, si), []).append(score)

rows = []
for (pi, si, label, raw) in meta:
    per = scores_per_sent[(pi, si)]
    rows.append({"passage": pi, "sent_idx": si, "label": label, "raw_label": raw,
                 "score": float(np.mean(per)), "n_samples": len(per)})
df = pd.DataFrame(rows)
df.to_parquet(OUT_DIR / "sentence_scores.parquet")

y = df.label.values
s = df.score.values
metrics = {
    "judge": JUDGE_ID, "n_passages": N_PASSAGES, "n_samples": N_SAMPLES,
    "n_sentences": len(df), "prevalence_nonfactual": float(y.mean()),
    "auc_pr_nonfactual": float(average_precision_score(y, s)),
    "auc_pr_factual": float(average_precision_score(1 - y, 1 - s)),
    "auroc": float(roc_auc_score(y, s)),
    "reference_auc_pr_nonfactual_gpt35": 93.42,
    "wall_seconds": round(time.time() - t0, 1),
}
(OUT_DIR / "run_meta.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
