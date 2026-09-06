"""E3: abstention incentives (RQ3) — R-Tuning-style LoRA vs prompting-only.

Pipeline (one model per run):
  1. Data generation: greedy-answer N_TRAIN TriviaQA *train* questions with the
     base model; label correct/incorrect by alias match (R-Tuning known/unknown split).
  2. SFT pairs: correct -> the model's own (correct) answer;
     incorrect -> "I don't know." LoRA fine-tune on completion-only loss.
  3. Evaluation on the SAME fixed 300-question validation set as the R2/R3 sweep
     (shuffle seed 20260901), three arms:
       A base model, plain prompt        (sweep baseline)
       B base model, abstention prompt   (Layer 0: prompting only)
       C tuned model, plain prompt       (does tuning alone induce abstention?)
       D tuned model, abstention prompt
     Metrics per arm: answer rate, accuracy on attempted, wrong rate, abstain rate,
     abstention-aware score s = (#correct - #wrong)/N  (guessing is penalized).
"""

import json
import os
import re
import string
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
N_TRAIN = int(os.environ.get("N_TRAIN", 1500))
N_EVAL = int(os.environ.get("N_EVAL", 300))
EPOCHS = int(os.environ.get("EPOCHS", 2))
BATCH_GEN = int(os.environ.get("BATCH_GEN", 32))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"      # QLoRA for 14B (fp16 14B does not fit 2xT4)
DEVICE_MAP = os.environ.get("DEVICE_MAP", "cuda")        # "auto" shards 8B fp16 across both T4s
SEED = 20260901
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)

IDK = "I don't know."
PLAIN = "Answer the question as briefly as possible.\nQ: {q}\nA:"
ABSTAIN = ("Answer the question as briefly as possible. "
           "If you are not sure of the answer, say exactly: I don't know.\nQ: {q}\nA:")

tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    # NOTE: peft.prepare_model_for_kbit_training is deliberately NOT used: it casts
    # the un-quantized embed_tokens/lm_head to fp32 (~3 GB each for Qwen3-14B's
    # 152k vocab) and OOMs a T4 (observed 2026-09-04). Gradient checkpointing and
    # input grads are enabled explicitly below instead.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map=DEVICE_MAP,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE_MAP
    )
model = model.eval()
print(f"loaded {MODEL_ID} 4bit={LOAD_4BIT} device_map={DEVICE_MAP} "
      f"hf_device_map={getattr(model, 'hf_device_map', None)}")


def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def is_correct(pred, aliases):
    p = normalize(pred)
    return any(normalize(a) in p or p in normalize(a) for a in aliases if a.strip())


def is_abstain(pred):
    p = pred.lower()
    return "don't know" in p or "do not know" in p or "not sure" in p


def to_chat(content):
    extra = {"enable_thinking": False} if "qwen3" in MODEL_ID.lower() else {}
    return tok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True, **extra)


@torch.no_grad()
def batch_greedy(questions, template):
    answers = []
    for b in range(0, len(questions), BATCH_GEN):
        chunk = [to_chat(template.format(q=q)) for q in questions[b:b + BATCH_GEN]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            answers.append(tok.decode(out[j, enc.input_ids.shape[1]:],
                                      skip_special_tokens=True).strip())
    return answers


# ---------- 1. known/unknown split on train questions ----------
t0 = time.time()
train_ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
train_ds = train_ds.shuffle(seed=SEED).select(range(N_TRAIN))
tq = [ex["question"] for ex in train_ds]
taliases = [list(ex["answer"]["aliases"]) + [ex["answer"]["value"]] for ex in train_ds]
print(f"generating base answers for {N_TRAIN} train questions")
tans = batch_greedy(tq, PLAIN)
tcorrect = [is_correct(a, al) for a, al in zip(tans, taliases)]
print(f"train accuracy {np.mean(tcorrect):.3f} | {time.time()-t0:.0f}s")

pairs = [(q, a if ok else IDK) for q, a, ok in zip(tq, tans, tcorrect)]

# ---------- 2. LoRA SFT, completion-only loss ----------
model = model.train()
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
# >=4B on a 14.5GB T4 needs activation checkpointing and a small micro-batch
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

examples = []
for q, target in pairs:
    prompt = to_chat(PLAIN.format(q=q))
    full = prompt + target + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids[:512]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    examples.append((f_ids, labels[:len(f_ids)]))

TB = int(os.environ.get("TRAIN_BATCH", 2))
steps = 0
for ep in range(EPOCHS):
    rng = np.random.default_rng(SEED + ep)
    order = rng.permutation(len(examples))
    for b in range(0, len(order), TB):
        batch = [examples[i] for i in order[b:b + TB]]
        maxlen = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), maxlen), tok.pad_token_id)
        lbl = torch.full((len(batch), maxlen), -100)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (f, l) in enumerate(batch):
            ids[i, :len(f)] = torch.tensor(f); lbl[i, :len(l)] = torch.tensor(l)
            att[i, :len(f)] = 1
        out = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device),
                    labels=lbl.to(model.device))
        out.loss.backward()
        opt.step(); opt.zero_grad()
        steps += 1
        if steps % 50 == 0:
            print(f"epoch {ep} step {steps} loss {out.loss.item():.4f}")
model = model.eval()
model.config.use_cache = True
model.save_pretrained(str(OUT_DIR / "lora_adapter"))

# ---------- 3. four-arm evaluation on the fixed sweep set ----------
eval_ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
eval_ds = eval_ds.shuffle(seed=SEED).select(range(N_EVAL))
eq = [ex["question"] for ex in eval_ds]
ealiases = [list(ex["answer"]["aliases"]) + [ex["answer"]["value"]] for ex in eval_ds]


def evaluate(template, tag):
    ans = batch_greedy(eq, template)
    abst = [is_abstain(a) for a in ans]
    corr = [(not ab) and is_correct(a, al) for a, ab, al in zip(ans, abst, ealiases)]
    wrong = [(not ab) and (not c) for ab, c in zip(abst, corr)]
    m = {"arm": tag, "n": N_EVAL,
         "abstain_rate": float(np.mean(abst)),
         "accuracy_overall": float(np.mean(corr)),
         "accuracy_on_attempted": float(np.sum(corr) / max(1, N_EVAL - np.sum(abst))),
         "wrong_rate": float(np.mean(wrong)),
         "abstention_aware_score": float((np.sum(corr) - np.sum(wrong)) / N_EVAL)}
    print(json.dumps(m), f"| {time.time()-t0:.0f}s")
    (OUT_DIR / f"answers_{tag}.json").write_text(json.dumps(ans))
    return m, ans


results = {}
with model.disable_adapter():
    results["A_base_plain"], _ = evaluate(PLAIN, "A_base_plain")
    results["B_base_abstainprompt"], _ = evaluate(ABSTAIN, "B_base_abstainprompt")
results["C_tuned_plain"], _ = evaluate(PLAIN, "C_tuned_plain")
results["D_tuned_abstainprompt"], _ = evaluate(ABSTAIN, "D_tuned_abstainprompt")

meta = {"model": MODEL_ID, "load_4bit": LOAD_4BIT, "device_map": DEVICE_MAP, "n_train": N_TRAIN, "train_accuracy": float(np.mean(tcorrect)),
        "epochs": EPOCHS, "seed": SEED, "arms": results,
        "wall_seconds": round(time.time() - t0, 1)}
(OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
