"""Core prompt study (project plan Section 5.2): build the fixed evaluation set.

Four benchmark datasets, one hallucination-prone failure mode each:
  truthfulqa  factual fabrication        truthfulqa/truthful_qa (generation, validation, 817 q)
  squad2      false confidence           rajpurkar/squad_v2 (validation; 50% unanswerable)
  fever       unsupported assertion      pietrolesci/nli_fever (dev; claim + Wikipedia evidence, 3 labels)
  kilt        citation grounding         facebook/kilt_tasks (nq, validation; gold Wikipedia provenance)

Items are sampled once with seed 20260901 (same seed as the extension study) and
frozen to harness/data/core_prompt/items.jsonl so every model, prompt class and
run sees exactly the same instances. The first SENTINEL_PER_DATASET items of each
dataset form the 20-item sentinel subset (plan Section 5.1, silent model updates).
"""
import argparse, json, random, re
from pathlib import Path
from datasets import load_dataset

SEED = 20260901
OUT = Path(__file__).resolve().parents[2] / "data" / "core_prompt" / "items.jsonl"


def clean_wiki(t: str) -> str:
    t = t.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\s+([,.;:!?)'])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    return re.sub(r"\s+", " ", t).strip()


def build(n_per: int, sentinel_per: int) -> list[dict]:
    rng = random.Random(SEED)
    items = []

    # --- TruthfulQA (fabrication) ---
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    idx = list(range(len(ds))); rng.shuffle(idx)
    for i in idx[:n_per]:
        r = ds[i]
        items.append({"dataset": "truthfulqa", "id": f"tqa-{i}", "category": r["category"], "type": r["type"],
                      "question": r["question"], "best_answer": r["best_answer"],
                      "correct_answers": r["correct_answers"], "incorrect_answers": r["incorrect_answers"],
                      "answerable": True})

    # --- SQuAD 2.0 (false confidence): half answerable, half unanswerable ---
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ans = [i for i, r in enumerate(ds) if len(r["answers"]["text"]) > 0]
    una = [i for i, r in enumerate(ds) if len(r["answers"]["text"]) == 0]
    rng.shuffle(ans); rng.shuffle(una)
    half = n_per // 2
    picked = [(i, True) for i in ans[:half]] + [(i, False) for i in una[:n_per - half]]
    rng.shuffle(picked)
    for i, answerable in picked:
        r = ds[i]
        items.append({"dataset": "squad2", "id": f"sq-{r['id']}", "title": r["title"], "context": r["context"],
                      "question": r["question"], "gold_answers": sorted(set(r["answers"]["text"])),
                      "answerable": answerable})

    # --- FEVER (unsupported assertion): balanced 3-way, evidence text supplied ---
    ds = load_dataset("pietrolesci/nli_fever", split="dev")
    by_label = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
    for i, r in enumerate(ds):
        if len(r["hypothesis"]) < 2000:
            by_label[r["fever_gold_label"]].append(i)
    per = [n_per // 3 + (1 if k < n_per % 3 else 0) for k in range(3)]
    picked = []
    for (lab, pool), k in zip(by_label.items(), per):
        rng.shuffle(pool); picked += [(i, lab) for i in pool[:k]]
    rng.shuffle(picked)
    for i, lab in picked:
        r = ds[i]
        # nli_fever stores the claim in `premise` and the Wikipedia evidence in `hypothesis`.
        items.append({"dataset": "fever", "id": f"fev-{r['cid']}", "claim": r["premise"],
                      "evidence": clean_wiki(r["hypothesis"]), "gold_label": lab,
                      "answerable": lab != "NOT ENOUGH INFO"})

    # --- KILT NQ (citation grounding): questions with gold Wikipedia provenance ---
    ds = load_dataset("facebook/kilt_tasks", "nq", split="validation")
    pool = []
    for i, r in enumerate(ds):
        answers = sorted({o["answer"] for o in r["output"] if o["answer"]})
        titles = sorted({p["title"] for o in r["output"] for p in o["provenance"] if p["title"]})
        if answers and titles and len(r["input"]) < 200:
            pool.append((i, answers, titles))
    rng.shuffle(pool)
    for i, answers, titles in pool[:n_per]:
        r = ds[i]
        items.append({"dataset": "kilt", "id": f"kilt-{r['id']}", "question": r["input"],
                      "gold_answers": answers, "gold_titles": titles, "answerable": True})

    # sentinel flags: first `sentinel_per` of each dataset in sampled order
    seen = {}
    for it in items:
        c = seen.get(it["dataset"], 0); it["sentinel"] = c < sentinel_per; seen[it["dataset"]] = c + 1
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-dataset", type=int, default=200)
    ap.add_argument("--sentinel-per-dataset", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    items = build(a.n_per_dataset, a.sentinel_per_dataset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print("wrote", a.out, len(items), Counter(i["dataset"] for i in items),
          "sentinel", sum(i["sentinel"] for i in items),
          "answerable", Counter((i["dataset"], i["answerable"]) for i in items))
