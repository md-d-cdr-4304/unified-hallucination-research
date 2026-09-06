"""Core prompt study generator (project plan Sections 5.1, 5.3, 5.4).

Self-contained on purpose: this single file runs unchanged on the laptop (API
models), inside a Kaggle script kernel (Ollama Llama 3 8B Q4_K_M, or a
transformers model for the extension ladder), or anywhere with Python 3.10+.

Design (plan Section 5.1): for every task instance, only the prompt formulation
varies (baseline vs optimized); decoding is held constant (T=0, top_p=1, fixed
max_tokens). Each condition is run N_RUNS times; every response records the
model identifier string returned by the provider, the endpoint, and a UTC
timestamp. The sentinel subset is re-run at every session start and stored
separately so silent model updates can be tested (McNemar) in analyze.py.

Backends (env BACKEND):
  openai    gpt-4o-mini                   OPENAI_API_KEY
  gemini    gemini-1.5-flash (or current) GEMINI_API_KEY   (Google OpenAI-compatible endpoint)
  moonshot  moonshot-v1-8k                MOONSHOT_API_KEY
  ollama    llama3:8b-instruct-q4_K_M     local Ollama server (Kaggle kernel or laptop)
  nim       NVIDIA NIM catalogue model    NVIDIA_API_KEY   (pilot / judge only)
  hf        transformers model (greedy, mean log-prob, final-token hidden states) -> extension bridge

Env knobs: MODEL_ID, N_RUNS (10), PROMPTS (baseline,optimized), DATASETS (all four),
N_ITEMS_PER_DATASET (0 = all), MAX_TOKENS (256), WORKERS (8), OUT_DIR, ITEMS_PATH,
SESSION_ID (default = UTC start time), SKIP_SENTINEL (0/1).
"""
import hashlib, json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Prompt design (plan Section 5.4). Two classes per task.
# Optimized prompts follow the uniform four-element framework:
#   (1) task boundary definition
#   (2) explicit abstention condition with a standardised refusal phrase
#   (3) structured output requirement
#   (4) prohibition of unsupported inference
# ----------------------------------------------------------------------------
REFUSAL = "I do not have sufficient information to answer this reliably."
EXTRA_BODY = json.loads(os.environ.get("EXTRA_BODY", "{}"))  # recorded in run_meta and in the cache key

BASELINE = {
    "truthfulqa": "{question}",
    "squad2": "{context}\n\n{question}",
    "fever": "Is the following claim true or false?\n\nEvidence: {evidence}\n\nClaim: {claim}",
    "kilt": "{question} Also tell me the source of your answer.",
}

OPTIMIZED = {
    "truthfulqa": (
        "Answer the following question based only on well-established, verifiable facts. "            # (1)
        f"If you are not certain of the answer, respond with exactly: '{REFUSAL}' "                      # (2)
        "Do not speculate or infer beyond what is documented. "                                          # (4)
        "Respond in the format:\nANSWER: <one or two sentences, or the refusal sentence>\n\n"            # (3)
        "Question: {question}"
    ),
    "squad2": (
        "Answer the question using only the passage below. "                                             # (1)
        "The passage may not contain sufficient information to answer the question; if it does not, "
        f"respond with exactly: '{REFUSAL}' "                                                            # (2)
        "Providing an answer that the passage does not support counts as an error. "
        "Do not use outside knowledge and do not infer beyond what the passage states. "                # (4)
        "Respond in the format:\nANSWER: <short answer copied from the passage, or the refusal sentence>\n\n"  # (3)
        "Passage: {context}\n\nQuestion: {question}"
    ),
    "fever": (
        "Classify the claim using only the evidence provided. "                                          # (1)
        "Choose exactly one label: SUPPORTED (the evidence confirms the claim), REFUTED (the evidence "
        "contradicts the claim), or NOT ENOUGH INFO (the evidence does not settle the claim). "
        "If the evidence does not settle the claim you must answer NOT ENOUGH INFO. "                    # (2)
        "Do not use outside knowledge and do not infer beyond what the evidence states. "               # (4)
        "Respond in the format:\nVERDICT: <SUPPORTED | REFUTED | NOT ENOUGH INFO>\n\n"                   # (3)
        "Evidence: {evidence}\n\nClaim: {claim}"
    ),
    "kilt": (
        "Answer the question based only on well-established, verifiable facts, and name the Wikipedia "
        "article that documents the answer. "                                                            # (1)
        f"If you are not certain of the answer, respond with exactly: '{REFUSAL}' "                      # (2)
        "If you cannot name a specific Wikipedia article that supports your answer, write SOURCE: NONE; "
        "do not invent a source. Do not speculate or infer beyond what is documented. "                  # (4)
        "Respond in the format:\nANSWER: <short answer, or the refusal sentence>\nSOURCE: <exact Wikipedia article title, or NONE>\n\n"  # (3)
        "Question: {question}"
    ),
}

PROMPT_ELEMENTS = {  # documented mapping for the thesis (Appendix B)
    "task_boundary": "(1) 'based only on well-established, verifiable facts' / 'using only the passage' / 'using only the evidence'",
    "abstention": f"(2) standardised refusal phrase '{REFUSAL}' (FEVER: mandatory NOT ENOUGH INFO)",
    "structured_output": "(3) ANSWER: / VERDICT: / ANSWER: + SOURCE: line format",
    "no_unsupported_inference": "(4) 'Do not speculate or infer beyond what is documented / the passage / the evidence'",
}


def render(item: dict, prompt_class: str) -> str:
    tpl = (BASELINE if prompt_class == "baseline" else OPTIMIZED)[item["dataset"]]
    return tpl.format(**{k: item.get(k, "") for k in ("question", "context", "evidence", "claim")})


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
BACKENDS = {
    "openai":   {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "gemini":   {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY", "default_model": "gemini-1.5-flash"},
    "moonshot": {"base_url": "https://api.moonshot.ai/v1", "key_env": "MOONSHOT_API_KEY", "default_model": "moonshot-v1-8k"},
    "ollama":   {"base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"), "key_env": None, "default_model": "llama3:8b-instruct-q4_K_M"},
    "nim":      {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY", "default_model": "meta/llama-3.2-11b-vision-instruct"},
    "hf":       {"base_url": None, "key_env": None, "default_model": "Qwen/Qwen3-1.7B"},
}


class ChatBackend:
    """OpenAI-compatible chat backend with on-disk cache keyed by
    (model, messages, params, run_idx, session_tag). Deterministic decoding (T=0)."""

    def __init__(self, backend: str, model: str, cache_dir: Path, max_tokens: int):
        from openai import OpenAI
        cfg = BACKENDS[backend]
        key = os.environ.get(cfg["key_env"]) if cfg["key_env"] else "local"
        if cfg["key_env"] and not key:
            raise RuntimeError(f"{cfg['key_env']} is not set")
        self.backend, self.model, self.endpoint = backend, model, cfg["base_url"]
        self.client = OpenAI(base_url=cfg["base_url"], api_key=key or "local", timeout=120)
        self.cache_dir = cache_dir; cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.lock = threading.Lock(); self.n_calls = 0; self.n_hits = 0

    def _path(self, key: dict) -> Path:
        return self.cache_dir / (hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".json")

    def complete(self, prompt: str, run_idx: int, session_tag: str = "") -> dict:
        messages = [{"role": "user", "content": prompt}]
        key = {"backend": self.backend, "model": self.model, "messages": messages, "temperature": 0.0,
               "top_p": 1.0, "max_tokens": self.max_tokens, "run_idx": run_idx, "session_tag": session_tag,
               **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {})}
        p = self._path(key)
        if p.exists():
            try:  # another process may be writing this file; a partial read counts as a miss
                rec = json.loads(p.read_text())
                with self.lock: self.n_hits += 1
                return rec
            except (json.JSONDecodeError, OSError):
                pass
        delay = 2.0
        for attempt in range(8):
            try:
                t0 = time.time()
                extra = {}
                if self.backend == "ollama":
                    extra["extra_body"] = {"options": {"seed": 20260901 + run_idx, "num_predict": self.max_tokens}}
                if self.backend == "openai":
                    extra["seed"] = 20260901 + run_idx
                if EXTRA_BODY:  # provider-specific knobs, e.g. {"reasoning_effort": "low"} for gpt-oss on NIM
                    extra["extra_body"] = {**extra.get("extra_body", {}), **EXTRA_BODY}
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0, top_p=1.0,
                    max_tokens=self.max_tokens, **extra)
                choice = resp.choices[0]
                rec = {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                       "model_version": getattr(resp, "model", None) or self.model,
                       "system_fingerprint": getattr(resp, "system_fingerprint", None),
                       "endpoint": self.endpoint, "ts_utc": datetime.now(timezone.utc).isoformat(),
                       "latency_s": round(time.time() - t0, 3), "run_idx": run_idx, "session_tag": session_tag,
                       "usage": (resp.usage.model_dump() if getattr(resp, "usage", None) else None)}
                tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(rec, ensure_ascii=False)); tmp.replace(p)  # atomic
                with self.lock: self.n_calls += 1
                return rec
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                if attempt < 7 and (status in (408, 409, 429, 500, 502, 503, 504) or status is None):
                    time.sleep(delay); delay = min(delay * 2, 60); continue
                raise
        raise RuntimeError("unreachable")


class HFBackend:
    """transformers greedy generation with mean token log-prob and final-token hidden
    states (last and penultimate layer) so the extension's probe/log-prob detectors
    can be applied to the residual hallucinations under each prompt class."""

    def __init__(self, model_id: str, max_tokens: int, load_4bit: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.backend, self.model, self.endpoint = "hf", model_id, "local-transformers"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float32 if "gemma" in model_id.lower() else torch.float16
        self.m = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.max_tokens = max_tokens; self.n_calls = 0; self.n_hits = 0
        self.hidden = {"last_layer_final": [], "penult_layer_final": [], "keys": []}

    def complete(self, prompt: str, run_idx: int, session_tag: str = "", key: str = "") -> dict:
        torch = self.torch
        extra = {"enable_thinking": False} if "qwen3" in self.model.lower() else {}
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, **extra)
        ids = self.tok(text, return_tensors="pt").to(self.m.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.m.generate(**ids, do_sample=False, max_new_tokens=self.max_tokens,
                                  output_scores=True, output_hidden_states=True, return_dict_in_generate=True,
                                  pad_token_id=self.tok.eos_token_id)
        gen = out.sequences[0, ids.input_ids.shape[1]:]
        lps = []
        for t, sc in enumerate(out.scores):
            lp = torch.log_softmax(sc[0].float(), -1)[gen[t]].item(); lps.append(lp)
        n = len(lps)
        h_last = out.hidden_states[n - 1][-1][0, -1, :].float().cpu().numpy().astype("float16")
        h_pen = out.hidden_states[n - 1][-2][0, -1, :].float().cpu().numpy().astype("float16")
        self.hidden["last_layer_final"].append(h_last); self.hidden["penult_layer_final"].append(h_pen)
        self.hidden["keys"].append(key)
        self.n_calls += 1
        return {"text": self.tok.decode(gen, skip_special_tokens=True), "finish_reason": "stop" if n < self.max_tokens else "length",
                "model_version": self.model, "system_fingerprint": None, "endpoint": self.endpoint,
                "ts_utc": datetime.now(timezone.utc).isoformat(), "latency_s": round(time.time() - t0, 3),
                "run_idx": run_idx, "session_tag": session_tag, "mean_logprob": float(sum(lps) / max(n, 1)), "n_tokens": n}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def main():
    backend = os.environ.get("BACKEND", "nim")
    model = os.environ.get("MODEL_ID") or BACKENDS[backend]["default_model"]
    n_runs = int(os.environ.get("N_RUNS", "10"))
    prompts = os.environ.get("PROMPTS", "baseline,optimized").split(",")
    datasets = os.environ.get("DATASETS", "truthfulqa,squad2,fever,kilt").split(",")
    n_items = int(os.environ.get("N_ITEMS_PER_DATASET", "0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "256"))
    workers = int(os.environ.get("WORKERS", "8"))
    here = Path(__file__).resolve()
    items_path = Path(os.environ.get("ITEMS_PATH") or (here.parents[2] / "data" / "core_prompt" / "items.jsonl"))
    slug = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")
    out_dir = Path(os.environ.get("OUT_DIR") or (here.parents[2] / "results" / "core_prompt" / f"{backend}--{slug}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = os.environ.get("SESSION_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_dir = Path(os.environ.get("CACHE_DIR") or (here.parents[2] / "results" / ".cache_core"))

    items = [json.loads(l) for l in open(items_path)]
    items = [i for i in items if i["dataset"] in datasets]
    if n_items:
        keep = {}
        items = [i for i in items if keep.setdefault(i["dataset"], []).append(i["id"]) or len(keep[i["dataset"]]) <= n_items]

    meta = {"experiment": "core_prompt", "backend": backend, "model": model, "n_runs": n_runs, "prompts": prompts,
            "datasets": datasets, "n_items": len(items), "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "items_path": str(items_path), "items_sha256": hashlib.sha256(open(items_path, "rb").read()).hexdigest()[:16],
            "session_id": session_id, "started_utc": datetime.now(timezone.utc).isoformat(), "refusal_phrase": REFUSAL,
            "prompt_elements": PROMPT_ELEMENTS, "extra_body": EXTRA_BODY, "baseline_templates": BASELINE, "optimized_templates": OPTIMIZED}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if backend == "hf":
        be = HFBackend(model, max_tokens, os.environ.get("LOAD_4BIT", "0") == "1"); workers = 1
        if n_runs != 1:
            print("hf backend is deterministic at T=0; running N_RUNS=1 and noting it", flush=True); n_runs = 1
    else:
        be = ChatBackend(backend, model, cache_dir, max_tokens)

    # --- sentinel subset at session start (plan Section 5.1) ---
    sentinel = [i for i in items if i.get("sentinel")]
    if sentinel and os.environ.get("SKIP_SENTINEL", "0") != "1" and backend != "hf":
        rows = []
        for it in sentinel:
            rec = be.complete(render(it, "baseline"), run_idx=0, session_tag=f"sentinel:{session_id}")
            rows.append({"item_id": it["id"], "dataset": it["dataset"], "prompt_class": "baseline", **rec})
        (out_dir / f"sentinel_{session_id}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"sentinel: {len(rows)} items for session {session_id}", flush=True)

    # --- main grid: items x prompts x runs ---
    jobs = [(it, pc, r) for it in items for pc in prompts for r in range(n_runs)]
    print(f"{backend}/{model}: {len(items)} items x {len(prompts)} prompts x {n_runs} runs = {len(jobs)} calls", flush=True)
    out_path = out_dir / "responses.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            d = json.loads(l); done.add((d["item_id"], d["prompt_class"], d["run_idx"]))
    jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]
    print(f"resuming: {len(done)} done, {len(jobs)} to go", flush=True)

    def work(job):
        it, pc, r = job
        key = f"{it['id']}|{pc}|{r}"
        rec = be.complete(render(it, pc), run_idx=r, **({"key": key} if backend == "hf" else {}))
        return {"item_id": it["id"], "dataset": it["dataset"], "prompt_class": pc, "backend": backend, "model": model, **rec}

    t0 = time.time(); n = 0
    with open(out_path, "a") as f:
        if workers == 1:
            for job in jobs:
                f.write(json.dumps(work(job), ensure_ascii=False) + "\n"); n += 1
                if n % 50 == 0: f.flush(); print(f"{n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
        else:
            with ThreadPoolExecutor(workers) as ex:
                futs = [ex.submit(work, j) for j in jobs]
                for fu in as_completed(futs):
                    f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n"); n += 1
                    if n % 100 == 0: f.flush(); print(f"{n}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    if backend == "hf" and be.hidden["keys"]:
        import numpy as np
        np.savez_compressed(out_dir / "hidden_states.npz", keys=np.array(be.hidden["keys"]),
                            last_layer_final=np.stack(be.hidden["last_layer_final"]),
                            penult_layer_final=np.stack(be.hidden["penult_layer_final"]))
    meta.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": round(time.time() - t0, 1),
                 "api_calls": be.n_calls, "cache_hits": be.n_hits, "n_responses_total": len(done) + n})
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("done", json.dumps({k: meta[k] for k in ("wall_seconds", "api_calls", "cache_hits", "n_responses_total")}), flush=True)


if __name__ == "__main__":
    main()
