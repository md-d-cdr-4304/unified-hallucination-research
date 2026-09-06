"""Build a Kaggle script kernel for the core prompt study on a local model.

Two modes:
  ollama  Llama 3 8B Instruct Q4_K_M served by Ollama on the T4 (the plan's local model)
  hf      transformers backend (extension ladder: Qwen3 etc., saves log-probs + hidden states)

Usage:
  build_core_prompt_kernel.py <slug> <out_dir> ollama [KEY=VAL ...]
  build_core_prompt_kernel.py <slug> <out_dir> hf MODEL_ID=Qwen/Qwen3-4B [LOAD_4BIT=1 ...]
The kernel embeds generate.py and items.jsonl (base64) so it is self-contained.
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[0] / "experiments" / "core_prompt" / "generate.py"
ITEMS = HERE.parents[0] / "data" / "core_prompt" / "items.jsonl"

OLLAMA_SETUP = r'''
import os, subprocess, sys, time, urllib.request
subprocess.run("apt-get install -y -qq zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd); curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MODELS="/kaggle/working/ollama_models", OLLAMA_NUM_PARALLEL="4", OLLAMA_KEEP_ALIVE="-1")
srv = subprocess.Popen(["ollama", "serve"], env=env, stdout=open("/kaggle/working/ollama_serve.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2); break
    except Exception: time.sleep(2)
subprocess.run(["ollama", "pull", os.environ.get("MODEL_ID", "llama3:8b-instruct-q4_K_M")], env=env, check=True)
print(subprocess.run(["ollama", "show", os.environ.get("MODEL_ID", "llama3:8b-instruct-q4_K_M")], env=env, capture_output=True, text=True).stdout[:2000], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai"], check=True)
'''

HF_SETUP = r'''
import glob, os, subprocess, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# HF token for gated models (Llama 3, Gemma): private dataset mastaan/hf-token-secret, web-UI secret as fallback
_m = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
if _m:
    os.environ["HF_TOKEN"] = open(_m[0]).read().strip()
else:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "numpy"], check=True)
'''

TAIL = r'''
import base64, os
os.makedirs("/kaggle/working/out", exist_ok=True)
open("/kaggle/working/items.jsonl", "wb").write(base64.b64decode(ITEMS_B64))
open("/kaggle/working/generate.py", "wb").write(base64.b64decode(GEN_B64))
os.environ.setdefault("ITEMS_PATH", "/kaggle/working/items.jsonl")
os.environ.setdefault("OUT_DIR", "/kaggle/working/out")
os.environ.setdefault("CACHE_DIR", "/kaggle/working/cache")
import runpy; runpy.run_path("/kaggle/working/generate.py", run_name="__main__")
import shutil; shutil.rmtree("/kaggle/working/ollama_models", ignore_errors=True); shutil.rmtree("/kaggle/working/cache", ignore_errors=True)
'''


def main():
    slug, out_dir, mode, *kv = sys.argv[1:]
    env = dict(k.split("=", 1) for k in kv)
    env.setdefault("BACKEND", mode)
    if mode == "ollama": env.setdefault("MODEL_ID", "llama3:8b-instruct-q4_K_M"); env.setdefault("WORKERS", "4")
    if mode == "hf": env.setdefault("N_RUNS", "1"); env.setdefault("WORKERS", "1")
    header = "import os\n" + "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in env.items())
    src = header + (OLLAMA_SETUP if mode == "ollama" else HF_SETUP)
    src += f"\nITEMS_B64 = {base64.b64encode(ITEMS.read_bytes()).decode()!r}\nGEN_B64 = {base64.b64encode(GEN.read_bytes()).decode()!r}\n" + TAIL
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / f"{slug}.py").write_text(src)
    (od / "kernel-metadata.json").write_text(json.dumps({
        "id": f"mastaan/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": (["mastaan/hf-token-secret"] if mode == "hf" else []), "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    print("built", od / f"{slug}.py", "env", env)


if __name__ == "__main__":
    main()
