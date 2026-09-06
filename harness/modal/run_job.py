"""Run any canonical harness/kaggle job on Modal (second GPU platform, runs in
parallel with Kaggle). One-time setup by the user:
    pip install modal && modal setup          # browser login, free monthly credits
Then, from the repo root:
    modal run harness/modal/run_job.py --script e5_transfer_generate.py \
        --name e5-qwen3-4b --gpu T4 --env MODEL_ID=Qwen/Qwen3-4B
    modal run harness/modal/run_job.py --script r2r3_generate.py \
        --name r2r3-qwen3-14b-fp16 --gpu A100-40GB --env MODEL_ID=Qwen/Qwen3-14B,LOAD_4BIT=0
    modal run --detach ...   # keeps running after the terminal closes
Outputs land in the Modal volume `uhr-results` under /<name>/ and are copied to
harness/results/_modal/<name>/ when the run returns (or fetch later with
    modal volume get uhr-results /<name> harness/results/_modal/<name>).
GPU choices: T4 (16 GB, cheapest), L4 (24 GB), A10G (24 GB), A100-40GB, A100-80GB.
"""
import os, subprocess, sys
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
KAGGLE_DIR = HERE.parent / "kaggle"
app = modal.App("uhr-hallucination")
vol = modal.Volume.from_name("uhr-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "accelerate", "datasets", "pyarrow", "pandas",
                      "numpy", "scikit-learn", "peft", "bitsandbytes", "sentencepiece")
         .add_local_dir(str(KAGGLE_DIR), remote_path="/job"))


def _run(script: str, name: str, env: dict):
    out = f"/results/{name}"
    os.makedirs(out, exist_ok=True)
    e = dict(os.environ, **env, OUT_DIR=out, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if "HF_TOKEN" in os.environ:
        e["HF_TOKEN"] = os.environ["HF_TOKEN"]
    log = open(f"{out}/job.log", "w")
    p = subprocess.Popen([sys.executable, f"/job/{script}"], env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.write(line)
    rc = p.wait(); log.close(); vol.commit()
    return rc


# one function per GPU type (Modal fixes the GPU per function)
def _make(gpu):
    @app.function(image=image, gpu=gpu, timeout=12 * 3600, volumes={"/results": vol},
                  secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
                  name=f"run_{gpu.replace('-', '_').lower()}")
    def f(script: str, name: str, env: dict):
        return _run(script, name, env)
    return f

RUNNERS = {g: _make(g) for g in ["T4", "L4", "A10G", "A100-40GB", "A100-80GB"]}


@app.local_entrypoint()
def main(script: str, name: str, gpu: str = "T4", env: str = ""):
    kv = dict(x.split("=", 1) for x in env.split(",") if x)
    print(f"launching {script} as {name} on {gpu} env={kv}")
    rc = RUNNERS[gpu].remote(script, name, kv)
    print(f"exit code {rc}")
    dest = HERE.parent / "results" / "_modal" / name
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "uhr-results", f"/{name}", str(dest)], check=False)
    print(f"outputs -> {dest}")
