"""Build a Kaggle script kernel that runs up to two canonical jobs concurrently,
one per T4 (CUDA_VISIBLE_DEVICES=0 / 1). Doubles throughput under Kaggle's
2-concurrent-session limit for jobs that fit on one 14.5 GiB GPU.

Usage:
  build_pair_kernel.py <kernel-slug> <out_dir> JOB [JOB]
  JOB = name:script.py:gpu:KEY=VAL,KEY=VAL      (gpu = "0", "1" or "0,1")
Example:
  build_pair_kernel.py uhr-e5-pair-1p7b-4b /tmp/k \
     e5-1p7b:e5_transfer_generate.py:0:MODEL_ID=Qwen/Qwen3-1.7B \
     e5-4b:e5_transfer_generate.py:1:MODEL_ID=Qwen/Qwen3-4B
Each job writes to /kaggle/working/out/<name>/ ; collect_pending.sh maps names to
harness/results dirs.
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).parent
PIP = ["transformers", "accelerate", "datasets", "pyarrow", "bitsandbytes", "peft"]

RUNNER = r'''
import base64, json, os, subprocess, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + PIP, check=True)
ROOT = "/kaggle/working/out"; os.makedirs(ROOT, exist_ok=True)
procs = []
for job in JOBS:
    path = f"/kaggle/working/{job['name']}.py"
    open(path, "w").write(base64.b64decode(job["src"]).decode())
    env = dict(os.environ, **job["env"], CUDA_VISIBLE_DEVICES=job["gpu"], OUT_DIR=f"{ROOT}/{job['name']}")
    log = open(f"{ROOT}/{job['name']}.log", "w")
    p = subprocess.Popen([sys.executable, path], env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append((job["name"], p, log)); print("started", job["name"], "on GPU", job["gpu"], flush=True)
t0 = time.time(); rc = {}
while len(rc) < len(procs):
    time.sleep(120)
    for name, p, log in procs:
        if name in rc: continue
        r = p.poll()
        if r is not None:
            rc[name] = r; log.close(); print(f"[{time.time()-t0:.0f}s] {name} exited {r}", flush=True)
    for name, p, log in procs:
        try:
            lines = open(f"{ROOT}/{name}.log").read().splitlines()
            prog = [l for l in lines if "/" in l and "s |" in l or l.startswith("epoch") or l.startswith("{")]
            print(f"[{time.time()-t0:.0f}s] {name}: {(prog or lines or ['(no output yet)'])[-1][:160]}", flush=True)
        except Exception: pass
for name, p, log in procs:
    print("=" * 30, name, "exit", rc[name]); print(open(f"{ROOT}/{name}.log").read()[-4000:])
json.dump(rc, open(f"{ROOT}/pair_status.json", "w"))
sys.exit(max(rc.values()))
'''


def main():
    slug, out_dir, *jobspecs = sys.argv[1:]
    jobs = []
    for spec in jobspecs:
        name, script, gpu, kv = spec.split(":", 3)
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        src = base64.b64encode((HERE / script).read_bytes()).decode()
        jobs.append({"name": name, "script": script, "gpu": gpu, "env": env, "src": src})
    d = Path(out_dir) / f"kernel-{slug}"; d.mkdir(parents=True, exist_ok=True)
    code = f"PIP = {PIP!r}\nJOBS = {json.dumps(jobs)}\n" + RUNNER
    (d / f"{slug}.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"mastaan/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    print(f"built {d}  jobs={[ (j['name'], j['gpu']) for j in jobs ]}")


if __name__ == "__main__":
    main()
