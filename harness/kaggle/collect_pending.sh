#!/usr/bin/env bash
# Collect finished Kaggle kernel outputs into harness/results. Idempotent.
# Handles single-job kernels (out/run_meta.json) and pair kernels built by
# build_pair_kernel.py (out/<job>/run_meta.json + out/pair_status.json).
# Usage: collect_pending.sh [kernel-slug ...]     (default: all slugs in KERNELS)
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP=$(mktemp -d)
# job/kernel name -> destination under harness/results
declare -A DEST=(
  [uhr-e3-abstention-qwen3-8b]=e3_abstention/qwen3-8b
  [uhr-e3-abstention-qwen3-14b]=e3_abstention/qwen3-14b-4bit
  [r2r3-qwen3-8b-4bit]=r2r3/qwen3-8b-4bit
  [e5-qwen3-1p7b]=e5_transfer/qwen3-1p7b
  [e5-qwen3-4b]=e5_transfer/qwen3-4b
  [e5-qwen3-8b]=e5_transfer/qwen3-8b
  [e5-qwen3-14b-4bit]=e5_transfer/qwen3-14b-4bit
  [e5-qwen3-1p7b-prefill]=e5_transfer/qwen3-1p7b-prefill
  [e5v2-qwen3-1p7b]=e5_transfer/v2/qwen3-1p7b
  [e5v2-qwen3-4b]=e5_transfer/v2/qwen3-4b
  [e5v2-qwen3-8b]=e5_transfer/v2/qwen3-8b
  [e5v2-qwen3-14b-4bit]=e5_transfer/v2/qwen3-14b-4bit
  [e5v2-mistral-7b]=e5_transfer/v2/mistral-7b
  [uhr-core-llama3-8b-ollama]=core_prompt/ollama--llama3-8b-instruct-q4_k_m
  [uhr-core-gemma3-4b-hf]=core_prompt/hf--google-gemma-3-4b-it
)
KERNELS=(uhr-core-llama3-8b-ollama uhr-core-gemma3-4b-hf uhr-e5v2-mistral-7b uhr-e3-abstention-qwen3-8b uhr-e3-abstention-qwen3-14b uhr-e5-pair-1p7b-4b uhr-pair-8b4bit-e5-14b uhr-e5-qwen3-8b uhr-e5-qwen3-1p7b-prefill uhr-e5v2-pair-1p7b-4b uhr-e5v2-qwen3-8b uhr-e5v2-qwen3-14b)
[ $# -gt 0 ] && KERNELS=("$@")
save() { # src_dir dest_key
  local d="$ROOT/harness/results/${DEST[$2]}"
  [ -z "${DEST[$2]}" ] && { echo "  !! no DEST for $2"; return; }
  mkdir -p "$d"; cp -rf "$1/." "$d/"; echo "  saved -> harness/results/${DEST[$2]}/"; head -c 400 "$d/run_meta.json"; echo
}
for k in "${KERNELS[@]}"; do
  st=$(kaggle kernels status "mastaan/$k" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*' || echo UNKNOWN)
  echo "--- $k: $st"
  case "$st" in *RUNNING|*QUEUED|UNKNOWN) continue;; esac
  mkdir -p "$TMP/$k" && (kaggle kernels output "mastaan/$k" -p "$TMP/$k" 2>/dev/null | grep -v outdated || true)
  if [ -f "$TMP/$k/out/pair_status.json" ]; then
    cat "$TMP/$k/out/pair_status.json"; echo
    for j in "$TMP/$k"/out/*/; do j=$(basename "$j"); [ -f "$TMP/$k/out/$j/run_meta.json" ] && save "$TMP/$k/out/$j" "$j" || echo "  $j: no run_meta.json (see $TMP/$k/out/$j.log)"; done
  elif [ -f "$TMP/$k/out/run_meta.json" ]; then
    save "$TMP/$k/out" "$k"
  else
    echo "  no run_meta.json (status $st); files: $(ls "$TMP/$k" 2>/dev/null | tr '\n' ' ')"
  fi
done
echo "--- done (tmp kept: $TMP)"
