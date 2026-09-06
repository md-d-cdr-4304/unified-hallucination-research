#!/usr/bin/env bash
# Keep Kaggle's 2 batch-GPU slots saturated: push queued kernel dirs as slots free,
# and collect outputs of finished kernels. Detach with:
#   setsid nohup harness/kaggle/dispatch_queue.sh <kernel-dir>... > harness/results/dispatch.log 2>&1 &
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUEUE=("$@")
ACTIVE=(uhr-e3-abstention-qwen3-8b uhr-e3-abstention-qwen3-14b)
declare -A COLLECTED
status() { kaggle kernels status "mastaan/$1" 2>/dev/null | grep -o 'Status\.[A-Z]*' | sed 's/Status\.//'; }
log() { echo "$(date '+%F %T') $*"; }
while :; do
  running=0
  for k in "${ACTIVE[@]}"; do
    st=$(status "$k")
    # empty = transient API error: assume still running rather than exiting early
    if [ "$st" = "RUNNING" ] || [ "$st" = "QUEUED" ] || [ -z "$st" ]; then running=$((running+1))
    elif [ -z "${COLLECTED[$k]}" ] && [ -n "$st" ]; then
      log "FINISHED $k ($st) — collecting"; COLLECTED[$k]=1
      bash "$ROOT/harness/kaggle/collect_pending.sh" "$k" 2>&1 | sed 's/^/    /'
    fi
  done
  while [ $running -lt 2 ] && [ ${#QUEUE[@]} -gt 0 ]; do
    d="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); slug=$(basename "$d" | sed 's/^kernel-//')
    out=$(cd "$d" && kaggle kernels push -p . 2>&1 | grep -v outdated)
    if echo "$out" | grep -q "successfully pushed"; then
      log "PUSHED $slug"; ACTIVE+=("$slug"); running=$((running+1)); sleep 60
    else
      log "PUSH FAILED $slug: $out"; QUEUE=("$d" "${QUEUE[@]}"); break
    fi
  done
  if [ ${#QUEUE[@]} -eq 0 ] && [ $running -eq 0 ]; then log "ALL DONE"; exit 0; fi
  sleep 120
done
