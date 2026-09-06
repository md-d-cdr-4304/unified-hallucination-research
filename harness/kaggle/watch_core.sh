#!/usr/bin/env bash
# Watch the two core-prompt kernels; collect each when it stops running. Log: results/core_prompt/watch_core.log
cd "$(dirname "$0")/.."
LOG=results/core_prompt/watch_core.log
declare -A DONE
while true; do
  for k in uhr-core-llama3-8b-ollama uhr-core-gemma3-4b-hf; do
    [ -n "${DONE[$k]}" ] && continue
    st=$(kaggle kernels status "mastaan/$k" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*')
    echo "$(date +%H:%M) $k $st" >> $LOG
    case "$st" in
      *COMPLETE|*ERROR|*CANCEL*) echo "$(date +%H:%M) collecting $k" >> $LOG; bash kaggle/collect_pending.sh "$k" >> $LOG 2>&1; DONE[$k]=1;;
    esac
  done
  [ ${#DONE[@]} -eq 2 ] && { echo "$(date +%H:%M) both collected" >> $LOG; exit 0; }
  sleep 120
done
