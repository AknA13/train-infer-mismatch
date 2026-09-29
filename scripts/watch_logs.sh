#!/bin/bash
# Emit filtered new lines from every logs/*.out (picks up files created later).
# usage: scripts/watch_logs.sh '<egrep pattern>' [interval]
PAT="${1:?pattern}"; INT="${2:-30}"
declare -A OFF
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
for f in logs/*.out; do OFF["$f"]=$(stat -c %s "$f" 2>/dev/null || echo 0); done
while true; do
  for f in logs/*.out; do
    [ -f "$f" ] || continue
    sz=$(stat -c %s "$f"); o="${OFF[$f]:-0}"
    if [ "$sz" -gt "$o" ]; then
      tail -c +$((o+1)) "$f" | head -c $((sz-o)) | grep -E "$PAT" | sed "s|^|[${f#logs/}] |"
      OFF["$f"]=$sz
    elif [ "$sz" -lt "$o" ]; then OFF["$f"]=0; fi
  done
  sleep "$INT"
done
