#!/bin/bash
# 3-step GRPO smoke on one GPU: proves engine+trainer colocate, weights sync,
# rewards verify, checkpoint/resume works. ~10 min including vLLM startup.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
ARM="${1:-smoke}"
rm -rf "$DATA/rl/$ARM"
TIM_GRPO_STEPS=3 bash scripts/04_grpo.sh "$ARM" --prompts-per-step 4 --group 4 --max-tokens 256 \
  --eval-n 8 --eval-every 3 --ckpt-every 2 --micro-batch 4 "${@:2}" || die "smoke run failed"
step "resume from checkpoint (step 2 -> 3)"
rm -f "$DATA/rl/$ARM/DONE"
TIM_GRPO_STEPS=4 bash scripts/04_grpo.sh "$ARM" --prompts-per-step 4 --group 4 --max-tokens 256 \
  --eval-n 8 --eval-every 4 --ckpt-every 2 --micro-batch 4 "${@:2}" || die "resume failed"
grep -c '"step"' "$DATA/rl/$ARM/steps.jsonl"
echo "[ok] smoke complete"
