#!/bin/bash
# Submit GRPO arms (one GPU each, requeue on preemption, checkpoint before wall).
#   scripts/launch_arms.sh nt_none nt_tis          # named arms from the table below
#   scripts/launch_arms.sh --list
# Regime "nt" = non-thinking, 768 max tokens (fast, ~200-400 tok completions).
# Regime "th" = thinking, 2048 max tokens (long-horizon stress).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
declare -A ARMS=(
  [nt_none]="--mode none"
  [nt_tis]="--mode tis"
  [nt_mis]="--mode mis"
  [nt_icepop]="--mode icepop"
  [nt_seqtis]="--mode seq_tis"
  [nt_vllmold]="--mode vllm_old"
  [nt_fp16]="--mode none --dtype fp16"
  [nt_fp32head]="--mode none --fp32-head both"
  [nt_fp32head_tr]="--mode none --fp32-head trainer"
  [nt_bi]="--mode none --batch-invariant"
  [th_none]="--mode none"
  [th_tis]="--mode tis"
  [th_fp16]="--mode none --dtype fp16"
)
NT="--no-thinking --max-tokens 768"
TH="--max-tokens 2048 --gmu 0.45"
[ "${1:-}" = "--list" ] && { for k in "${!ARMS[@]}"; do echo "$k: ${ARMS[$k]}"; done | sort; exit 0; }
for arm in "$@"; do
  [ -n "${ARMS[$arm]+x}" ] || die "unknown arm $arm (see --list)"
  case "$arm" in nt_*) REG="$NT";; th_*) REG="$TH";; esac
  # 6 h wall, checkpoint+requeue at 5.5 h
  TIM_STOP_AFTER_SEC=19800 slurm/submit.sh --job "grpo_$arm" --gpus 1 --time 6:00:00 --requeue -- \
    scripts/04_grpo.sh "$arm" ${ARMS[$arm]} $REG
done
