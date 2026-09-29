#!/bin/bash
# Stage 4: one GRPO arm on one GPU (resumable; exit 75 = out of time, requeue).
#   scripts/04_grpo.sh <arm> [extra rl.grpo flags]
#   scripts/04_grpo.sh tis --mode tis
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
ARM="${1:?arm name}"; shift
export VLLM_CACHE_ROOT="${TIM_VLLM_CACHE:-$DATA/vllm-cache}"; mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
[ -f "$DATA/rl/$ARM/DONE" ] && { info "arm $ARM already DONE -- skip"; exit 0; }
wait_free "${TIM_MIN_FREE_MIB:-60000}"
STOP="${TIM_STOP_AFTER_SEC:-0}"
"$PY" -m rl.grpo --arm "$ARM" --steps "${TIM_GRPO_STEPS:-200}" --stop-after-sec "$STOP" "$@"
rc=$?
if [ $rc -eq 75 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  info "requeueing $SLURM_JOB_ID to continue $ARM"; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
exit $rc
