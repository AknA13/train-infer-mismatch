#!/bin/bash
# Stage 3 (CPU): build results/measure_<tag>.json + docs/RESULTS_MEASURE_<tag>.md.
# Runs on the login node too: point TIM_DATA_ROOT at /net/<node>/data/... there.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[ -d "$DATA" ] || die "$DATA not visible from $(hostname); set TIM_DATA_ROOT=/net/horton$DATA"
MODELS="${1:-${TIM_MODELS:-$MODEL}}"
for m in $MODELS; do
  step "report $m"; run_soft "$PY" -m mismatch.report --model "$m"
done
report_soft_fails
