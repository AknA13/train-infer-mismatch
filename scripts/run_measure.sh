#!/bin/bash
# Stages 1-3 for one model in one GPU job (sample -> score -> report).
#   slurm/submit.sh --job measure_1p7b --gpus 1 --requeue -- scripts/run_measure.sh Qwen/Qwen3-1.7B
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
M="${1:-$MODEL}"
run bash scripts/00_check_env.sh
TIM_MODELS="$M" run bash scripts/01_sample.sh "$M"
TIM_MODELS="$M" run_soft bash scripts/02_score.sh "$M"
TIM_MODELS="$M" run_soft bash scripts/03_report.sh "$M"
report_soft_fails
