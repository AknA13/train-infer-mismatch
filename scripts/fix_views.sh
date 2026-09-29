#!/bin/bash
# Re-score selected views and rebuild the report:  scripts/fix_views.sh <model> <view,view>
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
M="${1:?model}"; ONLY="${2:?views}"
TIM_ONLY="$ONLY" TIM_MODELS="$M" run bash scripts/02_score.sh "$M"
TIM_MODELS="$M" run bash scripts/03_report.sh "$M"
