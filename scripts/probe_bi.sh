#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data; wait_free 30000
run "$PY" -m bench.probe_bi
