#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
wait_free 20000; run "$PY" -m bench.probe_bmm
