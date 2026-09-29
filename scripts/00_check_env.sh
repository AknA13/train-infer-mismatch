#!/bin/bash
# Fail fast on anything that would waste a GPU hour.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
step "environment"
info "python   : $PY"
case "$PY" in
  /data/*) info "           (node-local env)" ;;
  /scratch/*) warn "python is on /scratch (NFS): 690 s imports measured on compute nodes 2026-09-28" ;;
esac
"$PY" -c "import torch, transformers, vllm" 2>/dev/null || die "cannot import torch/transformers/vllm with $PY"
info "repo     : $REPO"
info "data root: $DATA"
"$PY" -c "import sys; sys.path.insert(0,'$REPO'); import config as C; print('[cfg] '+C.describe())"
case "$DATA" in
  /accounts/*|/scratch/*) die "TIM_DATA_ROOT=$DATA is on a quota'd filesystem; use node-local /data/\$USER/..." ;;
esac
mkdir -p "$DATA" 2>/dev/null || die "cannot create $DATA (is /data present on $(hostname)?)"

step "packages"
"$PY" - <<'PY'
import importlib, sys
for m in ("torch", "transformers", "vllm", "datasets", "numpy"):
    mod = importlib.import_module(m); print(f"  {m:14s} {getattr(mod,'__version__','?')}")
import torch; print(f"  cuda available {torch.cuda.is_available()}")
for m in ("flashinfer", "vllm.vllm_flash_attn", "triton", "math_verify"):
    print(f"  {m:22s} {'present' if importlib.util.find_spec(m) else 'ABSENT'}")
PY

step "models and datasets"
"$PY" - <<PY
import os, sys
from pathlib import Path
from transformers import AutoConfig
from transformers.utils import cached_file
ok = True
for mid in os.environ.get("TIM_MODELS", os.environ["TIM_MODEL_ID"]).split():
    try:
        c = AutoConfig.from_pretrained(mid)
        snap = Path(cached_file(mid, "config.json")).parent
        gb = sum(p.stat().st_size for p in snap.glob("*.safetensors")) / 1e9
        print(f"  {mid:22s} layers={c.num_hidden_layers} hidden={c.hidden_size} vocab={c.vocab_size} weights={gb:.1f}GB")
        if gb < 0.5: print("    !! no weights (config stub)"); ok = False
    except Exception as e:
        print(f"  {mid:22s} UNAVAILABLE: {type(e).__name__}: {e}"); ok = False
from datasets import load_dataset
for name, cfg, split in (("openai/gsm8k", "main", "train"), ("openai/gsm8k", "main", "test")):
    try:
        ds = load_dataset(name, cfg, split=split); print(f"  {name}/{split}: {len(ds)} rows")
    except Exception as e:
        print(f"  {name}/{split}: UNAVAILABLE {e}"); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || die "models/datasets not all reachable on $(hostname)"
echo "[ok] environment ready"
