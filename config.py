"""Central configuration for the train/inference mismatch project.

Machine-specific values come from the environment (see env.sh.example):

  TIM_DATA_ROOT  large artifacts (sampled corpora, per-view logprob arrays,
                 RL checkpoints). Node-local /data on this cluster, never $HOME.
  TIM_MODEL_ID   policy model for a single stage invocation.
  HF_HOME        HuggingFace cache root.

Small JSON summaries land in results/ and are committed.

The vocabulary of the project:
  view    one way of computing log pi(token | prefix) for a fixed token
          sequence: e.g. "vllm sampler at generation time", "vllm prefill
          re-score with FlashInfer", "HF bf16 sdpa", "HF fp32 eager".
  ref     the view we treat as ground truth: HF fp32, eager attention, TF32
          off, batch size 1. Its own error is bounded against fp64 on a subset.
  gap     the per-token difference between two views on the SAME tokens.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("TIM_DATA_ROOT", str(REPO_ROOT / "runs" / "default")))

CORPUS_DIR = DATA_ROOT / "corpus"      # <model_tag>/samples.jsonl  (+ meta.json)
SCORES_DIR = DATA_ROOT / "scores"      # <model_tag>/<view>.npz
RL_DIR = DATA_ROOT / "rl"              # GRPO checkpoints + step logs
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"

MODEL_ID = os.environ.get("TIM_MODEL_ID", "Qwen/Qwen3-1.7B")

# ---- sampling corpus --------------------------------------------------------
# RL-style sampling: the policy's own distribution, untempered and untruncated.
# Qwen3's generation_config ships top_k=20 / top_p=0.95 / T=0.6, which vLLM
# would silently apply; we override all three so the sampler distribution IS
# the model distribution and the sampler logprob is comparable to a forward.
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 1.0
SAMPLE_TOP_K = -1
SAMPLE_MAX_TOKENS = int(os.environ.get("TIM_SAMPLE_MAXTOK", 2048))
SAMPLE_N = int(os.environ.get("TIM_SAMPLE_N", 512))
SAMPLE_SEED = 1234
SAMPLE_DATASET = "gsm8k"
SAMPLE_SPLIT = "train"

THINK_CLOSE = "</think>"
MATH500_ID = "HuggingFaceH4/MATH-500"

# ---- metrics ----------------------------------------------------------------
# Trust band for the per-token ratio pi_train / pi_infer. Tokens outside it are
# the ones masking-style corrections (MIS / IcePop) drop.
RATIO_BAND = (0.8, 1.25)
TIS_CLIP = 2.0                       # truncated IS ceiling (Yao et al.)
PROB_BUCKETS = [0.0, 0.01, 0.1, 0.5, 0.9, 1.0001]   # by reference token probability
POS_BUCKET = 256                     # tokens per position bucket

# ---- gates ------------------------------------------------------------------
# M1  the reference is a reference: fp32 eager vs fp32 sdpa mean |gap| below
#     this, and fp32 vs fp64 (subset) below it too. Otherwise "ground truth"
#     is itself noise and nothing downstream means anything.
M1_REF_NOISE_MAX = 1e-3
# M2  the documented phenomenon reproduces: bf16 sampler vs bf16 trainer gap is
#     at least this many times the M1 noise floor.
M2_MIN_RATIO_OVER_NOISE = 10.0
# M3  attribution is additive enough to be useful: single-toggle deltas sum to
#     within this fraction of the full baseline gap (else report interaction).
M3_ADDITIVITY_TOL = 0.25
# M4  RL: either an unstable/stable pair, or measured drift of the mismatch
#     over training plus a cost table. Both are reportable; neither is skipped.
# M5  monitor overhead as a fraction of trainer step time.
M5_MONITOR_OVERHEAD_MAX = 0.02


def model_tag(model_id=None):
    return (model_id or MODEL_ID).split("/")[-1].replace(".", "p")


def ensure_dirs():
    for d in (CORPUS_DIR, SCORES_DIR, RL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def describe():
    return f"data_root={DATA_ROOT} model={MODEL_ID} n={SAMPLE_N} max_tokens={SAMPLE_MAX_TOKENS}"


def publish_result(name, payload):
    """Write results/<name>.json atomically (small, committed)."""
    import json
    import tempfile
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    fd, tmp = tempfile.mkstemp(dir=str(RESULTS_DIR), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, out)
    return out
