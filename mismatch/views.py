"""The view grid: one baseline per family plus one-toggle-at-a-time variants.

Names are stable identifiers used for file names and in results/. Each entry
says which scorer runs it and with which flags. `toggle` is the single thing
that differs from the family baseline, which is what makes the attribution in
report.py an additive decomposition rather than a pile of numbers.

Families
  vllm_sample   the sampler's own logprobs from stage 1 (decode path). Not a
                scorer run: read straight from the corpus.
  vllm:*        prefill re-score through vLLM
  hf:*          transformers forward
"""

REF = "hf_fp32_eager"

VIEWS = [
    # --- reference and its own noise floor (gate M1) -----------------------
    dict(name="hf_fp32_eager", scorer="hf", toggle="reference",
         flags=["--dtype", "fp32", "--attn", "eager", "--batch-size", "1", "--save-extra"]),
    dict(name="hf_fp64_eager", scorer="hf", toggle="fp64 (subset)", limit=64,
         flags=["--dtype", "fp64", "--attn", "eager", "--batch-size", "1"]),
    dict(name="hf_fp32_sdpa", scorer="hf", toggle="sdpa kernel @fp32",
         flags=["--dtype", "fp32", "--attn", "sdpa", "--batch-size", "1"]),
    dict(name="hf_fp32_tf32", scorer="hf", toggle="TF32 matmul @fp32",
         flags=["--dtype", "fp32", "--attn", "sdpa", "--batch-size", "1", "--tf32"]),

    # --- trainer-side baseline and toggles ---------------------------------
    dict(name="hf_bf16_sdpa_bs1", scorer="hf", toggle="trainer baseline", baseline="hf",
         flags=["--dtype", "bf16", "--attn", "sdpa", "--batch-size", "1"]),
    dict(name="hf_bf16_eager_bs1", scorer="hf", toggle="eager attention",
         flags=["--dtype", "bf16", "--attn", "eager", "--batch-size", "1"]),
    dict(name="hf_bf16_sdpa_bs8", scorer="hf", toggle="padded batch of 8",
         flags=["--dtype", "bf16", "--attn", "sdpa", "--batch-size", "8"]),
    dict(name="hf_bf16_sdpa_bs1_fp32head", scorer="hf", toggle="fp32 norm+lm_head",
         flags=["--dtype", "bf16", "--attn", "sdpa", "--batch-size", "1", "--fp32-head"]),
    dict(name="hf_bf16_sdpa_bs1_nativelogits", scorer="hf", toggle="log_softmax in bf16",
         flags=["--dtype", "bf16", "--attn", "sdpa", "--batch-size", "1", "--logits-dtype", "native"]),
    dict(name="hf_bf16_sdpa_bs1_bi", scorer="hf", toggle="batch-invariant aten ops",
         flags=["--dtype", "bf16", "--attn", "sdpa", "--batch-size", "1", "--batch-invariant"]),
    dict(name="hf_fp16_sdpa_bs1", scorer="hf", toggle="fp16 instead of bf16",
         flags=["--dtype", "fp16", "--attn", "sdpa", "--batch-size", "1"]),

    # --- inference-side baseline and toggles -------------------------------
    dict(name="vllm_bf16_fa", scorer="vllm", toggle="inference baseline", baseline="vllm",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN"]),
    dict(name="vllm_bf16_flashinfer", scorer="vllm", toggle="FlashInfer backend",
         flags=["--dtype", "bfloat16", "--backend", "FLASHINFER"]),
    dict(name="vllm_bf16_triton", scorer="vllm", toggle="Triton attention backend",
         flags=["--dtype", "bfloat16", "--backend", "TRITON_ATTN"]),
    dict(name="vllm_bf16_flex", scorer="vllm", toggle="FlexAttention backend",
         flags=["--dtype", "bfloat16", "--backend", "FLEX_ATTENTION"]),
    dict(name="vllm_bf16_fa_eager", scorer="vllm", toggle="enforce_eager (no cudagraph/compile)",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--enforce-eager"]),
    dict(name="vllm_bf16_fa_bs1", scorer="vllm", toggle="max_num_seqs=1",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--max-num-seqs", "1"]),
    dict(name="vllm_bf16_fa_chunk512", scorer="vllm", toggle="chunked prefill 512",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--max-num-batched-tokens", "512"]),
    dict(name="vllm_bf16_fa_noprefix", scorer="vllm", toggle="prefix caching off",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--no-prefix-caching"]),
    dict(name="vllm_bf16_fa_kvfp8", scorer="vllm", toggle="fp8 KV cache",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--kv-cache-dtype", "fp8"]),
    dict(name="vllm_bf16_fa_bi", scorer="vllm", toggle="batch-invariant mode",
         flags=["--dtype", "bfloat16", "--backend", "FLASH_ATTN", "--batch-invariant"]),
    dict(name="vllm_fp16_fa", scorer="vllm", toggle="fp16 instead of bf16",
         flags=["--dtype", "float16", "--backend", "FLASH_ATTN"]),
    dict(name="vllm_fp32_fa", scorer="vllm", toggle="fp32 weights+activations",
         flags=["--dtype", "float32", "--backend", "FLASH_ATTN"]),
]

# Pairs the report always prints, (train-side, infer-side, description).
HEADLINE_PAIRS = [
    ("hf_bf16_sdpa_bs1", "vllm_sample", "trainer bf16 vs sampler (the RL mismatch)"),
    ("hf_bf16_sdpa_bs1", "vllm_bf16_fa", "trainer bf16 vs vLLM prefill re-score"),
    ("vllm_bf16_fa", "vllm_sample", "vLLM prefill vs vLLM decode (same engine)"),
    ("vllm_bf16_fa_bi", "vllm_sample_bi", "batch-invariant: prefill vs decode"),
    ("hf_fp16_sdpa_bs1", "vllm_fp16_fa", "fp16 both sides"),
    ("hf_fp32_eager", "vllm_fp32_fa", "fp32 both sides (kernel-only residual)"),
    ("hf_bf16_sdpa_bs1_bi", "vllm_bf16_fa_bi", "batch-invariant both sides"),
    ("hf_bf16_sdpa_bs1_fp32head", "vllm_bf16_fa", "fp32 head on trainer only"),
]


def by_name(name):
    for v in VIEWS:
        if v["name"] == name:
            return v
    raise KeyError(name)


def baseline_of(family):
    for v in VIEWS:
        if v.get("baseline") == family:
            return v
    raise KeyError(family)


def scorer_cmd(py, view, model, corpus, out, extra=()):
    mod = {"hf": "mismatch.score_hf", "vllm": "mismatch.score_vllm"}[view["scorer"]]
    cmd = [py, "-m", mod, "--model", model, "--corpus", corpus, "--out", out] + list(view["flags"]) + list(extra)
    if view.get("limit"):
        cmd += ["--limit", str(view["limit"])]
    return cmd
