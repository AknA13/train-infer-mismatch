"""Score a corpus through vLLM's PREFILL path: feed prompt+completion as one
token prompt and read prompt_logprobs. Same weights as the sampler, different
kernels (prefill attention, chunked batches, no CUDA-graph decode), which is
exactly the difference between how a rollout engine samples and how a trainer
scores.

Every knob that could plausibly change the numbers is a flag so views.py can
build a one-toggle-at-a-time grid:

  --backend            VLLM_ATTENTION_BACKEND (FLASH_ATTN | FLASHINFER | TRITON_ATTN | FLEX_ATTENTION)
  --dtype              bfloat16 | float16 | float32
  --batch-invariant    VLLM_BATCH_INVARIANT=1 (deterministic reductions)
  --enforce-eager      no CUDA graphs / no torch.compile
  --max-num-seqs       scheduler batch width
  --max-num-batched-tokens   chunked-prefill chunk size
  --kv-cache-dtype     auto | fp8
  --no-prefix-caching

Env vars must be set before vLLM is imported, so this module is always run as
its own process (scripts/02_score.sh does that).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--gmu", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"[score_vllm] {out} exists -- skip", flush=True)
        return
    if args.backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.backend
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    from mismatch.common import read_corpus
    from mismatch.store import save_view
    rows = read_corpus(args.corpus)
    if args.limit:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams, TokensPrompt
    import vllm
    kw = dict(model=args.model, dtype=args.dtype, seed=args.seed,
              gpu_memory_utilization=args.gmu, max_model_len=args.max_model_len,
              enforce_eager=args.enforce_eager, max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=not args.no_prefix_caching,
              kv_cache_dtype=args.kv_cache_dtype, logprobs_mode="raw_logprobs")
    if args.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    t0 = time.time()
    llm = LLM(**kw)
    up = time.time() - t0
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, logprobs=0)

    prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["completion_ids"]) for r in rows]
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    score_s = time.time() - t1

    logp_rows, ids = [], []
    for r, o in zip(rows, outs):
        plp = o.prompt_logprobs
        P, comp = len(r["prompt_ids"]), r["completion_ids"]
        assert plp is not None and len(plp) == P + len(comp), (len(plp) if plp else None, P + len(comp))
        lps = []
        for t, d in zip(comp, plp[P:]):
            lps.append(float(d[t].logprob))
        logp_rows.append(lps); ids.append(r["id"])

    from vllm import envs
    meta = {"family": "vllm", "path": "prefill", "model": args.model,
            "vllm_version": vllm.__version__, "dtype": args.dtype,
            "backend": envs.VLLM_ATTENTION_BACKEND or "default(FLASH_ATTN)",
            "batch_invariant": args.batch_invariant, "enforce_eager": args.enforce_eager,
            "max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": args.max_num_batched_tokens,
            "kv_cache_dtype": args.kv_cache_dtype, "prefix_caching": not args.no_prefix_caching,
            "n_rows": len(ids), "tokens": int(sum(map(len, logp_rows))),
            "startup_s": up, "score_s": score_s}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_view(out, logp_rows, ids, meta)
    print(f"[score_vllm] {out.name}: {len(ids)} rows / {meta['tokens']} tokens "
          f"in {score_s:.1f}s (startup {up:.0f}s) {json.dumps({k: meta[k] for k in ('backend','dtype','batch_invariant','enforce_eager','max_num_seqs','kv_cache_dtype')})}",
          flush=True)


if __name__ == "__main__":
    main()
