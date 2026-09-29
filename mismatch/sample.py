"""Stage 1: sample an RL-style corpus from the policy with vLLM and record the
sampler's own logprob for every generated token.

This is the "inference side" of the mismatch. Everything downstream re-scores
exactly these token sequences, so the corpus is written once and never
regenerated (idempotent: exits if samples.jsonl is complete).

The sampler logprob comes from vLLM's decode path -- CUDA graphs, decode
attention kernels, whatever batch the scheduler happened to build. That is the
number an RL framework would hand the trainer as pi_old, and it is the number
the trainer's forward will disagree with.

Sampling is the policy's own distribution (T=1, no top-p/top-k). Qwen3 ships
top_k=20/top_p=0.95 in generation_config and vLLM applies those defaults
silently; we override them so the returned logprob is log pi(x), not the
logprob under a truncated distribution.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch.common import build_prompt, corpus_paths, load_problems, verify_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--n", type=int, default=C.SAMPLE_N)
    ap.add_argument("--max-tokens", type=int, default=C.SAMPLE_MAX_TOKENS)
    ap.add_argument("--dataset", default=C.SAMPLE_DATASET)
    ap.add_argument("--split", default=C.SAMPLE_SPLIT)
    ap.add_argument("--seed", type=int, default=C.SAMPLE_SEED)
    ap.add_argument("--gmu", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--variant", default="", help="corpus variant suffix, e.g. 'bi' for a batch-invariant sampler")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir, out_path, meta_path = corpus_paths(args.model, args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and meta_path.exists() and not args.overwrite:
        n_have = sum(1 for _ in open(out_path))
        if n_have >= args.n:
            print(f"[sample] {out_path} complete ({n_have} rows) -- skip", flush=True)
            return
        print(f"[sample] {out_path} has {n_have}/{args.n} rows -- regenerating", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tok = AutoTokenizer.from_pretrained(args.model)
    problems = load_problems(args.dataset, args.split, n=args.n, seed=args.seed)
    prompts = [build_prompt(tok, p["problem"], thinking=not args.no_thinking) for p in problems]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    print(f"[sample] {len(problems)} prompts, mean prompt len "
          f"{sum(map(len, prompt_ids))/len(prompt_ids):.0f} tokens", flush=True)

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gmu, max_model_len=args.max_model_len,
              enable_prefix_caching=True, logprobs_mode="raw_logprobs")
    sp = SamplingParams(temperature=C.SAMPLE_TEMPERATURE, top_p=C.SAMPLE_TOP_P,
                        top_k=C.SAMPLE_TOP_K, max_tokens=args.max_tokens, seed=args.seed,
                        logprobs=0, skip_special_tokens=False)
    print(f"[sample] engine up in {time.time()-t0:.0f}s; generating", flush=True)
    t1 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp)
    gen_s = time.time() - t1

    n_tok = 0
    rows = []
    for i, (prob, pids, o) in enumerate(zip(problems, prompt_ids, outs)):
        c = o.outputs[0]
        ids = list(c.token_ids)
        lps, ranks = [], []
        for t, d in zip(ids, c.logprobs):
            lp = d[t]
            lps.append(float(lp.logprob))
            ranks.append(int(lp.rank) if lp.rank is not None else -1)
        assert len(lps) == len(ids), (len(lps), len(ids))
        n_tok += len(ids)
        text = tok.decode(ids, skip_special_tokens=True)
        rows.append({"id": i, "idx": prob["idx"], "source": prob["source"],
                     "prompt_ids": pids, "completion_ids": ids,
                     "sampler_logprobs": lps, "sampler_ranks": ranks,
                     "finish_reason": c.finish_reason, "gold": prob["answer"],
                     "correct": bool(verify_answer(text, prob["answer"]))})

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)

    import vllm
    from vllm import envs
    meta = {"model": args.model, "n": len(rows), "tokens": n_tok,
            "mean_len": n_tok / max(1, len(rows)),
            "finish": {k: sum(r["finish_reason"] == k for r in rows) for k in ("stop", "length")},
            "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
            "sampling": {"temperature": C.SAMPLE_TEMPERATURE, "top_p": C.SAMPLE_TOP_P,
                         "top_k": C.SAMPLE_TOP_K, "max_tokens": args.max_tokens,
                         "seed": args.seed, "thinking": not args.no_thinking},
            "vllm": {"version": vllm.__version__, "dtype": "bfloat16",
                     "attention_backend": envs.VLLM_ATTENTION_BACKEND or "default(FLASH_ATTN)",
                     "batch_invariant": bool(int(os.environ.get("VLLM_BATCH_INVARIANT", "0"))),
                     "logprobs_mode": "raw_logprobs", "prefix_caching": True},
            "gen_seconds": gen_s, "tok_per_s": n_tok / max(gen_s, 1e-9),
            "dataset": f"{args.dataset}/{args.split}"}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[sample] wrote {out_path}: {len(rows)} rows, {n_tok} tokens "
          f"({meta['mean_len']:.0f}/seq), acc={meta['accuracy']:.3f}, "
          f"length-capped={meta['finish']['length']}, {meta['tok_per_s']:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
