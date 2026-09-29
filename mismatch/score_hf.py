"""Score a corpus through a transformers forward -- the TRAINER side.

The reference view is fp32 + eager attention + TF32 off + batch size 1: no
fused attention kernel, no reduced-precision accumulation, no padding. Its own
error is measured against an fp64 pass on a subset (gate M1), because a
"ground truth" nobody checked is just another view.

Toggles mirror what real RL trainers do differently from each other:

  --dtype bf16|fp16|fp32|fp64     parameter/activation dtype
  --attn sdpa|eager               fused SDPA kernel vs explicit softmax(QK^T)V
  --batch-size N                  right-padded batches (padding changes kernel
                                  tiling and therefore reduction order)
  --fp32-head                     final norm + lm_head computed in fp32 (the
                                  MiniMax-M1 fix)
  --tf32                          allow TF32 matmuls in fp32 runs
  --batch-invariant               apply vLLM's batch-invariant aten overrides
                                  to this process too
  --logits-dtype fp32|native      log_softmax in fp32 (what every trainer does)
                                  or in the model dtype

Reference runs also store per-token entropy and a top-1 flag so metrics can be
bucketed by how confident the model was.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mismatch.fp32head import Fp32Norm

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--fp32-head", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--batch-invariant", action="store_true")
    ap.add_argument("--rope-no-bmm", action="store_true", help="compute RoPE angles without bmm (fix for --batch-invariant)")
    ap.add_argument("--logits-dtype", default="fp32", choices=["fp32", "native"])
    ap.add_argument("--save-extra", action="store_true", help="store entropy/top1 (reference view)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def score_batch(model, batch_ids, prompt_lens, dtype, fp32_head, logits_fp32, want_extra):
    """batch_ids: list of full token lists. Returns per-row (logp, entropy, top1) for completion tokens."""
    device = next(model.parameters()).device
    L = max(len(x) for x in batch_ids)
    pad = 0
    inp = torch.full((len(batch_ids), L), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long, device=device)
    for i, x in enumerate(batch_ids):
        inp[i, : len(x)] = torch.tensor(x, device=device)
        mask[i, : len(x)] = 1
    out = model.model(input_ids=inp, attention_mask=mask, use_cache=False)
    h = out.last_hidden_state                                   # [B, L, H] post final norm
    head = model.lm_head
    # With --fp32-head the final norm was wrapped to emit fp32 (see main), so h
    # is already fp32 here and the head matmul below runs in fp32 too.
    w = head.weight.float() if fp32_head else head.weight
    results = []
    for i, x in enumerate(batch_ids):
        P, T = prompt_lens[i], len(x)
        # logits for positions P-1 .. T-2 predict tokens P .. T-1
        hi = h[i, P - 1 : T - 1]
        logits = hi @ w.t()
        if logits_fp32:
            logits = logits.float()
        lse = torch.logsumexp(logits, dim=-1)
        tgt = torch.tensor(x[P:], device=device)
        tok = logits.gather(1, tgt[:, None]).squeeze(1)
        logp = (tok - lse)
        ent = top1 = None
        if want_extra:
            lp_all = logits - lse[:, None]
            ent = -(lp_all.exp() * lp_all).sum(-1)
            top1 = (logits.argmax(-1) == tgt).to(torch.float32)
        results.append((logp.float().cpu().numpy(),
                        None if ent is None else ent.float().cpu().numpy(),
                        None if top1 is None else top1.cpu().numpy()))
    return results


def main():
    args = parse()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"[score_hf] {out} exists -- skip", flush=True)
        return
    from mismatch.common import read_corpus
    from mismatch.store import save_view
    rows = read_corpus(args.corpus)
    if args.limit:
        rows = rows[: args.limit]

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    if args.batch_invariant:
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
        enable_batch_invariant_mode()
    if args.rope_no_bmm:
        from mismatch.fp32head import patch_rope_no_bmm
        patch_rope_no_bmm()

    from transformers import AutoModelForCausalLM
    dtype = DTYPES[args.dtype]
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn, device_map=args.device)
    model.eval()
    if args.fp32_head:
        model.model.norm = Fp32Norm(model.model.norm)
    load_s = time.time() - t0

    # sort by length for tight batches, restore corpus order at the end
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt_ids"]) + len(rows[i]["completion_ids"]))
    res = [None] * len(rows)
    t1 = time.time()
    n_tok = 0
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        full = [rows[i]["prompt_ids"] + rows[i]["completion_ids"] for i in idx]
        plen = [len(rows[i]["prompt_ids"]) for i in idx]
        outs = score_batch(model, full, plen, dtype, args.fp32_head,
                           args.logits_dtype == "fp32", args.save_extra)
        for i, o in zip(idx, outs):
            assert len(o[0]) == len(rows[i]["completion_ids"])
            res[i] = o; n_tok += len(o[0])
        if (s // args.batch_size) % 50 == 0:
            print(f"[score_hf] {s + len(idx)}/{len(rows)} rows, {n_tok} tokens, "
                  f"{time.time()-t1:.0f}s", flush=True)
    score_s = time.time() - t1

    extra = None
    if args.save_extra:
        extra = {"entropy": [r[1] for r in res], "top1": [r[2] for r in res]}
    meta = {"family": "hf", "model": args.model, "dtype": args.dtype, "attn": args.attn,
            "batch_size": args.batch_size, "fp32_head": args.fp32_head, "tf32": args.tf32,
            "batch_invariant": args.batch_invariant, "rope_no_bmm": args.rope_no_bmm, "logits_dtype": args.logits_dtype,
            "n_rows": len(rows), "tokens": n_tok, "load_s": load_s, "score_s": score_s,
            "torch": torch.__version__}
    import transformers
    meta["transformers"] = transformers.__version__
    out.parent.mkdir(parents=True, exist_ok=True)
    save_view(out, [r[0] for r in res], [r["id"] for r in rows], meta, extra)
    print(f"[score_hf] {out.name}: {len(rows)} rows / {n_tok} tokens in {score_s:.1f}s "
          f"{json.dumps({k: meta[k] for k in ('dtype','attn','batch_size','fp32_head','tf32','batch_invariant')})}",
          flush=True)


if __name__ == "__main__":
    main()
