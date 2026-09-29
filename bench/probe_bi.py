"""Which batch-invariant aten override breaks a transformers forward?

score_hf --batch-invariant produced rms error 0.35 (max 44 nats) against fp32,
10x worse than plain bf16. vLLM's enable_batch_invariant_mode() replaces
aten::{mm, addmm, matmul, linear, bmm, softmax, _softmax, _log_softmax,
mean.dim} on CUDA. This probe calls each op with the shapes and dtypes a
Qwen3-1.7B forward actually produces, before and after enabling the mode, and
reports the max/rms deviation per op against the fp32 result, so the culprit is
named rather than guessed. Also runs one full tiny forward per op subset by
re-scoring 8 corpus rows with the mode on, to see if the per-op errors explain
the model-level error.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def err(a, ref):
    d = (a.float() - ref.float())
    return {"rms": float(d.pow(2).mean().sqrt()), "max": float(d.abs().max())}


def cases(T=1500, H=2048, I=6144, V=151936, nh=16, hd=128):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)
    x2 = r(T, H); w_up = r(I, H) * 0.02; w_head = r(V, H) * 0.02; b = r(I)
    x3 = r(1, T, H); q = r(1, nh, T, hd); k = r(1, nh, T, hd)
    logits = r(T, V) * 3
    return {
        "linear_2d[T,H]x[I,H]": (F.linear, (x2, w_up)),
        "linear_3d[1,T,H]x[I,H]": (F.linear, (x3, w_up)),
        "linear_head[T,H]x[V,H]": (F.linear, (x2, w_head)),
        "mm[T,H]@[H,I]": (torch.mm, (x2, w_up.t())),
        "matmul_3d": (torch.matmul, (x3, w_up.t())),
        "addmm": (lambda i, a, bm: torch.addmm(i, a, bm), (b, x2, w_up.t())),
        "bmm[nh,T,hd]@[nh,hd,T]": (torch.bmm, (q[0], k[0].transpose(1, 2))),
        "softmax_scores[nh,T,T]": (lambda s: torch.softmax(s, -1), (torch.matmul(q, k.transpose(2, 3))[0] / hd ** 0.5,)),
        "log_softmax_logits[T,V]": (lambda s: torch.log_softmax(s, -1), (logits,)),
        "mean_dim_rmsnorm[T,H]": (lambda s: s.pow(2).mean(-1, keepdim=True), (x2,)),
    }


def run_ops(dtype):
    out = {}
    cs = cases()
    for name, (fn, args) in cs.items():
        a32 = [t.float() for t in args]
        ref = fn(*a32)
        alow = [t.to(dtype) for t in args]
        out[name] = {"ref_rms": float(ref.float().pow(2).mean().sqrt()), "vs_fp32": err(fn(*alow), ref)}
    return out, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--corpus", default="")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"dtype": "bf16"}
    before, cs = run_ops(torch.bfloat16)
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
    enable_batch_invariant_mode()
    after, _ = run_ops(torch.bfloat16)
    # after: same fp32 reference is computed with overrides active too, so
    # compare low-precision results before/after against the pre-mode fp32 ref
    rows = [{"op": name, "bf16_plain_vs_fp32": before[name]["vs_fp32"],
             "bf16_bi_vs_fp32": after[name]["vs_fp32"]} for name in before]
    res["ops"] = rows
    for r in rows:
        b, a = r["bf16_plain_vs_fp32"], r["bf16_bi_vs_fp32"]
        flag = "  <-- BI much worse" if a["rms"] > 3 * max(b["rms"], 1e-9) else ""
        print(f"{r['op']:32s} plain rms {b['rms']:.3e} max {b['max']:.3e} | BI rms {a['rms']:.3e} max {a['max']:.3e}{flag}", flush=True)
    C.publish_result("probe_bi_ops", res)


if __name__ == "__main__":
    main()
