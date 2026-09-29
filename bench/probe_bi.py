"""Which part of vLLM's batch-invariant mode breaks a transformers forward?

score_hf --batch-invariant gave rms error 0.35 / max 44 nats vs fp32 (10x worse
than plain bf16). On Hopper + torch 2.9, enable_batch_invariant_mode() does
NOT replace matmuls; it (a) sets CUBLAS_WORKSPACE_CONFIG=:16:8 and
CUBLASLT_WORKSPACE_SIZE=1 (env vars, effective only before cuBLAS initialises),
(b) prefers cuBLASLt, (c) disables bf16/fp16 reduced-precision reductions,
(d) overrides aten softmax/_log_softmax/mean.dim/bmm with Triton kernels.

Each configuration runs in its own process (so the env vars take effect) and
scores the first 8 corpus rows through the real model, compared against the
stored fp32 reference view. Arms:
  plain            bf16, no mode
  bi               mode enabled before any CUDA work
  bi_noworkspace   mode enabled, then the two workspace env vars removed before
                   the first cuBLAS call (isolates (a))
  bi_nomean        mode enabled, RMSNorm rewritten as sum(x*x)/H so aten::mean.dim
                   is never called (isolates the mean override)
  bi_nolt          mode enabled, preferred BLAS set back to cublas (isolates (b))
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

ARMS = ["plain", "bi", "bi_noworkspace", "bi_nomean", "bi_nolt"]


def child(arm, model, corpus, ref_path, n_rows):
    import torch
    if arm != "plain":
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        if arm == "bi_noworkspace":
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None); os.environ.pop("CUBLASLT_WORKSPACE_SIZE", None)
        if arm == "bi_nolt":
            torch.backends.cuda.preferred_blas_library(backend="cublas")
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3 as mq
    if arm == "bi_nomean":
        def fwd(self, hidden_states):
            dt = hidden_states.dtype
            h = hidden_states.to(torch.float32)
            var = (h * h).sum(-1, keepdim=True) / h.shape[-1]
            h = h * torch.rsqrt(var + self.variance_epsilon)
            return self.weight * h.to(dt)
        mq.Qwen3RMSNorm.forward = fwd
    from mismatch.common import read_corpus
    from mismatch.store import View
    rows = read_corpus(corpus)[:n_rows]
    ref = View(ref_path)
    model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    import numpy as np
    errs = []
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r["prompt_ids"] + r["completion_ids"]], device="cuda")
            P = len(r["prompt_ids"])
            logits = model(ids).logits[0, P - 1 : -1].float()
            lp = logits.gather(1, ids[0, P:, None]).squeeze(1) - torch.logsumexp(logits, -1)
            errs.append(lp.cpu().numpy() - ref.row(int(r["id"])))
    d = np.concatenate(errs)
    out = {"arm": arm, "n_tokens": int(len(d)), "rms": float(np.sqrt((d ** 2).mean())), "mean_abs": float(np.abs(d).mean()),
           "max_abs": float(np.abs(d).max()), "frac_gt_1": float((np.abs(d) > 1).mean()),
           "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")}}
    print("RESULT " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--child", default="")
    a = ap.parse_args()
    from mismatch.common import corpus_paths
    _, corpus, _ = corpus_paths(a.model)
    ref = C.SCORES_DIR / C.model_tag(a.model) / "hf_fp32_eager.npz"
    if a.child:
        return child(a.child, a.model, str(corpus), str(ref), a.rows)
    results = []
    for arm in ARMS:
        env = dict(os.environ); env.pop("CUBLAS_WORKSPACE_CONFIG", None); env.pop("CUBLASLT_WORKSPACE_SIZE", None)
        p = subprocess.run([sys.executable, "-m", "bench.probe_bi", "--model", a.model, "--rows", str(a.rows), "--child", arm],
                           env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if line:
            r = json.loads(line[-1][7:]); results.append(r)
            print(f"{arm:16s} rms {r['rms']:.4f} mean|d| {r['mean_abs']:.4f} max {r['max_abs']:.2f} frac>1nat {r['frac_gt_1']:.2e} env={r['env']}", flush=True)
        else:
            print(f"{arm:16s} FAILED rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
            results.append({"arm": arm, "failed": True})
    C.publish_result("probe_bi", {"model": a.model, "rows": a.rows, "arms": results})


if __name__ == "__main__":
    main()
