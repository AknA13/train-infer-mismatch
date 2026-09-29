"""Stage 2 driver: score every view in views.VIEWS for one model, each in its
own subprocess (vLLM reads env vars at import; two engines in one process is
asking for trouble). Skips views whose .npz exists, so a preempted job resumes
where it stopped.

  python -m mismatch.run_views --model Qwen/Qwen3-1.7B
  python -m mismatch.run_views --model ... --only vllm_bf16_fa,hf_fp32_eager
  python -m mismatch.run_views --model ... --variant bi   # score the BI corpus
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch import views as V
from mismatch.common import corpus_paths

# Views scored on the batch-invariant corpus: only the ones that pairing needs.
BI_VIEWS = ["vllm_bf16_fa_bi", "hf_bf16_sdpa_bs1_bi", "hf_bf16_sdpa_bs1", "hf_fp32_eager"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--gmu", default="0.80")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    py = os.environ.get("TIM_PY_RESOLVED", sys.executable)
    _, corpus, meta = corpus_paths(args.model, args.variant)
    if not corpus.exists():
        sys.exit(f"[views] corpus missing: {corpus} -- run stage 1 first")
    out_dir = C.SCORES_DIR / C.model_tag(args.model) / (args.variant or "")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [v["name"] for v in V.VIEWS]
    if args.variant == "bi":
        names = BI_VIEWS
    if args.only:
        names = [n for n in args.only.split(",") if n]
    skip = set(args.skip.split(",")) if args.skip else set()

    done, failed = [], []
    for name in names:
        if name in skip:
            continue
        view = V.by_name(name)
        out = out_dir / f"{name}.npz"
        if out.exists():
            print(f"[views] {name}: exists -- skip", flush=True)
            done.append(name); continue
        extra = []
        if view["scorer"] == "vllm":
            extra += ["--gmu", args.gmu]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        cmd = V.scorer_cmd(py, view, args.model, str(corpus), str(out), extra)
        env = dict(os.environ)
        env.pop("VLLM_ATTENTION_BACKEND", None); env.pop("VLLM_BATCH_INVARIANT", None)
        print(f"\n[views] ===== {name} ({view['toggle']}) =====\n+ {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, env=env)
        dt = time.time() - t0
        if rc == 0 and out.exists():
            print(f"[views] {name}: ok in {dt:.0f}s", flush=True); done.append(name)
        else:
            print(f"[views] {name}: FAILED rc={rc} after {dt:.0f}s", flush=True); failed.append(name)
    print(f"\n[views] done={len(done)} failed={len(failed)} {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
