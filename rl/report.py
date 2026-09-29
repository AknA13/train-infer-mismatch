"""Stage 4 report (CPU): summarise every GRPO arm under $TIM_DATA_ROOT/rl.

Writes results/rl_summary.json and docs/RESULTS_RL.md with, per arm:
  stability   final/peak train accuracy, eval accuracy trajectory, entropy,
              grad-norm spikes, skipped steps, whether the run collapsed
              (accuracy falling > 50% from its peak for >= 10 steps)
  mismatch    KL k3 / |gap| / band fraction at start vs end (drift), sequence
              ESS, fraction of tokens each correction touched (w_mean)
  cost        seconds per step split into generate / train / sync, tokens/s,
              peak GPU memory, monitor overhead (gate M5)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def summarize(arm_dir):
    steps = read_jsonl(arm_dir / "steps.jsonl")
    evals = read_jsonl(arm_dir / "eval.jsonl")
    args = json.loads((arm_dir / "args.json").read_text()) if (arm_dir / "args.json").exists() else {}
    if not steps:
        return None
    # dedupe on resume: keep last record per step
    by = {}
    for s in steps:
        by[s["step"]] = s
    steps = [by[k] for k in sorted(by)]
    acc = np.array([s["acc"] for s in steps]); n = len(steps)
    w = 10
    sm = np.convolve(acc, np.ones(w) / w, mode="valid") if n >= w else acc
    peak = float(sm.max()); peak_i = int(sm.argmax())
    collapsed = bool(n >= w + 10 and (sm[peak_i:] < 0.5 * peak).sum() >= 10 and peak > 0.05)
    kl = [s["mismatch"]["kl_k3"] for s in steps]
    gap = [s["mismatch"]["mean_abs_gap"] for s in steps]
    band = [s["mismatch"]["frac_outside_band"] for s in steps]
    ess = [s["mismatch"]["seq_is_ess_frac"] for s in steps]
    head = slice(0, min(10, n)); tail = slice(max(0, n - 10), n)
    tok = sum(s["n_tokens"] for s in steps); secs = sum(s["t_step"] for s in steps)
    return {
        "arm": arm_dir.name, "mode": args.get("mode"), "dtype": args.get("dtype"), "fp32_head": args.get("fp32_head"),
        "batch_invariant": args.get("batch_invariant"), "steps_done": n, "target_steps": args.get("steps"),
        "done": (arm_dir / "DONE").exists(),
        "train_acc_first10": float(acc[head].mean()), "train_acc_last10": float(acc[tail].mean()),
        "train_acc_peak_smoothed": peak, "collapsed": collapsed,
        "eval": [{"step": e["step"], "acc": e["eval_acc"], "trunc": e["eval_trunc"], "len": e["eval_len"]} for e in evals],
        "entropy_first10": float(np.mean([s["entropy"] for s in steps[head]])),
        "entropy_last10": float(np.mean([s["entropy"] for s in steps[tail]])),
        "trunc_last10": float(np.mean([s["trunc"] for s in steps[tail]])),
        "len_last10": float(np.mean([s["mean_len"] for s in steps[tail]])),
        "grad_norm_median": float(np.median([s["grad_norm"] for s in steps])),
        "grad_norm_max": float(np.max([s["grad_norm"] for s in steps])),
        "skipped_steps": int(sum(s["skipped"] for s in steps)),
        "mismatch": {"kl_k3_first10": float(np.mean(kl[head])), "kl_k3_last10": float(np.mean(kl[tail])),
                     "gap_first10": float(np.mean(gap[head])), "gap_last10": float(np.mean(gap[tail])),
                     "band_first10": float(np.mean(band[head])), "band_last10": float(np.mean(band[tail])),
                     "ess_first10": float(np.mean(ess[head])), "ess_last10": float(np.mean(ess[tail])),
                     "kl_k3_max": float(np.max(kl))},
        "w_mean_last10": float(np.mean([s["w_mean"] for s in steps[tail]])),
        "cost": {"s_per_step": secs / n, "t_gen": float(np.mean([s["t_gen"] for s in steps])),
                 "t_train": float(np.mean([s["t_train"] for s in steps])), "t_sync": float(np.mean([s["t_sync"] for s in steps])),
                 "tokens_per_s": tok / secs, "gpu_mem_gib": float(max(s["gpu_mem_gib"] for s in steps)),
                 "monitor_overhead_frac": (steps[-1]["mismatch"].get("monitor_s", 0.0) / secs) if secs else None},
        "series": {"acc": acc.tolist(), "kl_k3": kl, "gap": gap, "band": band, "ess": ess,
                   "entropy": [s["entropy"] for s in steps], "grad_norm": [s["grad_norm"] for s in steps],
                   "mean_len": [s["mean_len"] for s in steps]},
    }


def f(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}g}" if (abs(x) < 1e-2 or abs(x) >= 1e4) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="")
    a = ap.parse_args()
    dirs = sorted(p for p in C.RL_DIR.glob("*") if p.is_dir())
    if a.arms:
        dirs = [C.RL_DIR / x for x in a.arms.split(",")]
    arms = [s for s in (summarize(d) for d in dirs) if s]
    C.publish_result("rl_summary", {"arms": [{k: v for k, v in s.items() if k != "series"} for s in arms]})
    C.publish_result("rl_series", {"arms": [{"arm": s["arm"], **s["series"]} for s in arms]})
    L = ["# GRPO arms: stability, mismatch drift, cost", "",
         "| arm | mode | dtype | head | BI | steps | acc first10 -> last10 | eval acc first -> last | collapsed | entropy first -> last | gn median / max | skipped |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in arms:
        ev = s["eval"]; e0 = ev[0]["acc"] if ev else None; e1 = ev[-1]["acc"] if ev else None
        L.append(f"| {s['arm']} | {s['mode']} | {s['dtype']} | {s['fp32_head']} | {s['batch_invariant']} | {s['steps_done']}/{s['target_steps']}{' done' if s['done'] else ''} | "
                 f"{f(s['train_acc_first10'])} -> {f(s['train_acc_last10'])} | {f(e0)} -> {f(e1)} | {'YES' if s['collapsed'] else 'no'} | "
                 f"{f(s['entropy_first10'])} -> {f(s['entropy_last10'])} | {f(s['grad_norm_median'])} / {f(s['grad_norm_max'])} | {s['skipped_steps']} |")
    L += ["", "## Mismatch during training (first 10 steps -> last 10)", "",
          "| arm | KL k3 | mean abs gap | frac outside band | seq ESS | max KL k3 | w_mean last10 |", "|---|---|---|---|---|---|---|"]
    for s in arms:
        m = s["mismatch"]
        L.append(f"| {s['arm']} | {f(m['kl_k3_first10'])} -> {f(m['kl_k3_last10'])} | {f(m['gap_first10'])} -> {f(m['gap_last10'])} | "
                 f"{f(m['band_first10'])} -> {f(m['band_last10'])} | {f(m['ess_first10'])} -> {f(m['ess_last10'])} | {f(m['kl_k3_max'])} | {f(s['w_mean_last10'])} |")
    L += ["", "## Cost", "", "| arm | s/step | gen | train | sync | tokens/s | peak GiB | monitor overhead |", "|---|---|---|---|---|---|---|---|"]
    for s in arms:
        c = s["cost"]
        L.append(f"| {s['arm']} | {f(c['s_per_step'])} | {f(c['t_gen'])} | {f(c['t_train'])} | {f(c['t_sync'])} | {f(c['tokens_per_s'], 4)} | {f(c['gpu_mem_gib'])} | {f(c['monitor_overhead_frac'])} |")
    p = C.REPO_ROOT / "docs" / "RESULTS_RL.md"; p.write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"[rl.report] wrote {p}")


if __name__ == "__main__":
    main()
