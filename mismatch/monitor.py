"""Drop-in mismatch monitor for an RL step.

    mon = MismatchMonitor()
    stats = mon.update(logp_train, logp_infer, mask)      # torch tensors, any device
    ...
    mon.summary()                                          # running view

Given the trainer's per-token logprobs and the sampler's, on the same tokens,
it returns the numbers that decide whether a correction is needed and which
one would act: KL estimate (k3, valid since x ~ sampler), mean |gap|, band
violations, the fraction of tokens each correction would drop or clip, the
sequence-level IS effective sample size, and a gap histogram bucketed by the
trainer's own probability (low-probability tokens are where the gap lives).

The whole thing is a handful of elementwise ops on tensors the trainer already
has, so it costs a fraction of a percent of a step (gate M5 measures this).
"""
import math
import time

import torch

import config as C


class MismatchMonitor:
    def __init__(self, band=C.RATIO_BAND, clip=C.TIS_CLIP, prob_edges=C.PROB_BUCKETS, hist_bins=41, hist_range=(-1.0, 1.0)):
        self.band, self.clip = band, clip
        self.prob_edges = list(prob_edges)
        self.hist_bins, self.hist_range = hist_bins, hist_range
        self.history = []
        self.overhead_s = 0.0

    @torch.no_grad()
    def update(self, logp_train, logp_infer, mask, step=None):
        t0 = time.perf_counter()
        m = mask.bool()
        a = logp_train.float()[m]; b = logp_infer.float()[m]
        n = int(a.numel())
        out = {"step": step, "n_tokens": n}
        if n == 0:
            return out
        d = a - b
        r = torch.exp(d.clamp(-30, 30))
        lo, hi = self.band
        out.update({
            "kl_k3": float((r - 1 - d).mean()),
            "kl_k1": float((-d).mean()),
            "mean_gap": float(d.mean()), "mean_abs_gap": float(d.abs().mean()),
            "rms_gap": float(d.pow(2).mean().sqrt()),
            "p99_abs_gap": float(torch.quantile(d.abs(), 0.99)) if n < 16_000_000 else float("nan"),
            "max_abs_gap": float(d.abs().max()),
            "frac_outside_band": float(((r < lo) | (r > hi)).float().mean()),
            "frac_tis_clipped": float((r > self.clip).float().mean()),
            "frac_mis_dropped": float(((r < 1 / self.clip) | (r > self.clip)).float().mean()),
            "frac_ppo_clipped_02": float(((r < 0.8) | (r > 1.2)).float().mean()),
            "mean_logp_infer": float(b.mean()), "mean_logp_train": float(a.mean()),
        })
        # sequence-level
        S = ((logp_train.float() - logp_infer.float()) * mask.float()).sum(dim=1)
        w = torch.exp((S - S.max()).clamp(min=-700))
        out["seq_is_ess_frac"] = float(w.sum() ** 2 / (S.numel() * (w * w).sum()))
        out["seq_logratio_std"] = float(S.std()) if S.numel() > 1 else 0.0
        out["frac_seq_weight_gt_clip"] = float((torch.exp(S.clamp(-30, 30)) > self.clip).float().mean())
        # by trainer probability bucket
        p = torch.exp(a).clamp(0.0, 1.0)   # bf16 noise can push a logprob above 0
        buckets = []
        for lo_e, hi_e in zip(self.prob_edges[:-1], self.prob_edges[1:]):
            sel = (p >= lo_e) & (p < hi_e)
            k = int(sel.sum())
            buckets.append({"p": f"[{lo_e:g},{hi_e:g})", "n": k,
                            "mean_abs_gap": float(d[sel].abs().mean()) if k else None,
                            "frac_outside_band": float(((r[sel] < lo) | (r[sel] > hi)).float().mean()) if k else None})
        out["by_train_prob"] = buckets
        out["hist"] = torch.histc(d.clamp(*self.hist_range), bins=self.hist_bins, min=self.hist_range[0], max=self.hist_range[1]).tolist()
        self.overhead_s += time.perf_counter() - t0
        out["monitor_s"] = self.overhead_s
        self.history.append({k: v for k, v in out.items() if k not in ("hist", "by_train_prob")})
        return out

    def summary(self, last=20):
        h = self.history[-last:]
        if not h:
            return {}
        keys = ("kl_k3", "mean_abs_gap", "frac_outside_band", "seq_is_ess_frac")
        return {k: sum(x[k] for x in h) / len(h) for k in keys if k in h[0]}

    def diagnose(self, stats):
        """Plain-language reading of one update() result."""
        msgs = []
        if stats.get("n_tokens", 0) == 0:
            return ["no tokens"]
        kl = stats["kl_k3"]
        msgs.append(f"KL(sampler||trainer) k3 = {kl:.2e}; mean |gap| {stats['mean_abs_gap']:.2e}; "
                    f"{100*stats['frac_outside_band']:.2f}% of tokens outside {self.band}")
        if stats["seq_is_ess_frac"] < 0.5:
            msgs.append(f"sequence-level IS is unusable (ESS {stats['seq_is_ess_frac']:.2f}): use token-level corrections")
        bt = [b for b in stats.get("by_train_prob", []) if b["n"]]
        if bt and bt[0]["mean_abs_gap"] and bt[-1]["mean_abs_gap"] and bt[0]["mean_abs_gap"] > 5 * bt[-1]["mean_abs_gap"]:
            msgs.append("gap concentrated on low-probability tokens: consistent with reduced-precision softmax/head, not attention")
        if abs(stats["mean_gap"]) > 3 * stats["rms_gap"] / math.sqrt(max(stats["n_tokens"], 1)):
            msgs.append(f"systematic bias {stats['mean_gap']:+.2e}: one side is consistently sharper")
        return msgs
