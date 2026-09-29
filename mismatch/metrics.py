"""Pairwise mismatch statistics on aligned per-token logprob arrays.

Conventions. For a pair (infer, train) of views scored on tokens x_t that were
SAMPLED from the inference distribution q:

  d_t      = log pi_train(x_t) - log pi_infer(x_t)          signed per-token gap
  r_t      = exp(d_t)                                        the IS ratio trainers use
  k3       = mean( r_t - 1 - d_t )                           low-variance estimate of
                                                             KL(q || p) -- valid only
                                                             because x_t ~ q
  k1       = mean( -d_t )                                    the naive KL estimate
  seq S    = sum_t d_t                                       sequence log-ratio
  ESS      = (sum_i w_i)^2 / (n sum_i w_i^2), w_i = exp(S_i) normalised effective
                                                             sample size of sequence IS

When the pair is (view, reference) instead, the same numbers are reported but
they are *errors against fp32*, not KL estimates: the tokens were not sampled
from either side. The report labels the two cases differently.

Everything here is numpy and CPU; tests exercise it with synthetic inputs.
"""
import numpy as np

import config as C


def _q(x, qs=(0.5, 0.9, 0.99, 0.999)):
    if len(x) == 0:
        return {f"p{int(q*1000)/10:g}": float("nan") for q in qs}
    v = np.quantile(x, qs)
    return {f"p{int(q*1000)/10:g}": float(a) for q, a in zip(qs, v)}


def token_stats(lp_a, lp_b, sampled_from_b=True):
    """a = 'train'/candidate, b = 'infer'/sampler (or reference). Returns dict."""
    d = (lp_a - lp_b).astype(np.float64)
    ad = np.abs(d)
    r = np.exp(np.clip(d, -50, 50))
    lo, hi = C.RATIO_BAND
    out = {
        "n_tokens": int(len(d)),
        "mean_gap": float(d.mean()) if len(d) else float("nan"),
        "mean_abs_gap": float(ad.mean()) if len(d) else float("nan"),
        "rms_gap": float(np.sqrt((d ** 2).mean())) if len(d) else float("nan"),
        "abs_gap_quantiles": _q(ad),
        "max_abs_gap": float(ad.max()) if len(d) else float("nan"),
        "frac_exact": float((ad == 0).mean()) if len(d) else float("nan"),
        "frac_outside_band": float(((r < lo) | (r > hi)).mean()) if len(d) else float("nan"),
        "frac_ratio_gt_tis_clip": float((r > C.TIS_CLIP).mean()) if len(d) else float("nan"),
        "frac_ratio_lt_inv_clip": float((r < 1.0 / C.TIS_CLIP).mean()) if len(d) else float("nan"),
    }
    if sampled_from_b:
        out["kl_k3"] = float((r - 1.0 - d).mean()) if len(d) else float("nan")
        out["kl_k1"] = float((-d).mean()) if len(d) else float("nan")
    return out


def sequence_stats(d, rows, n_rows):
    """Sequence-level log-ratio S_i and IS effective sample size."""
    if len(d) == 0:
        return {}
    S = np.zeros(n_rows); T = np.zeros(n_rows)
    np.add.at(S, rows, d); np.add.at(T, rows, 1)
    w = np.exp(np.clip(S - S.max(), -700, 0))
    ess = (w.sum() ** 2) / (n_rows * (w ** 2).sum())
    return {"seq_logratio_mean": float(S.mean()), "seq_logratio_std": float(S.std()),
            "seq_abs_logratio_quantiles": _q(np.abs(S)),
            "seq_geo_ratio_mean": float(np.exp(S / np.maximum(T, 1)).mean()),
            "seq_is_ess_frac": float(ess),
            "frac_seq_weight_gt2": float((np.exp(np.clip(S, -50, 50)) > 2).mean()),
            "frac_seq_weight_lt_half": float((np.exp(np.clip(S, -50, 50)) < 0.5).mean())}


def bucketed(d, key, edges, label):
    """mean |d| and mean d per bucket of `key` (e.g. reference prob, position)."""
    ad = np.abs(d)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        n = int(m.sum())
        out.append({label: f"[{lo:g},{hi:g})", "n": n,
                    "mean_abs_gap": float(ad[m].mean()) if n else float("nan"),
                    "mean_gap": float(d[m].mean()) if n else float("nan"),
                    "frac_outside_band": float(((np.exp(d[m]) < C.RATIO_BAND[0]) | (np.exp(d[m]) > C.RATIO_BAND[1])).mean()) if n else float("nan")})
    return out


def pair_report(lp_a, lp_b, rows, pos, n_rows, p_ref=None, entropy=None, sampled_from_b=True):
    d = (lp_a - lp_b).astype(np.float64)
    rep = {"token": token_stats(lp_a, lp_b, sampled_from_b),
           "sequence": sequence_stats(d, rows, n_rows)}
    if p_ref is not None:
        rep["by_ref_prob"] = bucketed(d, p_ref, np.array(C.PROB_BUCKETS), "p_ref")
    if pos is not None and len(pos):
        top = int(pos.max()) + 1
        edges = np.arange(0, top + C.POS_BUCKET, C.POS_BUCKET)
        rep["by_position"] = bucketed(d, pos, edges, "pos")
    if entropy is not None:
        rep["by_ref_entropy"] = bucketed(d, entropy, np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0, 100.0]), "H_ref")
    # correlation of the gap with confidence: is the trainer/infer disagreement
    # concentrated on tokens the model was unsure about?
    if p_ref is not None and len(d) > 2:
        rep["corr_absgap_logpref"] = float(np.corrcoef(np.abs(d), np.log(np.maximum(p_ref, 1e-12)))[0, 1])
    return rep


def error_decomposition(lp_a, lp_b, lp_ref):
    """Split the a-vs-b gap into each side's error against the reference.
    If errors are independent, var(a-b) ~= var(a-ref) + var(b-ref); the
    covariance term says how much the two sides err in the same direction."""
    ea = (lp_a - lp_ref).astype(np.float64); eb = (lp_b - lp_ref).astype(np.float64)
    if len(ea) < 2:
        return {}
    cov = float(np.cov(ea, eb)[0, 1])
    return {"rms_err_a": float(np.sqrt((ea ** 2).mean())), "rms_err_b": float(np.sqrt((eb ** 2).mean())),
            "mean_err_a": float(ea.mean()), "mean_err_b": float(eb.mean()),
            "cov_err": cov, "corr_err": float(np.corrcoef(ea, eb)[0, 1]),
            "rms_gap": float(np.sqrt(((ea - eb) ** 2).mean())),
            "share_var_a": float((ea ** 2).mean() / max(((ea - eb) ** 2).mean(), 1e-30)),
            "share_var_b": float((eb ** 2).mean() / max(((ea - eb) ** 2).mean(), 1e-30))}
