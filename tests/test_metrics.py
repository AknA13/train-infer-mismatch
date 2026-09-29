"""metrics.py on synthetic data: the estimators must recover known values."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch import metrics as M

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    rng = np.random.default_rng(0)
    # identical views -> zero everywhere
    x = rng.normal(-2, 1, 10000).astype(np.float32)
    t = M.token_stats(x, x)
    check("identical: zero gap", t["mean_abs_gap"] == 0 and t["frac_exact"] == 1.0 and t["frac_outside_band"] == 0)
    check("identical: k3 zero", abs(t["kl_k3"]) < 1e-12)

    # k3 recovers KL between two Gaussians' worth of log-ratio? Use a discrete
    # example instead: sample x ~ q over 5 symbols, compare against p.
    q = np.array([0.5, 0.2, 0.15, 0.1, 0.05]); p = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    n = 400000
    xs = rng.choice(5, size=n, p=q)
    lq, lp = np.log(q[xs]), np.log(p[xs])
    kl_true = float((q * np.log(q / p)).sum())
    t = M.token_stats(lp.astype(np.float32), lq.astype(np.float32), sampled_from_b=True)
    check("k3 estimates KL(q||p)", abs(t["kl_k3"] - kl_true) < 0.003, f"k3={t['kl_k3']:.4f} true={kl_true:.4f}")
    check("k1 estimates KL(q||p)", abs(t["kl_k1"] - kl_true) < 0.01, f"k1={t['kl_k1']:.4f}")

    # band fractions: ratio exactly at 1.5 is outside [0.8,1.25]
    a = np.zeros(10, np.float32); b = a - np.log(1.5).astype(np.float32)
    t = M.token_stats(a, b)
    check("band violation counted", t["frac_outside_band"] == 1.0)
    check("tis clip fraction", t["frac_ratio_gt_tis_clip"] == 0.0)

    # sequence stats: two rows, one with big positive S -> ESS drops
    d = np.array([0.0] * 5 + [1.0] * 5); rows = np.array([0] * 5 + [1] * 5)
    s = M.sequence_stats(d, rows, 2)
    w = np.array([1.0, np.exp(5)]); ess = w.sum() ** 2 / (2 * (w ** 2).sum())
    check("sequence ESS", abs(s["seq_is_ess_frac"] - ess) < 1e-9, f"{s['seq_is_ess_frac']:.4f} vs {ess:.4f}")
    check("sequence log-ratio mean", abs(s["seq_logratio_mean"] - 2.5) < 1e-9)

    # buckets partition the tokens
    key = rng.uniform(0, 1, 1000); dd = rng.normal(0, 0.1, 1000)
    b = M.bucketed(dd, key, np.array(C.PROB_BUCKETS), "p")
    check("buckets partition", sum(x["n"] for x in b) == 1000)

    # error decomposition identity: rms_gap^2 = ea^2 + eb^2 - 2 cov (population)
    ref = rng.normal(-3, 1, 5000); ea = rng.normal(0, 0.05, 5000); eb = rng.normal(0, 0.03, 5000)
    dec = M.error_decomposition(ref + ea, ref + eb, ref)
    lhs = dec["rms_gap"] ** 2
    rhs = dec["rms_err_a"] ** 2 + dec["rms_err_b"] ** 2 - 2 * np.mean(ea * eb)
    check("decomposition identity", abs(lhs - rhs) < 1e-9, f"{lhs:.6f} vs {rhs:.6f}")
    check("independent errors -> small corr", abs(dec["corr_err"]) < 0.05, f"corr={dec['corr_err']:.3f}")

    # pair_report runs end to end with all optional inputs
    pos = np.tile(np.arange(500), 10); rows = np.repeat(np.arange(10), 500)
    rep = M.pair_report(ref, ref + eb, rows, pos, 10, p_ref=np.exp(ref), entropy=np.abs(ref))
    check("pair_report keys", all(k in rep for k in ("token", "sequence", "by_ref_prob", "by_position", "by_ref_entropy")))
    print(); print("FAILED" if FAIL else "metrics OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
