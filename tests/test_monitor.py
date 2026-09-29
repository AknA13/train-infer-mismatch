"""MismatchMonitor on synthetic inputs, plus the M5 overhead budget on CPU."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch.monitor import MismatchMonitor

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    mon = MismatchMonitor()
    N, T = 64, 300
    mask = torch.ones(N, T); mask[:, 250:] = 0
    lp_inf = torch.log(torch.rand(N, T) * 0.9 + 0.05)
    s = mon.update(lp_inf, lp_inf, mask, step=0)
    check("identical: zero KL, zero gap", s["kl_k3"] == 0 and s["mean_abs_gap"] == 0 and s["frac_outside_band"] == 0)
    check("token count honours mask", s["n_tokens"] == N * 250)
    noise = torch.randn(N, T) * 0.05
    s = mon.update(lp_inf + noise, lp_inf, mask, step=1)
    check("small noise: k3 ~ var/2", abs(s["kl_k3"] - 0.05 ** 2 / 2) < 5e-4, f"{s['kl_k3']:.5f}")
    check("band fraction tiny for 0.05 noise", s["frac_outside_band"] < 0.01)
    check("by_train_prob partitions", sum(b["n"] for b in s["by_train_prob"]) == s["n_tokens"])
    check("hist bins", len(s["hist"]) == mon.hist_bins and sum(s["hist"]) == s["n_tokens"])
    big = lp_inf.clone(); big[0, :] += 3.0        # one sequence with ratio e^3 per token
    s = mon.update(big, lp_inf, mask, step=2)
    check("sequence ESS collapses on one outlier", s["seq_is_ess_frac"] < 0.05, f"{s['seq_is_ess_frac']:.3f}")
    check("diagnose returns text", isinstance(mon.diagnose(s), list) and mon.diagnose(s))
    check("history/summary", len(mon.history) == 3 and "kl_k3" in mon.summary())
    # M5 on CPU: a 512x1024 update must be a small fraction of a mock 30 s step
    m2 = MismatchMonitor(); a = torch.randn(512, 1024); b = a + torch.randn(512, 1024) * 0.01; mk = torch.ones(512, 1024)
    t0 = time.perf_counter(); m2.update(a, b, mk); dt = time.perf_counter() - t0
    check("M5 overhead small (CPU, 0.5M tokens)", dt < 0.02 * 30, f"{dt:.3f}s")
    print(); print("FAILED" if FAIL else "monitor OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
