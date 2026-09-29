"""rl/losses.py on synthetic tensors: each correction does what its docstring says."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl import losses as L

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    torch.manual_seed(0)
    # group advantages: zero-mean per group, unit-ish std
    r = torch.tensor([1., 0., 0., 1., 1., 1., 1., 1.])
    adv = L.group_advantages(r, 4)
    check("adv zero mean per group", torch.allclose(adv.view(2, 4).mean(1), torch.zeros(2), atol=1e-6))
    check("adv zero for constant group", torch.all(adv[4:] == 0))

    N, T = 3, 5
    mask = torch.ones(N, T); mask[0, 3:] = 0
    lp_inf = torch.log(torch.rand(N, T) * 0.5 + 0.25)
    d = torch.zeros(N, T); d[1, 0] = torch.log(torch.tensor(3.0)); d[2, 1] = torch.log(torch.tensor(0.25))
    lp_tr = (lp_inf + d).requires_grad_(True)

    w = L.correction_weights("none", lp_tr, lp_inf, mask)
    check("none: weights are the mask", torch.equal(w, mask))
    w = L.correction_weights("tis", lp_tr, lp_inf, mask, clip=2.0)
    check("tis: ratio 3 clipped to 2", abs(float(w[1, 0]) - 2.0) < 1e-6 and abs(float(w[2, 1]) - 0.25) < 1e-6)
    w = L.correction_weights("mis", lp_tr, lp_inf, mask, clip=2.0)
    check("mis: outside band dropped, inside reweighted", float(w[1, 0]) == 0 and float(w[2, 1]) == 0 and abs(float(w[1, 1]) - 1) < 1e-6)
    w = L.correction_weights("icepop", lp_tr, lp_inf, mask, clip=2.0)
    check("icepop: 0/1 weights", set(w.unique().tolist()) <= {0.0, 1.0} and float(w[1, 0]) == 0 and float(w[0, 0]) == 1)
    w = L.correction_weights("seq_tis", lp_tr, lp_inf, mask, clip=2.0)
    check("seq_tis: one weight per sequence", torch.allclose(w[1], torch.full((T,), 2.0)) and abs(float(w[2, 0]) - 0.25) < 1e-6)
    check("weights respect mask", float(w[0, 3]) == 0 and float(w[0, 4]) == 0)

    adv = torch.tensor([1.0, -1.0, 0.5])
    n_tok = int(mask.sum())
    for mode in L.MODES:
        lp = lp_tr.detach().clone().requires_grad_(True)
        loss, st = L.grpo_loss(mode, lp, lp_inf, adv, mask, n_tok)
        loss.backward()
        g = lp.grad
        check(f"{mode}: grad zero on padding", torch.all(g[0, 3:] == 0))
        check(f"{mode}: finite", torch.isfinite(loss).item() and torch.isfinite(g).all().item())
        if mode == "none":
            check("none: grad = -A/n", torch.allclose(g[0, :3], torch.full((3,), -1.0 / n_tok)))
        if mode == "tis":
            check("tis: stats w_mean", abs(st["w_mean"] - float(L.correction_weights("tis", lp, lp_inf, mask)[mask > 0].mean())) < 1e-6)
        if mode == "vllm_old":
            # ratio 3 with A<0: s1 = 3A, clipped = 1.2A; min = 3A (A<0) -> unclipped, gradient flows
            # ratio 0.25 with A>0: s1 = .25A, s2 = .8A; min = .25A -> unclipped
            # ratio 1 tokens: grad = -A * 1 / n
            check("vllm_old: grad -A/n at ratio 1", abs(float(g[0, 0]) + 1.0 / n_tok) < 1e-6)
    print(); print("FAILED" if FAIL else "losses OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
