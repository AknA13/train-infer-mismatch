"""GRPO policy-gradient loss with the mismatch corrections under test.

All corrections share one shape: the per-token surrogate is

    L = - sum_t  w_t * A_i * log pi_theta(x_t)   /  (number of tokens)

with w_t a *detached* weight that depends on the ratio

    r_t = pi_theta(x_t) / pi_infer(x_t)

between the trainer's forward and the logprob the sampler reported. On-policy
GRPO with one update per batch has theta == theta_old, so r_t would be exactly
1 if the two engines agreed; every deviation from 1 is train/inference
mismatch, and the corrections differ only in what they do with it.

  none      w = 1                       pretend the sampler was pi_theta
  tis       w = min(r, C)               truncated IS (Yao et al.)
  mis       w = r * 1[1/C <= r <= C]    masked IS: reweight in band, drop outside
  icepop    w = 1[1/C <= r <= C]        mask outliers, no reweighting (Ring-1T)
  seq_tis   w = min(prod_t r_t, C)      one sequence-level weight per completion
  vllm_old  PPO clipped surrogate with pi_old := pi_infer -- the framework bug
            where the sampler's logprobs are used as old logprobs. Its
            gradient is silently zeroed wherever |r-1| > eps, so mismatch
            turns into dropped tokens.

Advantage A_i is per completion (GRPO group normalisation), broadcast over its
tokens. Token-level averaging over the whole batch (DAPO-style) so long
completions are not down-weighted per token.
"""
import torch

MODES = ("none", "tis", "mis", "icepop", "seq_tis", "vllm_old")


def group_advantages(rewards, group_size, eps=1e-6, normalize_std=True):
    """rewards [N] ordered so consecutive group_size entries share a prompt."""
    r = rewards.view(-1, group_size)
    mean = r.mean(dim=1, keepdim=True)
    adv = r - mean
    if normalize_std:
        std = r.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    return adv.reshape(-1)


def correction_weights(mode, logp_train, logp_infer, mask, clip=2.0):
    """Detached per-token weights w_t [N, T] for the modes above (not vllm_old)."""
    with torch.no_grad():
        d = (logp_train - logp_infer) * mask
        r = torch.exp(d.clamp(-20, 20))
        lo, hi = 1.0 / clip, clip
        if mode == "none":
            w = torch.ones_like(r)
        elif mode == "tis":
            w = r.clamp(max=clip)
        elif mode == "mis":
            w = r * ((r >= lo) & (r <= hi)).to(r.dtype)
        elif mode == "icepop":
            w = ((r >= lo) & (r <= hi)).to(r.dtype)
        elif mode == "seq_tis":
            S = d.sum(dim=1, keepdim=True)
            w = torch.exp(S.clamp(-20, 20)).clamp(max=clip).expand_as(r)
        else:
            raise ValueError(mode)
        return w * mask


def grpo_loss(mode, logp_train, logp_infer, adv, mask, n_tokens_total, clip=2.0, ppo_eps=0.2):
    """Returns (loss, stats). loss is already divided by n_tokens_total so that
    summing over micro-batches gives the batch token-mean.

    logp_train [N, T] with grad; logp_infer [N, T] sampler logprobs (no grad);
    adv [N]; mask [N, T] 1 on completion tokens.
    """
    A = adv[:, None]
    if mode == "vllm_old":
        ratio = torch.exp((logp_train - logp_infer).clamp(-20, 20))
        s1 = ratio * A
        s2 = ratio.clamp(1 - ppo_eps, 1 + ppo_eps) * A
        per_tok = -torch.minimum(s1, s2) * mask
        with torch.no_grad():
            clipped = (((ratio < 1 - ppo_eps) & (A < 0)) | ((ratio > 1 + ppo_eps) & (A > 0))) & (mask > 0)
            w_eff = torch.where(clipped, torch.zeros_like(ratio), ratio) * mask
    else:
        w_eff = correction_weights(mode, logp_train, logp_infer, mask, clip)
        per_tok = -(w_eff * A * logp_train) * mask
    loss = per_tok.sum() / max(1, n_tokens_total)
    with torch.no_grad():
        m = mask > 0
        stats = {"w_mean": float(w_eff[m].mean()) if m.any() else 0.0,
                 "w_zero_frac": float((w_eff[m] == 0).float().mean()) if m.any() else 0.0,
                 "w_sum": float(w_eff[m].sum())}
    return loss, stats
