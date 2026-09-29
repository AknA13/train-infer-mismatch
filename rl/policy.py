"""Trainer-side policy: low-precision forward/backward, fp32 master weights.

This reproduces what FSDP mixed precision does on one GPU without FSDP:
  - parameters live in bf16 (or fp16) and every forward/backward runs in that
    dtype, exactly like a param_dtype=bf16 MixedPrecisionPolicy;
  - gradients are accumulated into fp32 buffers across micro-batches;
  - AdamW updates fp32 master copies, which are then cast back into the
    low-precision parameters.
So the trainer's logprobs are numerically the "hf_bf16_sdpa" view from the
measurement stage, and the same tensors are what the rollout engine receives
at every weight sync.

Logits are produced per sequence in fp32 (log_softmax in fp32 is what every RL
trainer does) and only the completion positions are kept, so the [T, V] logits
never exist for the whole padded batch at once.
"""
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Policy:
    def __init__(self, model_id, dtype="bf16", lr=2e-6, betas=(0.9, 0.999), weight_decay=0.0,
                 grad_clip=1.0, fp32_head=False, grad_ckpt=True, loss_scale=None, device="cuda"):
        from transformers import AutoModelForCausalLM
        self.dtype = DTYPES[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, attn_implementation="sdpa").to(device)
        self.model.config.use_cache = False
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if fp32_head:
            from mismatch.fp32head import patch_hf
            patch_hf(self.model)
        self.fp32_head = fp32_head
        self.model.train()
        self.named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        self.master = [p.detach().clone().float() for p in self.params]
        self.acc = [torch.zeros_like(m) for m in self.master]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        # fp16 needs loss scaling; bf16 does not. Dynamic: halve on overflow,
        # double after 200 clean steps (the usual GradScaler schedule).
        self.loss_scale = float(loss_scale) if loss_scale else (2.0 ** 12 if self.dtype == torch.float16 else 1.0)
        self._clean_steps = 0
        self.device = device
        self.n_params = sum(p.numel() for p in self.params)

    # ---- forward -------------------------------------------------------------
    def logprobs(self, input_ids, attn_mask, prompt_lens, comp_lens, want_entropy=True):
        """input_ids [B, L] right-padded. Returns logp [B, Tmax], entropy [B, Tmax],
        mask [B, Tmax] over completion tokens (Tmax = max comp_len)."""
        out = self.model.model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        h = out.last_hidden_state
        w = self.model.lm_head.weight
        if self.fp32_head:
            w = w.float()
        B = input_ids.shape[0]
        Tmax = int(max(comp_lens))
        logp = h.new_zeros((B, Tmax), dtype=torch.float32)
        ent = h.new_zeros((B, Tmax), dtype=torch.float32)
        mask = h.new_zeros((B, Tmax), dtype=torch.float32)
        for i in range(B):
            P, T = int(prompt_lens[i]), int(comp_lens[i])
            hi = h[i, P - 1 : P + T - 1]
            logits = (hi @ w.t()).float()
            lse = torch.logsumexp(logits, dim=-1)
            tgt = input_ids[i, P : P + T]
            logp[i, :T] = logits.gather(1, tgt[:, None]).squeeze(1) - lse
            if want_entropy:
                with torch.no_grad():
                    lp_all = logits - lse[:, None]
                    ent[i, :T] = -(lp_all.exp() * lp_all).sum(-1)
            mask[i, :T] = 1.0
        return logp, ent, mask

    # ---- backward / step -------------------------------------------------------
    def backward(self, loss):
        (loss * self.loss_scale).backward()
        with torch.no_grad():
            for p, a in zip(self.params, self.acc):
                if p.grad is not None:
                    a.add_(p.grad.float())
                    p.grad = None

    @torch.no_grad()
    def step(self):
        inv = 1.0 / self.loss_scale
        finite = True
        total = torch.zeros((), device=self.device)
        for a in self.acc:
            a.mul_(inv)
            s = a.pow(2).sum()
            total += s
        gnorm = float(total.sqrt())
        if not math.isfinite(gnorm):
            finite = False
        if finite:
            if self.grad_clip and gnorm > self.grad_clip:
                for a in self.acc:
                    a.mul_(self.grad_clip / (gnorm + 1e-6))
            for m, a in zip(self.master, self.acc):
                m.grad = a
            self.opt.step()
            for m in self.master:
                m.grad = None
            for p, m in zip(self.params, self.master):
                p.copy_(m.to(self.dtype))
            self._clean_steps += 1
            if self.dtype == torch.float16 and self._clean_steps >= 200:
                self.loss_scale *= 2.0; self._clean_steps = 0
        else:
            if self.dtype == torch.float16:
                self.loss_scale = max(self.loss_scale / 2.0, 1.0)
            self._clean_steps = 0
        for a in self.acc:
            a.zero_()
        return {"grad_norm": gnorm, "skipped": not finite, "loss_scale": self.loss_scale}

    def named_weights(self):
        """(hf_name, low-precision tensor) pairs for the rollout engine."""
        return [(n, p.data) for n, p in self.named]

    # ---- checkpoint --------------------------------------------------------------
    def save(self, d, extra):
        d = Path(d); tmp = d.with_name(d.name + ".tmp")
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save({"master": self.master, "opt": self.opt.state_dict(),
                    "loss_scale": self.loss_scale, "clean_steps": self._clean_steps}, tmp / "trainer.pt")
        (tmp / "extra.json").write_text(json.dumps(extra, default=str))
        if d.exists():
            import shutil; shutil.rmtree(d)
        os.replace(tmp, d)

    def load(self, d):
        d = Path(d)
        st = torch.load(d / "trainer.pt", map_location=self.device, weights_only=False)
        with torch.no_grad():
            for m, s in zip(self.master, st["master"]):
                m.copy_(s)
            for p, m in zip(self.params, self.master):
                p.copy_(m.to(self.dtype))
        self.opt.load_state_dict(st["opt"])
        self.loss_scale = st["loss_scale"]; self._clean_steps = st["clean_steps"]
        return json.loads((d / "extra.json").read_text())
