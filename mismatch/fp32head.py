"""fp32 "island" at the top of the network: final RMSNorm + lm_head in fp32.

MiniMax-M1 traced its train/inference probability gap to the LM head and fixed
it by computing the head in fp32 on both sides. This module gives the same
change for a transformers model (wrap the final norm so it emits fp32 and use
an fp32 view of the head weight) and for a vLLM model (patch the logits
processor's matmul).
"""
import torch


class Fp32Norm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        n = self.norm
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        return n.weight.float() * (x * torch.rsqrt(var + n.variance_epsilon))


def patch_hf(model):
    if not isinstance(model.model.norm, Fp32Norm):
        model.model.norm = Fp32Norm(model.model.norm)
    return model


def patch_vllm(model):
    """Called inside the vLLM worker via LLM.apply_model. compute_logits runs
    outside the captured CUDA graph in V1, so a Python-level patch takes effect."""
    lp = model.logits_processor
    org_vocab = lp.org_vocab_size

    def _get_logits(hidden_states, lm_head, embedding_bias):
        logits = torch.matmul(hidden_states.float(), lm_head.weight.float().t())
        if embedding_bias is not None:
            logits = logits + embedding_bias.float()
        return logits[..., :org_vocab]

    lp._get_logits = _get_logits
    return "fp32-head patched"


def patch_rope_no_bmm():
    """Make transformers' Qwen3 rotary embedding compute its angles as an
    elementwise outer product instead of an fp32 bmm with inner dim 1.

    Needed under vLLM's batch-invariant mode: its aten::bmm override runs fp32
    inputs at ~TF32 precision (bench/probe_bmm.py), and RoPE angles reach
    ~7e3 rad, so the override puts 99.9% of positions at the wrong angle."""
    from transformers.models.qwen3 import modeling_qwen3 as mq

    def _rope_no_bmm(self, x, position_ids):
        inv = self.inv_freq.float()[None, :, None].expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv * pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)

    mq.Qwen3RotaryEmbedding.forward = _rope_no_bmm
    return "rope patched (no bmm)"
