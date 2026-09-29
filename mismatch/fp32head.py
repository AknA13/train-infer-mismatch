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
