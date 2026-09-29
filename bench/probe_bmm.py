"""Verify WHERE vLLM's batch-invariant bmm breaks a transformers Qwen3 forward.

Hypothesis: Qwen3RotaryEmbedding computes
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float())
i.e. an fp32 aten::bmm with inner dimension K=1. If bmm_batch_invariant mishandles
that shape/dtype, every RoPE angle is wrong for some positions and attention
degrades exactly the way score_hf --batch-invariant showed.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from vllm.model_executor.layers import batch_invariant as B

torch.manual_seed(0)
res = {}
cases = {
    "rope_fp32_K1 [1,64,1]@[1,1,2048]": (torch.randn(1, 64, 1), torch.arange(2048, dtype=torch.float32)[None, None, :]),
    "rope_fp32_K1_bs2 [2,64,1]@[2,1,777]": (torch.randn(2, 64, 1), torch.arange(777, dtype=torch.float32).repeat(2, 1)[:, None, :]),
    "generic_fp32 [4,128,64]@[4,64,256]": (torch.randn(4, 128, 64), torch.randn(4, 64, 256)),
    "generic_bf16 [4,128,64]@[4,64,256]": (torch.randn(4, 128, 64).bfloat16(), torch.randn(4, 64, 256).bfloat16()),
    "K1_bf16 [1,64,1]@[1,1,2048]": (torch.randn(1, 64, 1).bfloat16(), torch.arange(2048, dtype=torch.float32)[None, None, :].bfloat16()),
    "K8_fp32 [1,64,8]@[1,8,2048]": (torch.randn(1, 64, 8), torch.randn(1, 8, 2048)),
}
for name, (a, b) in cases.items():
    a, b = a.cuda(), b.cuda()
    ref = torch.bmm(a.double(), b.double())
    plain = torch.bmm(a, b)
    bi = B.bmm_batch_invariant(a, b)
    e_plain = (plain.double() - ref).abs().max().item()
    e_bi = (bi.double() - ref).abs().max().item()
    res[name] = {"plain_max_err": e_plain, "bi_max_err": e_bi, "ref_scale": ref.abs().max().item()}
    print(f"{name:40s} plain max err {e_plain:.3e}   BI max err {e_bi:.3e}   (|ref| max {ref.abs().max().item():.1f})", flush=True)

# and the real module
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
cfg = AutoConfig.from_pretrained(C.MODEL_ID)
rope = Qwen3RotaryEmbedding(cfg).cuda()
x = torch.zeros(1, 2048, 8, device="cuda", dtype=torch.bfloat16)
pos = torch.arange(2048, device="cuda")[None]
cos0, sin0 = rope(x, pos)
B.enable_batch_invariant_mode()
cos1, sin1 = rope(x, pos)
d = (cos1.float() - cos0.float()).abs()
res["rotary_module"] = {"max_abs_diff_cos": d.max().item(), "frac_positions_changed": (d.amax(-1) > 1e-3).float().mean().item()}
print(f"Qwen3RotaryEmbedding cos: max |diff| {d.max().item():.3e}, positions changed {(d.amax(-1) > 1e-3).float().mean().item():.3f}", flush=True)
C.publish_result("probe_bmm", res)
