"""rl/policy.py on a tiny random Qwen3 on CPU: logprob shift, fp32-master
update path, checkpoint round trip, and that named_weights carries HF names."""
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from rl.policy import Policy
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=257, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True)
    with tempfile.TemporaryDirectory() as d:
        Qwen3ForCausalLM(cfg).save_pretrained(Path(d) / "tiny")
        # bf16 on CPU works for sdpa math path; keep it small
        pol = Policy(str(Path(d) / "tiny"), dtype="bf16", lr=1e-3, grad_ckpt=False, device="cpu")
        ids = torch.randint(1, 257, (3, 20)); am = torch.ones_like(ids)
        plen, clen = [5, 8, 3], [15, 12, 17]
        lp, ent, mask = pol.logprobs(ids, am, plen, clen)
        check("shapes", lp.shape == (3, 17) and mask.sum().item() == sum(clen))
        # direct check of the shift on row 0 with an fp32 copy of the model
        ref = Qwen3ForCausalLM.from_pretrained(str(Path(d) / "tiny"), dtype=torch.float32).eval()
        with torch.no_grad():
            full = torch.log_softmax(ref(ids[:1]).logits[0], -1)
            direct = full[4:19].gather(1, ids[0, 5:20, None]).squeeze(1)
        err = float((lp[0, :15].detach() - direct).abs().max())
        check("shift matches fp32 direct (bf16 noise)", err < 0.1, f"max abs {err:.3f}")
        check("entropy positive", bool((ent[mask > 0] > 0).all()))
        # a step moves the params and keeps master/param in sync
        before = [p.detach().clone() for p in pol.params]
        loss = -(lp * mask).sum() / mask.sum()
        pol.backward(loss); st = pol.step()
        moved = any(not torch.equal(a, b) for a, b in zip(before, pol.params))
        check("step moves params", moved and not st["skipped"] and st["grad_norm"] > 0)
        insync = all(torch.equal(p, m.to(p.dtype)) for p, m in zip(pol.params, pol.master))
        check("params == cast(master) after step", insync)
        check("acc grads zeroed", all(float(a.abs().sum()) == 0 for a in pol.acc))
        names = dict(pol.named_weights())
        check("HF names for engine sync", "model.embed_tokens.weight" in names and "model.layers.0.self_attn.q_proj.weight" in names)
        check("tied: no separate lm_head param", "lm_head.weight" not in names)
        # checkpoint round trip
        pol.save(Path(d) / "ckpt", {"step": 7, "cursor": 3, "rng": [1, 2]})
        pol2 = Policy(str(Path(d) / "tiny"), dtype="bf16", lr=1e-3, grad_ckpt=False, device="cpu")
        extra = pol2.load(Path(d) / "ckpt")
        same = all(torch.equal(a, b) for a, b in zip(pol.params, pol2.params))
        check("checkpoint round trip", same and extra["step"] == 7)
        # fp32 head keeps HF parameter names for the engine loader
        pol4 = Policy(str(Path(d) / "tiny"), dtype="bf16", lr=1e-3, grad_ckpt=False, device="cpu", fp32_head=True)
        n4 = dict(pol4.named_weights())
        check("fp32_head: model.norm.weight name preserved", "model.norm.weight" in n4 and not any(".norm.norm." in k for k in n4))
        lp4, _, m4 = pol4.logprobs(ids, am, plen, clen)
        check("fp32_head: logprobs close to plain", float((lp4.detach() - lp.detach()).abs().max()) < 0.05)
        # fp16 path: loss scaling + overflow skip
        pol3 = Policy(str(Path(d) / "tiny"), dtype="fp16", lr=1e-3, grad_ckpt=False, device="cpu")
        lp3, _, m3 = pol3.logprobs(ids, am, plen, clen)
        pol3.backward(-(lp3 * m3).sum()); st3 = pol3.step()
        check("fp16 step finite", not st3["skipped"], f"scale={st3['loss_scale']}")
        pol3.acc[0].fill_(float("inf")); st4 = pol3.step()
        check("fp16 overflow skipped and scale halved", st4["skipped"] and st4["loss_scale"] == st3["loss_scale"] / 2)
    print(); print("FAILED" if FAIL else "policy tiny OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
