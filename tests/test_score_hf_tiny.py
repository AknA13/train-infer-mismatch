"""End-to-end CPU test of score_hf on a tiny random Qwen3.

Checks the thing that is easiest to get subtly wrong: the shift. Position P-1
predicts token P. We compare score_hf's gathered logprobs against a direct
full-vocab log_softmax on a per-sequence forward, and check fp32 eager vs sdpa
agree to fp32 noise, and that --fp32-head on an fp32 model is a no-op.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mismatch.store import View

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def main():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=257, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=False)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        model = Qwen3ForCausalLM(cfg).eval()
        model.save_pretrained(d / "tiny")
        rng = np.random.default_rng(0)
        rows = []
        for i in range(6):
            P, T = int(rng.integers(3, 8)), int(rng.integers(5, 30))
            ids = rng.integers(1, 257, size=P + T).tolist()
            rows.append({"id": i, "prompt_ids": ids[:P], "completion_ids": ids[P:], "sampler_logprobs": [0.0] * T})
        corpus = d / "samples.jsonl"
        corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        # direct reference: full log_softmax per sequence
        direct = []
        with torch.no_grad():
            for r in rows:
                x = torch.tensor([r["prompt_ids"] + r["completion_ids"]])
                lp = torch.log_softmax(model(x).logits.float()[0], -1)
                P = len(r["prompt_ids"])
                tgt = torch.tensor(r["completion_ids"])
                direct.append(lp[P - 1:-1].gather(1, tgt[:, None]).squeeze(1).numpy())

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        def run(out, *flags):
            cmd = [sys.executable, "-m", "mismatch.score_hf", "--model", str(d / "tiny"), "--corpus", str(corpus),
                   "--out", str(d / out), "--device", "cpu", *flags]
            rc = subprocess.call(cmd, env=env, cwd=str(REPO), stdout=subprocess.DEVNULL)
            assert rc == 0, cmd
            return View(d / out)
        v_eager = run("eager.npz", "--dtype", "fp32", "--attn", "eager", "--batch-size", "1", "--save-extra")
        v_sdpa = run("sdpa.npz", "--dtype", "fp32", "--attn", "sdpa", "--batch-size", "4")
        v_head = run("head.npz", "--dtype", "fp32", "--attn", "eager", "--batch-size", "1", "--fp32-head")

        check("row count / lengths", v_eager.n_rows == 6 and v_eager.lengths().tolist() == [len(r["completion_ids"]) for r in rows])
        err = max(float(np.abs(v_eager.row(i) - direct[i]).max()) for i in range(6))
        check("shift matches direct log_softmax", err < 1e-4, f"max abs {err:.2e}")
        err2 = max(float(np.abs(v_sdpa.row(i) - direct[i]).max()) for i in range(6))
        check("sdpa + padded batch agrees at fp32", err2 < 1e-3, f"max abs {err2:.2e}")
        err3 = max(float(np.abs(v_head.row(i) - direct[i]).max()) for i in range(6))
        check("fp32-head is a no-op at fp32", err3 < 1e-4, f"max abs {err3:.2e}")
        ent = v_eager.extra["entropy"]
        check("entropy stored and positive", len(ent) == sum(v_eager.lengths()) and (ent > 0).all())
        top1 = v_eager.extra["top1"]
        check("top1 flag is 0/1", set(np.unique(top1).tolist()) <= {0.0, 1.0})
    print(); print("FAILED" if FAIL else "score_hf tiny OK"); return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
