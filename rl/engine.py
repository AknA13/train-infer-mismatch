"""In-process vLLM rollout engine colocated with the trainer on one GPU.

VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the engine core in this process, so
weight updates are a device-to-device copy through LLM.apply_model and no
IPC/NCCL plumbing is needed. That is the smallest honest RL loop: the trainer
and the sampler are two different forward implementations of the same weights
in the same process, and everything they disagree on is kernels and dtype.

After every weight update the prefix cache is reset: cached KV blocks were
computed under the old weights and would otherwise leak stale activations
into the next step's prompts (a silent, separate source of "mismatch").
"""
import os
import time

assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") == "0", \
    "set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing rl.engine"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams, TokensPrompt


class RolloutEngine:
    def __init__(self, model, dtype="bfloat16", gmu=0.40, max_model_len=4096, seed=0,
                 prefix_caching=True, fp32_head=False, max_num_seqs=256):
        t0 = time.time()
        self.llm = LLM(model=model, dtype=dtype, seed=seed, gpu_memory_utilization=gmu,
                       max_model_len=max_model_len, enable_prefix_caching=prefix_caching,
                       max_num_seqs=max_num_seqs, logprobs_mode="raw_logprobs")
        self.startup_s = time.time() - t0
        self.fp32_head = fp32_head
        if fp32_head:
            from mismatch.fp32head import patch_vllm
            print("[engine]", self.llm.apply_model(patch_vllm), flush=True)
        self.sync_count = 0

    def generate(self, prompt_ids, n, max_tokens, temperature=1.0, seed=None):
        """Returns list (per prompt) of lists (n) of dicts {ids, logprobs, finish}."""
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, top_k=-1,
                            max_tokens=max_tokens, logprobs=0, seed=seed, skip_special_tokens=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp, use_tqdm=False)
        groups = []
        for o in outs:
            g = []
            for c in o.outputs:
                ids = list(c.token_ids)
                lps = [float(d[t].logprob) for t, d in zip(ids, c.logprobs)]
                g.append({"ids": ids, "logprobs": lps, "finish": c.finish_reason})
            groups.append(g)
        return groups

    def greedy(self, prompt_ids, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp, use_tqdm=False)
        return [{"ids": list(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason} for o in outs]

    FINGERPRINT_NAMES = ("model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight", "model.norm.weight")

    def fingerprint(self):
        """float sums of a few tensors, by HF name, read from inside the engine."""
        names = self.FINGERPRINT_NAMES

        def _fp(model):
            params = dict(model.named_parameters())
            return {n: float(params[n].float().sum()) for n in names if n in params}

        return self.llm.apply_model(_fp)[0]

    @torch.no_grad()
    def sync_weights(self, named_tensors):
        """named_tensors: iterable of (hf_name, tensor) in the engine's dtype."""
        t0 = time.time()
        items = list(named_tensors)

        def _load(model):
            loaded = model.load_weights(iter(items))
            return len(loaded)

        n = self.llm.apply_model(_load)[0]
        self.llm.reset_prefix_cache()
        torch.cuda.synchronize()
        self.sync_count += 1
        return {"sync_s": time.time() - t0, "n_loaded": n}
