# Results: where the train/inference mismatch comes from, and what each fix buys

Model Qwen3-1.7B, vLLM 0.12.0 (sampler) vs transformers 4.57.6 (trainer), one
H200. Measurement corpus: 512 gsm8k-train completions sampled at T=1 with no
truncation, 771,300 tokens (mean 1506/seq, thinking mode). RL: single-GPU GRPO
with an in-process vLLM engine, 32 prompts x 8 samples per step, lr 2e-6, 200
steps, weights pushed to the engine after every step. Full tables:
`RESULTS_MEASURE_Qwen3-1p7B.md`, `RESULTS_RL.md`; figures in `fig/`.

## 1. The gap is real and it is rounding, not kernels or scheduling

Reference = HF fp32, eager attention, TF32 off, batch 1. It agrees with an fp64
pass to 3e-6 nats per token (gate M1). Against it:

| view | rms error (nats/token) |
|---|---|
| trainer, bf16 SDPA (what RL frameworks do) | 0.0346 |
| vLLM bf16, FlashAttention, CUDA graphs (the sampler) | 0.0298 |
| vLLM bf16 with `enforce_eager` | 0.0348 |
| vLLM fp32 (Triton backend) | 0.0024 |
| trainer fp16 / vLLM fp16 | 0.0043 / 0.0038 |
| trainer bf16 + fp32 norm and lm_head | 0.0252 |
| vLLM bf16 + fp8 KV cache | 0.178 |

Trainer-vs-sampler on the same tokens: mean |gap| 0.014, rms 0.045, KL(k3) 1.0e-3,
0.8% of tokens outside the [0.8, 1.25] ratio band, 4190x the reference noise
(gate M2). Sequence-level importance weights are unusable at these lengths:
effective sample size 5%, p99 |sum of log-ratios| = 7 nats.

Things that do **not** move the number (all within 3e-4 rms of each other):
attention backend (FlashAttention, FlashInfer, Triton, FlexAttention), scheduler
batch width 1 vs 256, chunked-prefill size, prefix caching. vLLM's
`VLLM_BATCH_INVARIANT=1` makes prefill and decode **bit-identical** (gap exactly
0 on the batch-invariant corpus) and does not touch the trainer-vs-sampler gap
at all (delta 3e-9).

The two stacks' errors are independent: measured gap 0.0449 vs
sqrt(0.0346^2 + 0.0298^2) = 0.0457, corr(err_trainer, err_vllm) = 0.035 (gate M3).
With both sides at fp32 the residual, kernel-only gap is 0.0024, 19x smaller.
So the mismatch is two separate bf16 rounding processes, and any fix has to
act on precision, not on which attention kernel runs.

Where it lives: tokens the model gave < 1% probability have a mean |gap| of
0.12 and 25% of them fall outside the trust band; tokens above 90% have 0.001.
The gap is flat across positions 0 to 2048.

Two surprises. (1) vLLM with CUDA graphs and torch.compile is *more* accurate
than the transformers trainer (0.030 vs 0.035); with `enforce_eager` it matches
the trainer exactly, so the fused kernels help. (2) Computing log-softmax in bf16
instead of fp32 adds a systematic +0.005 bias (the trainer looks sharper than
it is), and is the single cheapest thing to get wrong.

## 2. vLLM's batch-invariant mode breaks a transformers forward, and why

Applying `enable_batch_invariant_mode()` to the trainer process gave rms error
0.35 with a max of 44 nats. Bisecting the mode one component at a time
(`bench/probe_bi.py`): the CUBLAS workspace variables, cuBLASLt preference,
reduced-precision flags, and the softmax / log-softmax / mean overrides are all
harmless (the mean override alone even helps slightly). The `aten::bmm`
override alone reproduces the whole failure.

`bench/probe_bmm.py` shows the mechanism: the batch-invariant bmm kernel runs
fp32 inputs at roughly TF32 precision (relative error 1e-3). transformers'
Qwen3 rotary embedding computes its angles as an fp32 bmm with inner dimension
1 whose values reach 7e3 radians, so angles come out wrong by up to 4 radians
and 99.9% of positions get the wrong cos/sin. vLLM never hits this because its
RoPE uses a precomputed cache. Computing the angles as an elementwise outer
product (bit-identical in plain torch) restores the trainer to the ordinary
bf16 error of 0.0346. This is the fix `rl/grpo.py --batch-invariant` applies.

## 3. What happens in RL (non-thinking regime, 768-token cap)

Nine arms, same seed, same prompt order, 200 steps each. No arm collapsed.
Final greedy accuracy on 200 gsm8k-test problems:

| arm | what it changes | eval acc 0 -> 200 | KL(k3) first10 -> last10 | tokens/s |
|---|---|---|---|---|
| none | nothing | 0.840 -> 0.875 | 6.1e-4 -> 6.1e-4 | 7171 |
| tis | truncated IS, C=2 | 0.840 -> 0.895 | 6.3e-4 -> 7.2e-4 | 6268 |
| mis | masked IS | 0.840 -> 0.890 | 6.3e-4 -> 6.4e-4 | 7070 |
| icepop | mask outside band | 0.840 -> 0.880 | 6.2e-4 -> 6.8e-4 | 6842 |
| vllm_old | vLLM logprobs as PPO old-logprobs (framework bug) | 0.840 -> 0.850 | 6.2e-4 -> 7.9e-4 | 7051 |
| seq_tis | sequence-level IS | 0.840 -> 0.875 | 6.3e-4 -> 1.4e-3 | 6605 |
| fp16 | fp16 both sides | 0.850 -> 0.860 | 1.0e-5 -> 4.0e-5 | 6971 |
| fp32head | fp32 norm+head both sides | 0.835 -> 0.895 | 3.4e-4 -> 4.1e-4 | 5004 |
| bi | batch-invariant both sides (+RoPE fix) | 0.840 -> 0.890 | 6.3e-4 -> 6.5e-4 | 5652 |

Reading it:

- The token-level corrections (tis, mis, icepop) are indistinguishable from no
  correction at this scale; 0.875 to 0.895 is one seed's noise. They touch
  under 1% of tokens because that is how many are outside the band.
- The framework bug is the one arm that visibly misbehaved: between steps 80
  and 130 its train accuracy fell to 0.77 (others 0.88), entropy collapsed to
  0.05, completions lengthened to 560 tokens with 15% truncation, and the
  mismatch KL doubled. It recovered, but finished lowest. PPO's clip zeroes the
  gradient on exactly the tokens the sampler disagrees about, which are the
  low-probability ones the policy most needs signal on.
- Sequence-level IS is the wrong tool: with ~300-token completions its weights
  had ESS 0.15 to 0.55 and mean 0.89, and it is the only arm whose mismatch
  drifted by more than 2x.
- fp16 keeps the mismatch 40 to 60x lower throughout training for free (same
  tokens/s), reproducing the Sea AI Lab result inside a live RL loop. Its lower
  final accuracy is within noise, but note its entropy fell furthest (0.044).
- The fp32 head halves the mismatch at a 29% throughput cost (the head is 17%
  of the 1.7B model's FLOPs and runs in fp32 on both sides).
- Batch-invariant mode costs 21% throughput and changes nothing about the
  trainer-vs-sampler gap, as section 1 predicted; it solves a different
  problem (prefill vs decode determinism inside the engine).
- In every arm the mismatch tracks policy entropy: as the policy sharpens the
  KL rises, then falls back when entropy recovers. The drift over 200 steps is
  at most 2x for the token-level arms.

Monitor overhead (gate M5): 3.4 s over a 40-minute run, 0.13%.

## 4. Long-horizon (thinking mode, 2048-token cap)

_Pending: th_none, th_tis, th_fp16 running._ Starting point: 71% greedy accuracy,
27% of completions truncated at 2048, mean 1474 tokens, KL(k3) 1.0e-3 with
sequence ESS 0.22. By step 50 the baseline had learned to cut mean length to
555 tokens with 3% truncation.

## Gates

| gate | result |
|---|---|
| M1 reference noise floor < 1e-3 | PASS (3.3e-6) |
| M2 gap >= 10x floor | PASS (4190x) |
| M3 independent-error prediction within 25% | PASS (1.7%, corr 0.035) |
| M4 RL informative | PASS as a measured negative: no collapse at 1.7B / 200 steps; framework bug and sequence IS hurt; fp16 removes the gap for free; drift tracks entropy |
| M5 monitor overhead < 2% | PASS (0.13%) |

## What I would claim, and what I would not

Claim: on this model and hardware the train/inference mismatch is bf16
rounding on two independent paths, concentrated on low-probability tokens,
unaffected by attention backend or batching, removed 8x by fp16 on both sides,
and not by itself destabilising for 200 GRPO steps at 1.7B. The realistic way
to get hurt is to feed the sampler's logprobs into PPO's ratio.

Not claimed: that the corrections never matter (larger models, MoE routing,
longer horizons and more steps are exactly the settings the literature reports
collapse in, and none of them were run here); that fp16 is safe for training
in general (it needed dynamic loss scaling and its entropy fell fastest).
