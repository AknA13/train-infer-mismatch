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


### Does it change with model size? (Qwen3-8B, same protocol, 837k tokens)

| quantity | 1.7B | 8B |
|---|---|---|
| trainer bf16 error vs fp32 (rms) | 0.0346 | 0.0301 |
| vLLM bf16 error vs fp32 (rms) | 0.0298 | 0.0267 |
| trainer-vs-sampler KL(k3) | 1.0e-3 | 7.8e-4 |
| frac tokens outside [0.8,1.25] | 0.82% | 0.60% |
| sequence IS effective sample size | 5% | 12% |
| corr(err_trainer, err_vllm) | 0.035 | 0.023 |
| fp32-both-sides residual (rms) | 0.0024 | 0.0019 |
| fp16 both sides, gap rms | 0.0057 | 0.0050 |
| fp32 head on trainer, trainer error | 0.0252 (-27%) | 0.0197 (-35%) |
| fp8 KV cache, vLLM error | 0.178 | 0.131 |
| batch-invariant on trainer, broken / RoPE-fixed | 0.346 / 0.0346 | 0.218 / 0.0300 |
| vLLM prefill vs decode (same engine) | 0.0102 | 0.0148 |

Same picture, slightly smaller gap at 8B; every ranking of toggles is
preserved, including the RoPE failure (8B has an untied lm_head, so the fp32-head
result is not a tied-embedding artefact). The one quantity that grows with size
is vLLM's own prefill-vs-decode disagreement (0.010 to 0.015), which the
batch-invariant mode zeroes on both models.

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

Second seed (different prompt order, different 200-problem eval subset, so
compare the change from step 0, not the level):

| arm | seed 0: eval 0 -> 200 | seed 1: eval 0 -> 200 | seed 1 outcome |
|---|---|---|---|
| none | 0.840 -> 0.875 | 0.785 -> 0.855 | stable |
| tis | 0.840 -> 0.895 | 0.785 -> 0.860 | stable |
| fp16 | 0.850 -> 0.860 | 0.790 -> 0.875 | stable |
| vllm_old | 0.840 -> 0.850 | 0.785 -> 0.780 | **collapsed at step 95** |

Reading it:

- **The framework bug collapses.** On seed 1 the "sampler logprobs as PPO
  old-logprobs" arm went from 0.74 to 0.03 train accuracy between steps 90 and
  100 with 96% of completions running to the length cap, then partially
  recovered to ~0.75. On seed 0 the same arm only dipped to 0.77 around step
  100 and recovered. No other arm on either seed did anything like this. The
  mechanism is visible in the logs: entropy fell from 0.11 to 0.05 by step 90,
  and over steps 60 to 90 the mismatch KL rose from 6e-4 to 1.5e-3 and the
  band-violation fraction doubled, twenty to thirty steps *before* the reward
  moved. PPO's clip then zeroes the gradient on precisely the tokens the two
  engines disagree about, which are the low-probability ones, and the policy
  drifts with no signal to pull it back. This is the unstable/stable pair gate
  M4 asked for, and the monitor's KL and band signals are a usable early warning.
- The token-level corrections (tis, mis, icepop) are indistinguishable from no
  correction at this scale. Seed-to-seed spread of the final eval is about
  +-0.02, and they sit inside it. They touch under 1% of tokens because that is
  how many are outside the band.
- Sequence-level IS is the wrong tool: with ~300-token completions its weights
  had ESS 0.15 to 0.55 and mean 0.89, it is the only arm whose mismatch drifted
  by more than 2x, and it finished lowest of the stable arms (0.805).
- fp16 keeps the mismatch 40 to 60x lower throughout training for free (same
  tokens/s), reproducing the Sea AI Lab result inside a live RL loop. It needed
  dynamic loss scaling (no overflow steps occurred). One caution: seed 0's
  entropy fell to 0.044, the lowest of any stable arm, and its KL rose 5x in the
  last 25 steps with grad-norm spikes; seed 1 did not show this.
- The fp32 head halves the mismatch at a 29% throughput cost (the head is about
  18% of the 1.7B model's FLOPs and runs in fp32 on both sides).
- Batch-invariant mode costs 21% throughput and changes nothing about the
  trainer-vs-sampler gap, as section 1 predicted; it solves a different
  problem (prefill vs decode determinism inside the engine).
- In every arm the mismatch tracks policy entropy: as the policy sharpens the
  KL rises, then falls back when entropy recovers. The drift over 200 steps is
  at most 2x for the stable arms.

Monitor overhead (gate M5): 3.4 s over a 40-minute run, 0.13%.

## 4. Long-horizon (thinking mode, 2048-token cap)

Same recipe with Qwen3's thinking mode on and a 2048-token cap. Start: 71%
greedy accuracy, 27% of completions truncated, mean 1474 tokens, mismatch KL
1.0e-3 with sequence ESS 0.15 (the sequence-level ratio is hopeless at this
length). The reward is correctness with truncation counted as wrong, so the
policy learns to be shorter: mean length 1474 -> 555 by step 50 -> ~400 by
step 200 for the bf16 arms, ~650 to 840 for fp16.

| arm | eval 0 -> 200 | KL(k3) first10 -> last10 | max KL | entropy first -> last | s/step |
|---|---|---|---|---|---|
| th_none | 0.710 -> 0.890 | 1.0e-3 -> 2.4e-3 | 3.2e-3 | 0.215 -> 0.047 | 21 |
| th_tis | 0.710 -> 0.890 | 1.0e-3 -> 1.0e-3 | 1.2e-3 | 0.218 -> 0.075 | 28 |
| th_fp16 (189/200 steps, preempted 3x, resumed) | 0.670 -> 0.895 | 1.5e-5 -> 1.4e-5 | 2.2e-5 | 0.192 -> 0.113 | 29 |

No collapse. The bf16 baseline's mismatch grew 2.4x as its entropy fell to
0.047 (the largest drift of any arm), TIS held it flat, and fp16 held it 100x
lower. Same final accuracy for all three. The long-horizon stress mostly
disappears because RL shortens the outputs; a run that rewarded length would
be the real test and was not done.

Preemption note: th_fp16 was preempted three times; every resume reproduced
the pre-preemption step exactly (same accuracy, KL and lengths at step 150),
so the checkpoint/resume path is deterministic.

## Gates

| gate | result |
|---|---|
| M1 reference noise floor < 1e-3 | PASS (3.3e-6) |
| M2 gap >= 10x floor | PASS (4190x) |
| M3 independent-error prediction within 25% | PASS (1.7%, corr 0.035) |
| M4 RL informative | PASS: unstable/stable pair found (framework bug collapses on seed 1, every correction and fix stays stable); the monitor's KL and band signals lead the collapse by ~25 steps; fp16 removes the gap for free; drift tracks entropy |
| M5 monitor overhead < 2% | PASS (0.13%) |

## What I would claim, and what I would not

Claim: on this model and hardware the train/inference mismatch is bf16
rounding on two independent paths, concentrated on low-probability tokens,
unaffected by attention backend or batching, removed 8x by fp16 on both sides,
and not by itself destabilising for 200 GRPO steps at 1.7B when the loss is a
plain policy gradient. The realistic way to get hurt is to feed the sampler's
logprobs into PPO's ratio: that arm collapsed on one of two seeds and dipped on
the other, and the mismatch monitor saw it coming 25 steps early.

Not claimed: that the corrections never matter (larger models, MoE routing,
longer horizons and more steps are exactly the settings the literature reports
collapse in, and none of them were run here); that fp16 is safe for training
in general (it needed dynamic loss scaling and its entropy fell fastest).
