# Train/inference mismatch: plan and definition of done

## Question

RL fine-tuning of LLMs samples from one implementation of the policy (an
inference engine: vLLM, CUDA graphs, fused attention, bf16 everywhere) and
takes gradients through another (the trainer: transformers/FSDP, SDPA, bf16
params with fp32 master weights). They are the same weights, but not the same
function. The per-token gap between log pi_infer(x) and log pi_train(x) is the
train/inference mismatch: it makes nominally on-policy RL off-policy, and it
has been blamed for training collapse.

Prior work documents the phenomenon and proposes fixes one at a time (TIS/MIS
from Yao et al., fp32 LM head from MiniMax-M1, fp16 from Sea AI Lab, IcePop
masking from Ring-1T, batch-invariant kernels from Thinking Machines / vLLM).
What is missing is a controlled decomposition: on one model and one corpus,
which component of each stack contributes how much, and what each fix costs
in throughput against what it buys in stability.

## Stages

1. **Corpus** (`mismatch/sample.py`). Sample gsm8k-train completions from
   Qwen3-1.7B (and 8B) through vLLM at T=1 with no truncation, recording the
   sampler's per-token logprob. This is pi_infer, decode path.
2. **Views** (`mismatch/score_vllm.py`, `mismatch/score_hf.py`,
   `mismatch/views.py`). Re-score the same tokens through 23 configurations:
   vLLM prefill under four attention backends, batch-invariant mode, eager vs
   CUDA graphs, batch width, chunk size, prefix caching, fp8 KV, fp16, fp32;
   transformers under bf16/fp16/fp32/fp64, SDPA vs eager, padded batches,
   fp32 head, bf16 log-softmax, batch-invariant aten ops.
3. **Report** (`mismatch/report.py`). Reference = fp32 eager, TF32 off,
   batch 1, checked against fp64. Every view's error vs reference; the
   headline trainer-vs-sampler gap with KL(k3), band violations, sequence-level
   IS effective sample size, bucketed by token probability and position; one
   toggle at a time attribution; dtype-vs-kernel split; independence check.
4. **RL** (`rl/grpo.py`). Single-GPU GRPO with an in-process vLLM engine and
   weight sync every step. Arms: no correction, TIS, MIS, IcePop, sequence TIS,
   "vLLM logprobs as PPO old-logprobs" (the framework bug), fp16 both sides,
   fp32 head (trainer / both), batch-invariant both sides. The
   `MismatchMonitor` logs the gap every step so drift over training is
   measured, not assumed.
5. **Monitor** (`mismatch/monitor.py`). Drop-in for any trainer: takes the
   trainer's and sampler's logprobs on a step's tokens, returns KL, band
   violations, what each correction would clip/drop, sequence ESS, and a
   probability-bucketed gap histogram.

## Gates

| gate | statement | threshold |
|---|---|---|
| M1 | reference is a reference | fp32 eager vs fp32 sdpa and vs fp64 mean abs gap < 1e-3 |
| M2 | phenomenon reproduces | trainer-bf16 vs sampler mean abs gap >= 10x the M1 floor |
| M3 | attribution is usable | independent-error prediction of the gap within 25% of measured |
| M4 | RL: informative either way | an unstable/stable pair, or measured drift + cost table |
| M5 | monitor is free | overhead < 2% of step time |

No stage is done until its gate passes or its failure is reported with the
metric that shows it.

## Compute

Node-local `/data` on horton for everything large. One H200 per job,
`qos=preemptive`, at most 4 GPUs in flight, every stage resumable per unit of
work (per view file, per GRPO checkpoint).
