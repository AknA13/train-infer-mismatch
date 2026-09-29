# Measurement report: Qwen/Qwen3-1.7B

Corpus: 512 gsm8k-train prompts, 771300 sampled tokens (mean 1506/seq, 141 length-capped), T=1.0 top_p=1.0 top_k=-1, vLLM 0.12.0 default(FLASH_ATTN). Accuracy 0.861.

## M1 reference noise floor

| pair vs hf_fp32_eager | mean abs gap | rms | max | tokens |
|---|---|---|---|---|
| hf_fp32_sdpa | 2.314e-06 | 6.795e-06 | 0.0003929 | 771300 |
| hf_fp64_eager | 3.309e-06 | 9.261e-06 | 0.0003567 | 97227 |
| hf_fp32_tf32 | 0.0005075 | 0.00169 | 0.2440 | 771300 |

Noise floor (mean abs) = 3.309e-06; gate M1 (< 0.001): **PASS**

## Every view's error against the fp32 reference

| view | family | toggle | rms err | mean abs | bias | p99 abs | max abs | frac outside band |
|---|---|---|---|---|---|---|---|---|
| hf_fp32_sdpa | hf | sdpa kernel @fp32 | 6.795e-06 | 2.314e-06 | -2.18e-09 | 3.052e-05 | 0.0003929 | 0 |
| hf_fp64_eager | hf | fp64 (subset) | 9.261e-06 | 3.309e-06 | -3.355e-08 | 4.196e-05 | 0.0003567 | 0 |
| hf_fp32_tf32 | hf | TF32 matmul @fp32 | 0.00169 | 0.0005075 | -8.784e-07 | 0.00729 | 0.2440 | 1.297e-06 |
| vllm_fp32_triton | vllm | fp32 weights+activations (Triton backend) | 0.002358 | 0.0007256 | -7.543e-06 | 0.0105 | 0.1891 | 0 |
| vllm_fp16_fa | vllm | fp16 instead of bf16 | 0.003807 | 0.001341 | -2.315e-07 | 0.0174 | 0.2815 | 2.593e-06 |
| hf_fp16_sdpa_bs1 | hf | fp16 instead of bf16 | 0.004316 | 0.001504 | -1.925e-05 | 0.0198 | 0.2474 | 1.297e-06 |
| hf_bf16_sdpa_bs1_fp32head | hf | fp32 norm+lm_head | 0.0252 | 0.007864 | -0.0002985 | 0.1130 | 1.3731 | 0.001555 |
| vllm_bf16_fa_chunk512 | vllm | chunked prefill 512 | 0.0297 | 0.0106 | -0.000113 | 0.1382 | 1.3845 | 0.00146 |
| vllm_sample | vllm | sampler (decode path) | 0.0298 | 0.0106 | 0.0004696 | 0.1381 | 1.5359 | 0.001474 |
| vllm_bf16_fa_bs1 | vllm | max_num_seqs=1 | 0.0298 | 0.0107 | 0.0004179 | 0.1383 | 1.5359 | 0.001455 |
| vllm_bf16_flashinfer | vllm | FlashInfer backend | 0.0298 | 0.0107 | -0.0001997 | 0.1385 | 1.1555 | 0.001521 |
| vllm_bf16_fa | vllm | inference baseline | 0.0298 | 0.0107 | 0.0004239 | 0.1384 | 1.5359 | 0.00147 |
| vllm_bf16_fa_noprefix | vllm | prefix caching off | 0.0298 | 0.0107 | 0.0004239 | 0.1384 | 1.5359 | 0.00147 |
| vllm_bf16_fa_bi | vllm | batch-invariant mode | 0.0298 | 0.0107 | 0.0004239 | 0.1384 | 1.5359 | 0.00147 |
| vllm_bf16_flex | vllm | FlexAttention backend | 0.0299 | 0.0107 | -0.0003078 | 0.1387 | 1.5373 | 0.001555 |
| vllm_bf16_triton | vllm | Triton attention backend | 0.0300 | 0.0107 | -0.0003332 | 0.1391 | 1.2924 | 0.001648 |
| hf_bf16_sdpa_bs8 | hf | padded batch of 8 | 0.0345 | 0.0120 | -0.0005568 | 0.1578 | 1.5010 | 0.003064 |
| hf_bf16_sdpa_bs1 | hf | trainer baseline | 0.0346 | 0.0121 | -0.0005879 | 0.1587 | 1.4206 | 0.003173 |
| hf_bf16_sdpa_bs1_bi_ropefix | hf | batch-invariant ops + RoPE without bmm | 0.0346 | 0.0121 | -0.0006528 | 0.1579 | 1.9834 | 0.003153 |
| vllm_bf16_fa_eager | vllm | enforce_eager (no cudagraph/compile) | 0.0348 | 0.0120 | -0.0006012 | 0.1579 | 2.4626 | 0.003187 |
| hf_bf16_eager_bs1 | hf | eager attention | 0.0420 | 0.0141 | -0.000795 | 0.1880 | 2.7227 | 0.006239 |
| hf_bf16_sdpa_bs1_nativelogits | hf | log_softmax in bf16 | 0.0542 | 0.0266 | 0.004794 | 0.2003 | 1.4306 | 0.006623 |
| vllm_bf16_fa_kvfp8 | vllm | fp8 KV cache | 0.1780 | 0.0529 | -0.0130 | 0.7582 | 15.2498 | 0.0686 |
| hf_bf16_sdpa_bs1_bi | hf | batch-invariant aten ops | 0.3464 | 0.0230 | -0.0102 | 0.1929 | 43.9108 | 0.007582 |

## Headline pairs

| pair | rms gap | mean abs | KL k3 (if x~b) | frac outside band | frac r>2 | seq ESS frac | seq |S| p99 |
|---|---|---|---|---|---|---|---|
| trainer bf16 vs sampler (the RL mismatch) (hf_bf16_sdpa_bs1 vs vllm_sample) | 0.0449 | 0.0139 | 0.001002 | 0.008233 | 1.167e-05 | 0.0496 | 7.0280 |
| trainer bf16 vs vLLM prefill re-score (hf_bf16_sdpa_bs1 vs vllm_bf16_fa) | 0.0449 | 0.0139 | - | 0.008212 | 1.167e-05 | 0.0707 | 6.9348 |
| vLLM prefill vs vLLM decode (same engine) (vllm_bf16_fa vs vllm_sample) | 0.0102 | 0.0008769 | 5.157e-05 | 0.000328 | 0 | 0.8733 | 1.6420 |
| batch-invariant: prefill vs decode (vllm_bf16_fa_bi vs vllm_sample @bi corpus) | 0 | 0 | 0 | 0 | 0 | 1.0000 | 0 |
| fp16 both sides (hf_fp16_sdpa_bs1 vs vllm_fp16_fa) | 0.005698 | 0.001754 | - | 5.186e-06 | 0 | 0.9525 | 0.6338 |
| fp32 both sides (kernel-only residual) (hf_fp32_eager vs vllm_fp32_triton) | 0.002358 | 0.0007256 | - | 0 | 0 | 0.9912 | 0.2733 |
| batch-invariant both sides (broken RoPE) (hf_bf16_sdpa_bs1_bi vs vllm_bf16_fa_bi) | 0.3476 | 0.0248 | - | 0.0127 | 0.0003293 | 0.1670 | 158.0875 |
| batch-invariant both sides, RoPE fixed (hf_bf16_sdpa_bs1_bi_ropefix vs vllm_bf16_fa_bi) | 0.0450 | 0.0139 | - | 0.008243 | 2.204e-05 | 0.0146 | 6.6419 |
| fp32 head on trainer only (hf_bf16_sdpa_bs1_fp32head vs vllm_bf16_fa) | 0.0384 | 0.0132 | - | 0.00468 | 7.779e-06 | 0.1989 | 5.5493 |

### Trainer-vs-sampler gap by reference token probability

| p_ref bucket | tokens | mean abs gap | mean gap | frac outside band |
|---|---|---|---|---|
| [0,0.01) | 2099 | 0.1236 | -0.004209 | 0.2530 |
| [0.01,0.1) | 13645 | 0.1082 | -0.005459 | 0.2394 |
| [0.1,0.5) | 61666 | 0.0780 | -0.006394 | 0.0371 |
| [0.5,0.9) | 116287 | 0.0306 | -0.002395 | 0.002262 |
| [0.9,1.0001) | 577603 | 0.001022 | -0.000103 | 3.463e-06 |

### ... by position

| position | tokens | mean abs gap | mean gap |
|---|---|---|---|
| [0,256) | 131072 | 0.0122 | -0.001275 |
| [256,512) | 130958 | 0.0133 | -0.001041 |
| [512,768) | 127563 | 0.0146 | -0.00107 |
| [768,1024) | 114247 | 0.0144 | -0.0009834 |
| [1024,1280) | 94051 | 0.0142 | -0.001076 |
| [1280,1536) | 74368 | 0.0142 | -0.0009316 |
| [1536,1792) | 57284 | 0.0140 | -0.0008825 |
| [1792,2048) | 41757 | 0.0159 | -0.001012 |

## Attribution (one toggle at a time)

Baseline gap hf_bf16_sdpa_bs1 vs vllm_bf16_fa: rms 0.0449, mean abs 0.0139, outside band 0.008212.

| toggle | side | gap rms vs other baseline | delta gap | err rms vs ref | delta err |
|---|---|---|---|---|---|
| batch-invariant aten ops (hf_bf16_sdpa_bs1_bi) | hf | 0.3476 | 0.3027 | 0.3464 | 0.3118 |
| fp8 KV cache (vllm_bf16_fa_kvfp8) | vllm | 0.1801 | 0.1352 | 0.1780 | 0.1482 |
| log_softmax in bf16 (hf_bf16_sdpa_bs1_nativelogits) | hf | 0.0613 | 0.0164 | 0.0542 | 0.0196 |
| eager attention (hf_bf16_eager_bs1) | hf | 0.0511 | 0.006186 | 0.0420 | 0.007352 |
| enforce_eager (no cudagraph/compile) (vllm_bf16_fa_eager) | vllm | 0.0420 | -0.00294 | 0.0348 | 0.004936 |
| Triton attention backend (vllm_bf16_triton) | vllm | 0.0452 | 0.0002553 | 0.0300 | 0.0001796 |
| FlexAttention backend (vllm_bf16_flex) | vllm | 0.0452 | 0.0003408 | 0.0299 | 9.979e-05 |
| batch-invariant ops + RoPE without bmm (hf_bf16_sdpa_bs1_bi_ropefix) | hf | 0.0450 | 4.678e-05 | 0.0346 | 3.358e-05 |
| batch-invariant mode (vllm_bf16_fa_bi) | vllm | 0.0449 | 2.711e-09 | 0.0298 | 1.878e-09 |
| prefix caching off (vllm_bf16_fa_noprefix) | vllm | 0.0449 | 0 | 0.0298 | 0 |
| FlashInfer backend (vllm_bf16_flashinfer) | vllm | 0.0450 | 6.691e-05 | 0.0298 | -7.888e-06 |
| max_num_seqs=1 (vllm_bf16_fa_bs1) | vllm | 0.0449 | -2.023e-05 | 0.0298 | -1.538e-05 |
| chunked prefill 512 (vllm_bf16_fa_chunk512) | vllm | 0.0450 | 6.27e-05 | 0.0297 | -0.0001005 |
| padded batch of 8 (hf_bf16_sdpa_bs8) | hf | 0.0449 | 1.707e-05 | 0.0345 | -0.00012 |
| fp32 norm+lm_head (hf_bf16_sdpa_bs1_fp32head) | hf | 0.0384 | -0.006521 | 0.0252 | -0.009432 |
| fp16 instead of bf16 (vllm_fp16_fa) | vllm | 0.0348 | -0.0101 | 0.003807 | -0.0260 |
| fp32 weights+activations (Triton backend) (vllm_fp32_triton) | vllm | 0.0347 | -0.0102 | 0.002358 | -0.0275 |
| fp16 instead of bf16 (hf_fp16_sdpa_bs1) | hf | 0.0301 | -0.0148 | 0.004316 | -0.0303 |
| TF32 matmul @fp32 (hf_fp32_tf32) | hf | 0.0299 | -0.0150 | 0.00169 | -0.0329 |
| sdpa kernel @fp32 (hf_fp32_sdpa) | hf | 0.0298 | -0.0151 | 6.795e-06 | -0.0346 |

### dtype vs kernels

- fp32 both sides (kernel-only residual): rms 0.002358
- trainer side: bf16 error 0.0346 vs sdpa@fp32 error 6.795e-06
- inference side: bf16 error 0.0298 vs fp32-vLLM error 0.002358
- independence check: predicted gap sqrt(ea^2+eb^2) = 0.0457, measured 0.0449, corr(err_a, err_b) = 0.0351 -> gate M3 **PASS**

## M2 phenomenon reproduces

trainer-vs-sampler mean abs gap 0.0139 = 4190x the noise floor; KL k3 = 0.001002 -> **PASS**
