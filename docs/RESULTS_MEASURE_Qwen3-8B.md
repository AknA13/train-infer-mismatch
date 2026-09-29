# Measurement report: Qwen/Qwen3-8B

Corpus: 512 gsm8k-train prompts, 837371 sampled tokens (mean 1635/seq, 180 length-capped), T=1.0 top_p=1.0 top_k=-1, vLLM 0.12.0 default(FLASH_ATTN). Accuracy 0.871.

## M1 reference noise floor

| pair vs hf_fp32_eager | mean abs gap | rms | max | tokens |
|---|---|---|---|---|
| hf_fp32_sdpa | 2.597e-06 | 7.803e-06 | 0.0002365 | 837371 |
| hf_fp64_eager | 4.461e-06 | 1.272e-05 | 0.000206 | 103589 |
| hf_fp32_tf32 | 0.0003522 | 0.001195 | 0.0507 | 837371 |

Noise floor (mean abs) = 4.461e-06; gate M1 (< 0.001): **PASS**

## Every view's error against the fp32 reference

| view | family | toggle | rms err | mean abs | bias | p99 abs | max abs | frac outside band |
|---|---|---|---|---|---|---|---|---|
| hf_fp32_sdpa | hf | sdpa kernel @fp32 | 7.803e-06 | 2.597e-06 | 9.134e-10 | 3.815e-05 | 0.0002365 | 0 |
| hf_fp64_eager | hf | fp64 (subset) | 1.272e-05 | 4.461e-06 | -4.918e-08 | 6.104e-05 | 0.000206 | 0 |
| hf_fp32_tf32 | hf | TF32 matmul @fp32 | 0.001195 | 0.0003522 | -2.64e-06 | 0.005348 | 0.0507 | 0 |
| vllm_fp32_triton | vllm | fp32 weights+activations (Triton backend) | 0.001917 | 0.000562 | -3.066e-06 | 0.008624 | 0.1660 | 0 |
| vllm_fp16_fa | vllm | fp16 instead of bf16 | 0.003374 | 0.001175 | 3.745e-06 | 0.0159 | 0.0919 | 0 |
| hf_fp16_sdpa_bs1 | hf | fp16 instead of bf16 | 0.003852 | 0.00131 | -6.384e-06 | 0.0179 | 0.1122 | 0 |
| hf_bf16_sdpa_bs1_fp32head | hf | fp32 norm+lm_head | 0.0197 | 0.005856 | -0.0002139 | 0.0892 | 0.7944 | 0.0007547 |
| vllm_bf16_flashinfer | vllm | FlashInfer backend | 0.0267 | 0.009312 | -0.0002634 | 0.1259 | 1.0269 | 0.0009506 |
| vllm_bf16_fa | vllm | inference baseline | 0.0267 | 0.009315 | 0.0002504 | 0.1259 | 0.7063 | 0.0009327 |
| vllm_bf16_fa_bs1 | vllm | max_num_seqs=1 | 0.0267 | 0.009315 | 0.0002504 | 0.1259 | 0.7063 | 0.0009327 |
| vllm_bf16_fa_noprefix | vllm | prefix caching off | 0.0267 | 0.009315 | 0.0002504 | 0.1259 | 0.7063 | 0.0009327 |
| vllm_bf16_fa_bi | vllm | batch-invariant mode | 0.0267 | 0.009315 | 0.0002505 | 0.1259 | 0.7063 | 0.0009327 |
| vllm_bf16_flex | vllm | FlexAttention backend | 0.0267 | 0.009307 | -0.0002881 | 0.1263 | 0.8194 | 0.0009315 |
| vllm_sample | vllm | sampler (decode path) | 0.0267 | 0.009308 | 0.0003402 | 0.1257 | 0.7063 | 0.0009506 |
| vllm_bf16_fa_chunk512 | vllm | chunked prefill 512 | 0.0267 | 0.00932 | 0.0002429 | 0.1259 | 0.7063 | 0.0009387 |
| vllm_bf16_triton | vllm | Triton attention backend | 0.0268 | 0.009339 | -0.0002267 | 0.1263 | 1.0296 | 0.001021 |
| vllm_bf16_fa_eager | vllm | enforce_eager (no cudagraph/compile) | 0.0299 | 0.0102 | -0.0004353 | 0.1393 | 1.0989 | 0.001955 |
| hf_bf16_sdpa_bs1_bi_ropefix | hf | batch-invariant ops + RoPE without bmm | 0.0300 | 0.0102 | -0.0003935 | 0.1397 | 0.9825 | 0.001962 |
| hf_bf16_sdpa_bs8 | hf | padded batch of 8 | 0.0300 | 0.0102 | -0.0004698 | 0.1392 | 1.2058 | 0.001962 |
| hf_bf16_sdpa_bs1 | hf | trainer baseline | 0.0301 | 0.0103 | -0.0004767 | 0.1398 | 0.8723 | 0.002013 |
| hf_bf16_eager_bs1 | hf | eager attention | 0.0348 | 0.0115 | -0.0005732 | 0.1595 | 1.5195 | 0.003683 |
| hf_bf16_sdpa_bs1_nativelogits | hf | log_softmax in bf16 | 0.0497 | 0.0239 | 0.004426 | 0.1839 | 0.8569 | 0.0045 |
| vllm_bf16_fa_kvfp8 | vllm | fp8 KV cache | 0.1311 | 0.0386 | -0.007937 | 0.5793 | 8.2735 | 0.0474 |
| hf_bf16_sdpa_bs1_bi | hf | batch-invariant aten ops | 0.2184 | 0.0156 | -0.004709 | 0.1648 | 46.4062 | 0.005341 |

## Headline pairs

| pair | rms gap | mean abs | KL k3 (if x~b) | frac outside band | frac r>2 | seq ESS frac | seq |S| p99 |
|---|---|---|---|---|---|---|---|
| trainer bf16 vs sampler (the RL mismatch) (hf_bf16_sdpa_bs1 vs vllm_sample) | 0.0398 | 0.0117 | 0.0007837 | 0.006032 | 3.583e-06 | 0.1216 | 5.8188 |
| trainer bf16 vs vLLM prefill re-score (hf_bf16_sdpa_bs1 vs vllm_bf16_fa) | 0.0398 | 0.0117 | - | 0.005989 | 3.583e-06 | 0.1242 | 5.3553 |
| vLLM prefill vs vLLM decode (same engine) (vllm_bf16_fa vs vllm_sample) | 0.0148 | 0.001919 | 0.0001093 | 0.0006294 | 1.194e-06 | 0.6465 | 2.2006 |
| batch-invariant: prefill vs decode (vllm_bf16_fa_bi vs vllm_sample @bi corpus) | 0 | 0 | 0 | 0 | 0 | 1.0000 | 0 |
| fp16 both sides (hf_fp16_sdpa_bs1 vs vllm_fp16_fa) | 0.00501 | 0.00148 | - | 0 | 0 | 0.9651 | 0.5117 |
| fp32 both sides (kernel-only residual) (hf_fp32_eager vs vllm_fp32_triton) | 0.001917 | 0.000562 | - | 0 | 0 | 0.9936 | 0.2405 |
| batch-invariant both sides (broken RoPE) (hf_bf16_sdpa_bs1_bi vs vllm_bf16_fa_bi) | 0.2200 | 0.0170 | - | 0.00923 | 0.0001947 | 0.0186 | 82.1036 |
| batch-invariant both sides, RoPE fixed (hf_bf16_sdpa_bs1_bi_ropefix vs vllm_bf16_fa_bi) | 0.0397 | 0.0117 | - | 0.00603 | 5.971e-06 | 0.0793 | 5.1868 |
| fp32 head on trainer only (hf_bf16_sdpa_bs1_fp32head vs vllm_bf16_fa) | 0.0327 | 0.0110 | - | 0.002892 | 4.777e-06 | 0.2895 | 3.9487 |

### Trainer-vs-sampler gap by reference token probability

| p_ref bucket | tokens | mean abs gap | mean gap | frac outside band |
|---|---|---|---|---|
| [0,0.01) | 1916 | 0.1104 | -0.0247 | 0.2375 |
| [0.01,0.1) | 13467 | 0.0972 | -0.0151 | 0.2196 |
| [0.1,0.5) | 64430 | 0.0693 | -0.006764 | 0.0233 |
| [0.5,0.9) | 118924 | 0.0275 | -0.0003135 | 0.00116 |
| [0.9,1.0001) | 638634 | 0.0008436 | 6.322e-05 | 1.566e-06 |

### ... by position

| position | tokens | mean abs gap | mean gap |
|---|---|---|---|
| [0,256) | 131072 | 0.00903 | -0.0007748 |
| [256,512) | 130918 | 0.0105 | -0.000824 |
| [512,768) | 129444 | 0.0121 | -0.000882 |
| [768,1024) | 123001 | 0.0127 | -0.000656 |
| [1024,1280) | 107794 | 0.0130 | -0.0009376 |
| [1280,1536) | 89988 | 0.0126 | -0.0007058 |
| [1536,1792) | 71498 | 0.0125 | -0.0009019 |
| [1792,2048) | 53656 | 0.0127 | -0.0009441 |

## Attribution (one toggle at a time)

Baseline gap hf_bf16_sdpa_bs1 vs vllm_bf16_fa: rms 0.0398, mean abs 0.0117, outside band 0.005989.

| toggle | side | gap rms vs other baseline | delta gap | err rms vs ref | delta err |
|---|---|---|---|---|---|
| batch-invariant aten ops (hf_bf16_sdpa_bs1_bi) | hf | 0.2200 | 0.1802 | 0.2184 | 0.1883 |
| fp8 KV cache (vllm_bf16_fa_kvfp8) | vllm | 0.1348 | 0.0950 | 0.1311 | 0.1043 |
| log_softmax in bf16 (hf_bf16_sdpa_bs1_nativelogits) | hf | 0.0561 | 0.0163 | 0.0497 | 0.0196 |
| eager attention (hf_bf16_eager_bs1) | hf | 0.0434 | 0.003609 | 0.0348 | 0.004765 |
| enforce_eager (no cudagraph/compile) (vllm_bf16_fa_eager) | vllm | 0.0379 | -0.001871 | 0.0299 | 0.003214 |
| Triton attention backend (vllm_bf16_triton) | vllm | 0.0396 | -0.0001267 | 0.0268 | 0.0001156 |
| chunked prefill 512 (vllm_bf16_fa_chunk512) | vllm | 0.0398 | 4.342e-05 | 0.0267 | 2.285e-05 |
| FlexAttention backend (vllm_bf16_flex) | vllm | 0.0396 | -0.0001175 | 0.0267 | 6.416e-06 |
| batch-invariant mode (vllm_bf16_fa_bi) | vllm | 0.0398 | 1.344e-09 | 0.0267 | 4.9e-10 |
| max_num_seqs=1 (vllm_bf16_fa_bs1) | vllm | 0.0398 | 0 | 0.0267 | 0 |
| prefix caching off (vllm_bf16_fa_noprefix) | vllm | 0.0398 | 0 | 0.0267 | 0 |
| FlashInfer backend (vllm_bf16_flashinfer) | vllm | 0.0396 | -0.0001853 | 0.0267 | -1.222e-07 |
| padded batch of 8 (hf_bf16_sdpa_bs8) | hf | 0.0397 | -6.506e-06 | 0.0300 | -5.31e-05 |
| batch-invariant ops + RoPE without bmm (hf_bf16_sdpa_bs1_bi_ropefix) | hf | 0.0397 | -6.802e-05 | 0.0300 | -7.551e-05 |
| fp32 norm+lm_head (hf_bf16_sdpa_bs1_fp32head) | hf | 0.0327 | -0.007083 | 0.0197 | -0.0104 |
| fp16 instead of bf16 (vllm_fp16_fa) | vllm | 0.0303 | -0.009495 | 0.003374 | -0.0233 |
| fp32 weights+activations (Triton backend) (vllm_fp32_triton) | vllm | 0.0301 | -0.009677 | 0.001917 | -0.0248 |
| fp16 instead of bf16 (hf_fp16_sdpa_bs1) | hf | 0.0269 | -0.0129 | 0.003852 | -0.0262 |
| TF32 matmul @fp32 (hf_fp32_tf32) | hf | 0.0267 | -0.0130 | 0.001195 | -0.0289 |
| sdpa kernel @fp32 (hf_fp32_sdpa) | hf | 0.0267 | -0.0130 | 7.803e-06 | -0.0301 |

### dtype vs kernels

- fp32 both sides (kernel-only residual): rms 0.001917
- trainer side: bf16 error 0.0301 vs sdpa@fp32 error 7.803e-06
- inference side: bf16 error 0.0267 vs fp32-vLLM error 0.001917
- independence check: predicted gap sqrt(ea^2+eb^2) = 0.0402, measured 0.0398, corr(err_a, err_b) = 0.0230 -> gate M3 **PASS**

## M2 phenomenon reproduces

trainer-vs-sampler mean abs gap 0.0117 = 2622x the noise floor; KL k3 = 0.0007837 -> **PASS**
