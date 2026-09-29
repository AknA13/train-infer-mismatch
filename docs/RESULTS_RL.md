# GRPO arms: stability, mismatch drift, cost

| arm | mode | dtype | head | BI | steps | acc first10 -> last10 | eval acc first -> last | collapsed | entropy first -> last | gn median / max | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nt_none | none | bf16 | none | False | 200/200 done | 0.835 -> 0.920 | 0.840 -> 0.875 | no | 0.133 -> 0.096 | 0.405 / 0.703 | 0 |
| nt_tis | tis | bf16 | none | False | 200/200 done | 0.834 -> 0.899 | 0.840 -> 0.895 | no | 0.131 -> 0.085 | 0.546 / 0.847 | 0 |
| nt_vllmold | vllm_old | bf16 | none | False | 177/200 | 0.843 -> 0.842 | 0.840 -> 0.870 | no | 0.126 -> 0.076 | 0.531 / 1.828 | 0 |

## Mismatch during training (first 10 steps -> last 10)

| arm | KL k3 | mean abs gap | frac outside band | seq ESS | max KL k3 | w_mean last10 |
|---|---|---|---|---|---|---|
| nt_none | 0.000614 -> 0.000607 | 0.00858 -> 0.00742 | 0.00515 -> 0.00552 | 0.679 -> 0.682 | 0.000813 | 1.000 |
| nt_tis | 0.000629 -> 0.000716 | 0.00863 -> 0.00775 | 0.00526 -> 0.00661 | 0.702 -> 0.591 | 0.00143 | 1.000 |
| nt_vllmold | 0.000624 -> 0.00104 | 0.00839 -> 0.0089 | 0.0053 -> 0.00911 | 0.692 -> 0.502 | 0.0015 | 0.998 |

## Cost

| arm | s/step | gen | train | sync | tokens/s | peak GiB | monitor overhead |
|---|---|---|---|---|---|---|---|
| nt_none | 12.850 | 6.151 | 6.589 | 0.012 | 7171.1143 | 89.755 | 0.00133 |
| nt_tis | 13.792 | 7.390 | 6.299 | 0.011 | 6267.6031 | 89.755 | 0.00123 |
| nt_vllmold | 12.728 | 6.135 | 6.469 | 0.011 | 7021.6016 | 89.755 | 0.00133 |
