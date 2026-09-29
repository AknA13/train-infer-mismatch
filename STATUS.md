# Status

**2026-09-29, evening: done.** Every stage ran; gates M1-M5 pass (see docs/RESULTS.md).

- Measurement: Qwen3-1.7B and Qwen3-8B, 24 views each (23 + fixed-BI view), all scored.
- RL: 17 GRPO arms (9 non-thinking seed 0, 4 non-thinking seed 1, 3 thinking mode, plus
  smoke). All 17 arms ran their full 200 steps (two were preempted up to four times and resumed from checkpoint).
- Root cause of vLLM batch-invariant mode breaking a transformers forward isolated to
  the `aten::bmm` override x Qwen3 RoPE (bench/probe_bi.py, bench/probe_bmm.py); fix in
  mismatch/fp32head.py::patch_rope_no_bmm.

Compute used: 14 GPU-hours (sacct, 28 job records incl. requeues) on H200s, never more than 4 at once, qos=preemptive.

Not done / next: larger models and MoE (routing mismatch), a length-rewarding
long-horizon run, more seeds for the correction arms, the vLLM-side fp32 head as a
proper option rather than a monkeypatch, and an upstream note about the
batch-invariant bmm precision.
