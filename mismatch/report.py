"""Stage 3 (CPU): turn per-view logprob arrays into the measurement report.

Outputs results/measure_<model_tag>.json and docs/RESULTS_MEASURE_<tag>.md.

Sections
  M1  reference noise floor (fp32 eager vs fp32 sdpa, vs fp64 subset, vs TF32)
  errors   every view's error against the reference (rms, bias, band violations)
  headline the pairs an RL practitioner cares about (trainer vs sampler etc.)
  attribution  one-toggle deltas per family, dtype-vs-kernel split,
               independence check (gate M3)
  M2  the documented phenomenon reproduces (gap >> noise floor)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from mismatch import metrics as M
from mismatch import views as V
from mismatch.common import corpus_paths, read_corpus
from mismatch.store import View, flatten_pair, save_view


def sampler_view(model, variant=""):
    """Materialise the stage-1 sampler logprobs as a view file (idempotent)."""
    d, corpus, meta = corpus_paths(model, variant)
    out = C.SCORES_DIR / C.model_tag(model) / (variant or "") / ("vllm_sample.npz")
    if not out.exists():
        rows = read_corpus(corpus)
        m = json.load(open(meta)) if meta.exists() else {}
        save_view(out, [r["sampler_logprobs"] for r in rows], [r["id"] for r in rows],
                  {"family": "vllm", "path": "decode(sampler)", **m.get("vllm", {})})
    return View(out)


def load_views(model, variant=""):
    d = C.SCORES_DIR / C.model_tag(model) / (variant or "")
    out = {}
    for p in sorted(d.glob("*.npz")):
        out[p.stem] = View(p)
    out["vllm_sample"] = sampler_view(model, variant)
    return out


def pair(views, a, b, ref=None, sampled_from_b=None):
    if a not in views or b not in views:
        return None
    la, lb, rows, pos, n = flatten_pair(views[a], views[b])
    if sampled_from_b is None:
        sampled_from_b = b.startswith("vllm_sample")
    p_ref = ent = None
    if ref is not None and ref in views:
        lr, la2, rows_r, _, _ = flatten_pair(views[ref], views[a])
        if len(lr) == len(la):
            p_ref = np.exp(lr.astype(np.float64))
            if "entropy" in views[ref].extra and len(views[ref].extra["entropy"]) == len(views[ref].logp):
                # entropy is stored in ref row order; flatten_pair kept ref order
                ids, ra, _ = __import__("mismatch.store", fromlist=["align"]).align(views[ref], views[a])
                ent = np.concatenate([views[ref].extra["entropy"][views[ref].offsets[i]:views[ref].offsets[i + 1]] for i in ra])
    rep = M.pair_report(la, lb, rows, pos, n, p_ref=p_ref, entropy=ent, sampled_from_b=sampled_from_b)
    rep["a"], rep["b"], rep["n_rows"] = a, b, n
    return rep


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-2 or abs(x) >= 1e4 else f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL_ID)
    args = ap.parse_args()
    tag = C.model_tag(args.model)
    views = load_views(args.model)
    try:
        views_bi = load_views(args.model, "bi")
    except FileNotFoundError:
        views_bi = {}
    ref = V.REF
    have = sorted(views)
    print(f"[report] {tag}: {len(have)} views: {have}")
    R = {"model": args.model, "views": have, "corpus_meta": json.load(open(corpus_paths(args.model)[2]))}

    # ---- M1: is the reference a reference? -----------------------------------
    m1 = {}
    for other in ("hf_fp32_sdpa", "hf_fp64_eager", "hf_fp32_tf32"):
        p = pair(views, other, ref, sampled_from_b=False)
        if p:
            m1[other] = {"mean_abs_gap": p["token"]["mean_abs_gap"], "rms_gap": p["token"]["rms_gap"],
                         "max_abs_gap": p["token"]["max_abs_gap"], "n_tokens": p["token"]["n_tokens"]}
    noise = max([m1[k]["mean_abs_gap"] for k in ("hf_fp32_sdpa", "hf_fp64_eager") if k in m1] or [float("nan")])
    m1["noise_floor_mean_abs"] = noise
    m1["pass"] = bool(noise < C.M1_REF_NOISE_MAX) if not np.isnan(noise) else None
    R["M1"] = m1

    # ---- every view's error vs reference ---------------------------------------
    errs = {}
    for name in have:
        if name == ref:
            continue
        p = pair(views, name, ref, ref=ref, sampled_from_b=False)
        if p:
            t = p["token"]
            vw = views[name]
            errs[name] = {"toggle": (V.by_name(name)["toggle"] if name in {v["name"] for v in V.VIEWS} else "sampler (decode path)"),
                          "family": vw.meta.get("family", "?"),
                          "rms_err": t["rms_gap"], "mean_abs_err": t["mean_abs_gap"], "bias": t["mean_gap"],
                          "p99_abs_err": t["abs_gap_quantiles"]["p99"], "max_abs_err": t["max_abs_gap"],
                          "frac_outside_band": t["frac_outside_band"], "n_tokens": t["n_tokens"],
                          "by_ref_prob": p.get("by_ref_prob"), "by_position": p.get("by_position")}
    R["errors_vs_ref"] = errs

    # ---- headline pairs ----------------------------------------------------------
    head = []
    for a, b, desc in V.HEADLINE_PAIRS:
        vs = views
        if b == "vllm_sample_bi":
            vs, b = views_bi, "vllm_sample"
        p = pair(vs, a, b, ref=ref)
        if p:
            p["desc"] = desc
            if b == "vllm_sample" and vs is views_bi:
                p["corpus"] = "bi"
            head.append(p)
    R["headline"] = head

    # ---- attribution -------------------------------------------------------------
    att = {"per_toggle": []}
    hb, vb = V.baseline_of("hf")["name"], V.baseline_of("vllm")["name"]
    base = pair(views, hb, vb, ref=ref, sampled_from_b=False)
    if base:
        att["baseline_gap"] = {"a": hb, "b": vb, "rms": base["token"]["rms_gap"],
                               "mean_abs": base["token"]["mean_abs_gap"],
                               "frac_outside_band": base["token"]["frac_outside_band"]}
        for v in V.VIEWS:
            n = v["name"]
            if n not in views or n in (hb, vb, ref) or v.get("limit"):
                continue
            fam = v["scorer"]
            other = vb if fam == "hf" else hb
            g = pair(views, n, other, sampled_from_b=False) if fam == "hf" else pair(views, other, n, sampled_from_b=False)
            e = errs.get(n, {}).get("rms_err")
            e0 = errs.get(hb if fam == "hf" else vb, {}).get("rms_err")
            att["per_toggle"].append({
                "view": n, "family": fam, "toggle": v["toggle"],
                "gap_rms_vs_other_baseline": g["token"]["rms_gap"] if g else None,
                "gap_delta_vs_baseline": (g["token"]["rms_gap"] - base["token"]["rms_gap"]) if g else None,
                "err_rms_vs_ref": e, "err_delta_vs_family_baseline": (e - e0) if (e is not None and e0 is not None) else None,
                "frac_outside_band_vs_other": g["token"]["frac_outside_band"] if g else None})
        # dtype vs kernel split and the independence check (gate M3)
        if all(k in views for k in ("hf_fp32_eager", "vllm_fp32_fa", "hf_fp32_sdpa", "vllm_bf16_fa", hb)):
            la, lb, *_ = flatten_pair(views[hb], views[vb])
            lr, _, *_ = flatten_pair(views[ref], views[hb])
            dec = M.error_decomposition(la, lb, lr)
            pred = float(np.sqrt(dec["rms_err_a"] ** 2 + dec["rms_err_b"] ** 2))
            att["split"] = {
                "kernel_only_fp32_gap_rms": pair(views, "hf_fp32_eager", "vllm_fp32_fa", sampled_from_b=False)["token"]["rms_gap"],
                "trainer_dtype_effect_rms": errs[hb]["rms_err"],
                "trainer_kernel_effect_rms": errs["hf_fp32_sdpa"]["rms_err"],
                "infer_dtype_effect_rms": errs[vb]["rms_err"] if vb in errs else None,
                "infer_kernel_effect_rms": errs["vllm_fp32_fa"]["rms_err"],
                "decomposition": dec,
                "gap_predicted_if_independent": pred,
                "gap_measured": dec["rms_gap"],
                "independence_rel_err": abs(pred - dec["rms_gap"]) / max(dec["rms_gap"], 1e-12)}
            att["split"]["M3_pass"] = bool(att["split"]["independence_rel_err"] < C.M3_ADDITIVITY_TOL)
    R["attribution"] = att

    # ---- M2 -----------------------------------------------------------------------
    hp = next((h for h in head if h["a"] == hb and h["b"] == "vllm_sample" and "corpus" not in h), None)
    if hp and not np.isnan(noise):
        ratio = hp["token"]["mean_abs_gap"] / max(noise, 1e-12)
        R["M2"] = {"gap_mean_abs": hp["token"]["mean_abs_gap"], "noise_floor": noise, "ratio": ratio,
                   "kl_k3": hp["token"].get("kl_k3"), "pass": bool(ratio >= C.M2_MIN_RATIO_OVER_NOISE)}

    out = C.publish_result(f"measure_{tag}", R)
    write_md(R, tag)
    print(f"[report] wrote {out}")


def write_md(R, tag):
    L = [f"# Measurement report: {R['model']}", ""]
    cm = R["corpus_meta"]
    L += [f"Corpus: {cm['n']} gsm8k-train prompts, {cm['tokens']} sampled tokens "
          f"(mean {cm['mean_len']:.0f}/seq, {cm['finish']['length']} length-capped), "
          f"T={cm['sampling']['temperature']} top_p={cm['sampling']['top_p']} top_k={cm['sampling']['top_k']}, "
          f"vLLM {cm['vllm']['version']} {cm['vllm']['attention_backend']}. Accuracy {cm['accuracy']:.3f}.", ""]
    m1 = R["M1"]
    L += ["## M1 reference noise floor", "", "| pair vs hf_fp32_eager | mean abs gap | rms | max | tokens |", "|---|---|---|---|---|"]
    for k, v in m1.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {fmt(v['mean_abs_gap'])} | {fmt(v['rms_gap'])} | {fmt(v['max_abs_gap'])} | {v['n_tokens']} |")
    L += ["", f"Noise floor (mean abs) = {fmt(m1['noise_floor_mean_abs'])}; gate M1 (< {C.M1_REF_NOISE_MAX}): **{'PASS' if m1['pass'] else 'FAIL'}**", ""]
    L += ["## Every view's error against the fp32 reference", "",
          "| view | family | toggle | rms err | mean abs | bias | p99 abs | max abs | frac outside band |", "|---|---|---|---|---|---|---|---|---|"]
    for n, e in sorted(R["errors_vs_ref"].items(), key=lambda kv: kv[1]["rms_err"]):
        L.append(f"| {n} | {e['family']} | {e['toggle']} | {fmt(e['rms_err'])} | {fmt(e['mean_abs_err'])} | {fmt(e['bias'])} | {fmt(e['p99_abs_err'])} | {fmt(e['max_abs_err'])} | {fmt(e['frac_outside_band'])} |")
    L += ["", "## Headline pairs", "",
          "| pair | rms gap | mean abs | KL k3 (if x~b) | frac outside band | frac r>2 | seq ESS frac | seq |S| p99 |", "|---|---|---|---|---|---|---|---|"]
    for h in R["headline"]:
        t, s = h["token"], h["sequence"]
        L.append(f"| {h['desc']} ({h['a']} vs {h['b']}{' @bi corpus' if h.get('corpus') else ''}) | {fmt(t['rms_gap'])} | {fmt(t['mean_abs_gap'])} | {fmt(t.get('kl_k3'))} | {fmt(t['frac_outside_band'])} | {fmt(t['frac_ratio_gt_tis_clip'])} | {fmt(s.get('seq_is_ess_frac'))} | {fmt(s.get('seq_abs_logratio_quantiles', {}).get('p99'))} |")
    hp = next((h for h in R["headline"] if h["b"] == "vllm_sample" and not h.get("corpus")), None)
    if hp and hp.get("by_ref_prob"):
        L += ["", "### Trainer-vs-sampler gap by reference token probability", "", "| p_ref bucket | tokens | mean abs gap | mean gap | frac outside band |", "|---|---|---|---|---|"]
        for b in hp["by_ref_prob"]:
            L.append(f"| {b['p_ref']} | {b['n']} | {fmt(b['mean_abs_gap'])} | {fmt(b['mean_gap'])} | {fmt(b['frac_outside_band'])} |")
        L += ["", "### ... by position", "", "| position | tokens | mean abs gap | mean gap |", "|---|---|---|---|"]
        for b in hp["by_position"]:
            L.append(f"| {b['pos']} | {b['n']} | {fmt(b['mean_abs_gap'])} | {fmt(b['mean_gap'])} |")
    att = R["attribution"]
    if att.get("baseline_gap"):
        bg = att["baseline_gap"]
        L += ["", "## Attribution (one toggle at a time)", "",
              f"Baseline gap {bg['a']} vs {bg['b']}: rms {fmt(bg['rms'])}, mean abs {fmt(bg['mean_abs'])}, outside band {fmt(bg['frac_outside_band'])}.", "",
              "| toggle | side | gap rms vs other baseline | delta gap | err rms vs ref | delta err |", "|---|---|---|---|---|---|"]
        for t in sorted(att["per_toggle"], key=lambda x: -(x["err_delta_vs_family_baseline"] or 0)):
            L.append(f"| {t['toggle']} ({t['view']}) | {t['family']} | {fmt(t['gap_rms_vs_other_baseline'])} | {fmt(t['gap_delta_vs_baseline'])} | {fmt(t['err_rms_vs_ref'])} | {fmt(t['err_delta_vs_family_baseline'])} |")
        if att.get("split"):
            s = att["split"]
            L += ["", "### dtype vs kernels", "",
                  f"- fp32 both sides (kernel-only residual): rms {fmt(s['kernel_only_fp32_gap_rms'])}",
                  f"- trainer side: bf16 error {fmt(s['trainer_dtype_effect_rms'])} vs sdpa@fp32 error {fmt(s['trainer_kernel_effect_rms'])}",
                  f"- inference side: bf16 error {fmt(s['infer_dtype_effect_rms'])} vs fp32-vLLM error {fmt(s['infer_kernel_effect_rms'])}",
                  f"- independence check: predicted gap sqrt(ea^2+eb^2) = {fmt(s['gap_predicted_if_independent'])}, measured {fmt(s['gap_measured'])}, corr(err_a, err_b) = {fmt(s['decomposition']['corr_err'])} -> gate M3 **{'PASS' if s['M3_pass'] else 'FAIL'}**"]
    if R.get("M2"):
        m2 = R["M2"]
        L += ["", "## M2 phenomenon reproduces", "",
              f"trainer-vs-sampler mean abs gap {fmt(m2['gap_mean_abs'])} = {m2['ratio']:.0f}x the noise floor; KL k3 = {fmt(m2['kl_k3'])} -> **{'PASS' if m2['pass'] else 'FAIL'}**"]
    p = C.REPO_ROOT / "docs" / f"RESULTS_MEASURE_{tag}.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    print(f"[report] wrote {p}")


if __name__ == "__main__":
    main()
