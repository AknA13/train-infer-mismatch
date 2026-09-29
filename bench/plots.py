"""Figures from results/*.json (CPU, matplotlib).

  python -m bench.plots            # writes docs/fig/*.png

Palette and marks follow the repo's dataviz conventions: one hue per family,
fixed order, thin marks, direct labels where <= 4 series, no dual axes.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

FIG = C.REPO_ROOT / "docs" / "fig"
FAM = {"hf": "#2F6DB5", "vllm": "#D9782D", "ref": "#6B7280"}   # trainer blue, inference orange, neutral
ARM_COLORS = ["#2F6DB5", "#D9782D", "#3B8F5A", "#9B4FB0", "#C23B3B", "#8A6D1F", "#3A9FA8", "#6B7280", "#B0578C"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)


def errors_vs_ref(R, tag):
    e = R["errors_vs_ref"]
    items = sorted(e.items(), key=lambda kv: kv[1]["rms_err"])
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(items) + 1.2))
    y = np.arange(len(items))
    ax.barh(y, [v["rms_err"] for _, v in items], color=[FAM.get(v["family"], "#999") for _, v in items], height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([f"{k}  ({v['toggle']})" for k, v in items], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("rms |log p_view − log p_fp32|  (per token)")
    ax.set_title(f"{tag}: every view's error against the fp32 reference", loc="left", fontsize=11)
    style(ax); ax.grid(axis="x", color="#E5E7EB", linewidth=0.6); ax.grid(axis="y", visible=False)
    for k, c in (("trainer (transformers)", FAM["hf"]), ("inference (vLLM)", FAM["vllm"])):
        ax.plot([], [], color=c, lw=6, label=k)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"errors_vs_ref_{tag}.png", dpi=160); plt.close(fig)


def gap_by_prob(R, tag):
    hp = next((h for h in R["headline"] if h["b"] == "vllm_sample" and not h.get("corpus")), None)
    if not hp or not hp.get("by_ref_prob"):
        return
    b = hp["by_ref_prob"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    x = np.arange(len(b))
    ax.bar(x, [v["mean_abs_gap"] for v in b], color=FAM["vllm"], width=0.62)
    ax.set_xticks(x); ax.set_xticklabels([v["p_ref"] for v in b], fontsize=8)
    ax.set_xlabel("reference token probability"); ax.set_ylabel("mean |gap| (trainer vs sampler)")
    ax.set_title("gap lives on low-probability tokens", loc="left", fontsize=10); style(ax)
    for xi, v in zip(x, b):
        ax.text(xi, v["mean_abs_gap"], f"n={v['n']}", ha="center", va="bottom", fontsize=7, color="#374151")
    ax = axes[1]
    p = hp["by_position"]
    ax.plot([i for i in range(len(p))], [v["mean_abs_gap"] for v in p], color=FAM["vllm"], lw=2, marker="o", ms=4)
    ax.set_xticks(range(len(p))); ax.set_xticklabels([v["pos"] for v in p], rotation=45, fontsize=7)
    ax.set_xlabel("position in completion"); ax.set_ylabel("mean |gap|")
    ax.set_title("... and does not grow with position", loc="left", fontsize=10); style(ax)
    fig.suptitle(f"{tag}: {hp['desc']}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / f"gap_by_prob_pos_{tag}.png", dpi=160); plt.close(fig)


def attribution(R, tag):
    att = R["attribution"]
    if not att.get("per_toggle"):
        return
    rows = sorted(att["per_toggle"], key=lambda t: (t["family"], -(t["err_delta_vs_family_baseline"] or 0)))
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(rows) + 1.4))
    y = np.arange(len(rows))
    vals = [t["err_delta_vs_family_baseline"] or 0 for t in rows]
    ax.barh(y, vals, color=[FAM[t["family"]] for t in rows], height=0.62)
    ax.axvline(0, color="#374151", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([t["toggle"] for t in rows], fontsize=8)
    ax.set_xlabel("change in rms error vs fp32 when this ONE toggle is applied to its side's baseline")
    ax.set_title(f"{tag}: one-toggle attribution (negative = closer to fp32)", loc="left", fontsize=11)
    style(ax); ax.grid(axis="x", color="#E5E7EB", linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / f"attribution_{tag}.png", dpi=160); plt.close(fig)


RL_GROUPS = {
    "corrections": ["nt_none", "nt_tis", "nt_mis", "nt_icepop", "nt_seqtis", "nt_vllmold"],
    "system_fixes": ["nt_none", "nt_fp16", "nt_fp32head", "nt_bi"],
    "seeds": ["nt_none", "nt_none_s1", "nt_tis", "nt_tis_s1", "nt_fp16", "nt_fp16_s1", "nt_vllmold", "nt_vllmold_s1"],
    "thinking": ["th_none", "th_tis", "th_fp16"],
}


def rl_curves():
    p = C.RESULTS_DIR / "rl_series.json"
    if not p.exists():
        return
    S = {s["arm"]: s for s in json.load(open(p))["arms"]}
    keys = [("acc", "train accuracy (reward), 10-step mean"), ("kl_k3", "KL k3 sampler‖trainer"),
            ("band", "frac tokens outside [0.8,1.25]"), ("entropy", "policy entropy (nats/token)"),
            ("grad_norm", "grad norm"), ("mean_len", "mean completion length")]
    for gname, arms in RL_GROUPS.items():
        arms = [a for a in arms if a in S]
        if not arms:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, (k, title) in zip(axes.ravel(), keys):
            for i, a in enumerate(arms):
                v = np.array(S[a][k], dtype=float)
                if k == "acc" and len(v) >= 10:
                    v = np.convolve(v, np.ones(10) / 10, mode="valid")
                ls = "--" if a.endswith("_s1") else "-"
                ax.plot(np.arange(len(v)), v, lw=1.6, ls=ls, color=ARM_COLORS[i % len(ARM_COLORS)], label=a)
            if k in ("kl_k3", "band", "grad_norm"):
                ax.set_yscale("log")
            ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("step"); style(ax)
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(f"GRPO arms: {gname}", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(); fig.savefig(FIG / f"rl_{gname}.png", dpi=160); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for p in sorted(C.RESULTS_DIR.glob("measure_*.json")):
        R = json.load(open(p)); tag = p.stem[len("measure_"):]
        errors_vs_ref(R, tag); gap_by_prob(R, tag); attribution(R, tag)
        print(f"[plots] {tag}")
    rl_curves()
    print(f"[plots] wrote {FIG}")


if __name__ == "__main__":
    main()
