#!/usr/bin/env python3
"""RQ3 report-comparison figures (PR-Reports expert vs. VLM reports).

Reads the augmented RQ3 results JSON (BLEU-4, ROUGE-1/2/L, BERTScore-F1,
KL divergence, JS divergence per case) and renders publication figures that
match the style of paper_figures.py (Okabe-Ito palette, 300 DPI, PDF + PNG).

    python rq3_figures.py
    python rq3_figures.py --results <path.json> --out-dir <dir>

Figures:
    fig_rq3_similarity   grouped bars, higher-is-better similarity metrics
    fig_rq3_divergence   per-case KL and JS divergence distributions (lower better)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Resolve paths relative to this file so the script works both in the main
# checkout and inside a git worktree.
ROOT = Path(__file__).resolve().parent
RESULTS_DEFAULT = ROOT / "Results" / "rq3_reports" / "rq3_pr_all_20260820_040310.json"
OUT_DEFAULT = ROOT / "Main Job Outputs" / "figures"

MODEL_ORDER = ["llava", "llava_med", "huatuogpt_vision", "dentvlm"]
MODEL_LABEL = {
    "llava": "LLaVA-1.5-7B",
    "llava_med": "LLaVA-Med",
    "huatuogpt_vision": "HuatuoGPT-Vision",
    "dentvlm": "DentVLM",
}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
DPI = 300

SIM_METRICS = [
    ("bleu4", "BLEU-4"),
    ("rouge_1_f1", "ROUGE-1"),
    ("rouge_2_f1", "ROUGE-2"),
    ("rouge_l_f1", "ROUGE-L"),
    ("bertscore_f1", "BERTScore-F1"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI)
        paths.append(path)
    plt.close(fig)
    return paths


def _load(results_path: Path) -> dict:
    payload = json.loads(results_path.read_text())
    models = payload["models"]
    data = {}
    for key in MODEL_ORDER:
        entry = models.get(key, {})
        summary = entry.get("summary", {})
        per_case = {"kl": [], "js": []}
        for case in entry.get("cases", []):
            m = case.get("metrics")
            if not m:
                continue
            if m.get("kl_divergence") is not None:
                per_case["kl"].append(m["kl_divergence"])
            if m.get("js_divergence") is not None:
                per_case["js"].append(m["js_divergence"])
        data[key] = {"summary": summary, "per_case": per_case}
    return data


def figure_similarity(data: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    n_models = len(MODEL_ORDER)
    n_metrics = len(SIM_METRICS)
    group_w = 0.8
    bar_w = group_w / n_models
    x = np.arange(n_metrics)

    for mi, mk in enumerate(MODEL_ORDER):
        vals = [data[mk]["summary"].get(f"{key}_mean") or 0.0 for key, _ in SIM_METRICS]
        offset = (mi - (n_models - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w * 0.94,
            label=MODEL_LABEL[mk], color=OKABE[mi], edgecolor="white", linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=6.3, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in SIM_METRICS])
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, max(0.75, ax.get_ylim()[1]))
    ax.set_title("RQ3: model vs. expert panoramic reports (n=121) — similarity metrics")
    ax.legend(ncol=4, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_similarity")


def figure_divergence(data: dict, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    labels = [MODEL_LABEL[mk] for mk in MODEL_ORDER]
    colors = [OKABE[i] for i in range(len(MODEL_ORDER))]

    panels = [
        ("kl", "KL divergence  D$_{KL}$(expert $\\parallel$ model)  [bits]",
         "KL divergence per case (lower is better)"),
        ("js", "JS divergence  [bits, 0–1]", "JS divergence per case (lower is better)"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        series = [data[mk]["per_case"][key] for mk in MODEL_ORDER]
        bp = ax.boxplot(series, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="#222222", linewidth=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#444444")
        for wk in ("whiskers", "caps"):
            for ln in bp[wk]:
                ln.set_color("#666666")
        # jittered points + mean marker
        rng = np.random.default_rng(20260820)
        for i, vals in enumerate(series, 1):
            jx = i + (rng.random(len(vals)) - 0.5) * 0.28
            ax.scatter(jx, vals, s=7, color=colors[i - 1], alpha=0.35,
                       edgecolors="none", zorder=2)
            ax.scatter([i], [np.mean(vals)], marker="D", s=34, color="white",
                       edgecolors="#222222", linewidths=1.1, zorder=4)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[1].set_ylim(0, 1)
    fig.suptitle("RQ3: token-distribution distance to expert reports (◆ = mean)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_divergence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    _style()
    data = _load(args.results)
    written = []
    written += figure_similarity(data, args.out_dir)
    written += figure_divergence(data, args.out_dir)
    for p in written:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
