#!/usr/bin/env python3
"""Publication-quality figures for the hallucination analysis.

Reads the CSVs written by hallucination/run_analysis.py and renders the figure
set requested in the brief. Because the project spans two datasets (DENTEX,
Tufts) x three strategies, figures are produced per (dataset, strategy) into
    figures/hallucination/<dataset>_<strategy>/
using the brief's suggested filenames. Every figure is saved as PNG (300 DPI)
and PDF.

Style (fonts, Okabe-Ito colours, 300 DPI, no top/right spines) matches the
repository's existing paper_figures.py so the new figures sit alongside the
current ones. Styling is deliberately plain, not decorative.

Usage (repo root):
    python -m hallucination.figures
    python -m hallucination.figures --csv-dir results/hallucination --out figures/hallucination
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the repository's model ordering, labels, palette, DPI (paper_figures.py).
MODEL_ORDER = ["llava", "llava_med", "huatuogpt_vision", "dentvlm"]
MODEL_LABEL = {
    "llava": "LLaVA-1.5-7B",
    "llava_med": "LLaVA-Med",
    "huatuogpt_vision": "HuatuoGPT-Vision",
    "dentvlm": "DentVLM",
}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
DPI = 300


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=DPI)
    plt.close(fig)


def _order_models(present: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in present]
    return known + [m for m in present if m not in MODEL_ORDER]


def _labels(models: list[str]) -> list[str]:
    return [MODEL_LABEL.get(m, m) for m in models]


def _bar_value_labels(ax, bars, values, fmt="{:.2f}") -> None:
    """Numeric value above each bar; skips NaN."""
    for rect, v in zip(bars, values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            txt = "NA"
            y = 0
        else:
            txt = fmt.format(v)
            y = rect.get_height()
        ax.annotate(txt, (rect.get_x() + rect.get_width() / 2, y),
                    ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")


# ---------------------------------------------------------------------------
# Figures for one (dataset, strategy)
# ---------------------------------------------------------------------------


def figures_for_run(
    summary: pd.DataFrame,
    conditions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    dataset: str,
    strategy: str,
    out_dir: Path,
) -> list[str]:
    made: list[str] = []
    models = _order_models(list(summary["Model"]))
    labels = _labels(models)
    idx = {m: summary[summary.Model == m].iloc[0] for m in models}
    title_sfx = f"{dataset} / {strategy}"

    # --- FIGURE 1: CHR by model (with bootstrap 95% CI error bars) ----------
    chr_vals = [idx[m]["CHR"] for m in models]
    boot = {r["Model"]: r for _, r in bootstrap.iterrows()}
    yerr = np.zeros((2, len(models)))
    have_ci = False
    for i, m in enumerate(models):
        row = boot.get(m)
        pt = chr_vals[i]
        if row is not None and not math.isnan(pt) \
                and not math.isnan(row.get("CHR_CI95_low", math.nan)):
            yerr[0, i] = max(0.0, pt - row["CHR_CI95_low"])
            yerr[1, i] = max(0.0, row["CHR_CI95_high"] - pt)
            have_ci = True
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_vals = [0 if (v is None or math.isnan(v)) else v for v in chr_vals]
    bars = ax.bar(labels, plot_vals, color=OKABE[0],
                  yerr=yerr if have_ci else None, capsize=3, ecolor="#444444")
    _bar_value_labels(ax, bars, chr_vals)
    ax.set_ylabel("Clinical Hallucination Rate (FP / predicted)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Clinical Hallucination Rate by model — {title_sfx}\n(lower is better; "
                 f"bars with error = image-level bootstrap 95% CI)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, out_dir, "hallucination_rate_by_model")
    made.append("hallucination_rate_by_model")

    # --- FIGURE 2: Precision / Recall / F1 grouped bars ---------------------
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, met in enumerate(metrics):
        vals = [idx[m][met] for m in models]
        pv = [0 if (v is None or math.isnan(v)) else v for v in vals]
        b = ax.bar(x + (j - 1) * w, pv, w, label=met, color=OKABE[j])
        _bar_value_labels(ax, b, vals, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Precision, Recall, F1 (finding level) — {title_sfx}")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, out_dir, "precision_recall_f1_by_model")
    made.append("precision_recall_f1_by_model")

    # --- FIGURE 3: Hallucinations per image ---------------------------------
    vals = [idx[m]["Hallucinations_per_image"] for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [0 if math.isnan(v) else v for v in vals], color=OKABE[3])
    _bar_value_labels(ax, bars, vals, fmt="{:.2f}")
    ax.set_ylabel("Mean false-positive findings per image")
    ax.set_title(f"Hallucinations per image — {title_sfx} (lower is better)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, out_dir, "hallucinations_per_image_by_model")
    made.append("hallucinations_per_image_by_model")

    # --- FIGURE 4: % images with >=1 hallucination --------------------------
    vals = [idx[m]["Hallucination_image_rate"] * 100 for m in models]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [0 if math.isnan(v) else v for v in vals], color=OKABE[4])
    _bar_value_labels(ax, bars, vals, fmt="{:.1f}")
    ax.set_ylabel("% images with ≥1 hallucinated finding")
    ax.set_ylim(0, 105)
    ax.set_title(f"Images containing hallucinations — {title_sfx}")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, out_dir, "hallucination_image_rate")
    made.append("hallucination_image_rate")

    # --- FIGURE 5: TP / FP / FN grouped bars (TN excluded on purpose) -------
    comps = ["TP", "FP", "FN"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, c in enumerate(comps):
        vals = [idx[m][c] for m in models]
        b = ax.bar(x + (j - 1) * w, vals, w, label=c, color=OKABE[j])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Finding count")
    ax.set_title(f"Confusion components (TP / FP / FN) — {title_sfx}\n"
                 f"TN omitted: it dwarfs these on the negative-heavy slot grid")
    ax.legend(frameon=False, ncol=3)
    _save(fig, out_dir, "tp_fp_fn_by_model")
    made.append("tp_fp_fn_by_model")

    # --- FIGURE 8: Hallucination (CHR) vs Recall scatter --------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, m in enumerate(models):
        r, c = idx[m]["Recall"], idx[m]["CHR"]
        if math.isnan(r) or math.isnan(c):
            continue
        ax.scatter(r, c, s=70, color=OKABE[i % len(OKABE)], zorder=3)
        ax.annotate(MODEL_LABEL.get(m, m), (r, c), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Recall / Sensitivity (finding level)")
    ax.set_ylabel("Clinical Hallucination Rate")
    ax.set_title(f"Hallucination vs recall — {title_sfx}\n"
                 f"ideal = high recall (right), low CHR (bottom)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "hallucination_vs_recall")
    made.append("hallucination_vs_recall")

    return made


# ---------------------------------------------------------------------------
# Cross-model condition heatmaps (DENTEX; multiple conditions)
# ---------------------------------------------------------------------------


def condition_heatmaps(
    conditions: pd.DataFrame,
    dataset: str,
    strategy: str,
    out_dir: Path,
) -> list[str]:
    made: list[str] = []
    cond_names = list(dict.fromkeys(conditions["Condition"]))
    if len(cond_names) < 2:
        # Single-condition task (Tufts): a model x condition heatmap is a single
        # column and adds nothing over Figure 3. Skip and document.
        (out_dir / "heatmaps_skipped.txt").write_text(
            f"Condition heatmaps skipped for {dataset}/{strategy}: only one "
            f"condition ('{cond_names[0] if cond_names else 'n/a'}'), so a "
            f"model x condition matrix is a single column (see "
            f"hallucinations_per_image_by_model instead)."
        )
        return made

    models = _order_models(list(dict.fromkeys(conditions["Model"])))
    labels = _labels(models)
    count = np.zeros((len(models), len(cond_names)))
    rate = np.full((len(models), len(cond_names)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(cond_names):
            sub = conditions[(conditions.Model == m) & (conditions.Condition == c)]
            if len(sub):
                count[i, j] = sub.iloc[0]["FP"]
                rate[i, j] = sub.iloc[0]["CHR"]

    # FIGURE 6a: hallucination COUNT heatmap.
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cond_names), 0.9 + 0.7 * len(models)))
    im = ax.imshow(count, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(cond_names)), cond_names, rotation=20, ha="right")
    ax.set_yticks(range(len(models)), labels)
    for i in range(len(models)):
        for j in range(len(cond_names)):
            ax.text(j, i, f"{int(count[i, j])}", ha="center", va="center",
                    fontsize=8, color="black" if count[i, j] < count.max() / 2 else "white")
    fig.colorbar(im, ax=ax, label="Hallucinated (FP) findings")
    ax.set_title(f"Hallucination COUNT by condition — {dataset} / {strategy}")
    _save(fig, out_dir, "hallucination_count_heatmap")
    made.append("hallucination_count_heatmap")

    # FIGURE 6b: hallucination RATE (condition CHR) heatmap.
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(cond_names), 0.9 + 0.7 * len(models)))
    im = ax.imshow(rate, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cond_names)), cond_names, rotation=20, ha="right")
    ax.set_yticks(range(len(models)), labels)
    for i in range(len(models)):
        for j in range(len(cond_names)):
            v = rate[i, j]
            ax.text(j, i, "NA" if math.isnan(v) else f"{v:.2f}", ha="center",
                    va="center", fontsize=8,
                    color="black" if (math.isnan(v) or v < 0.5) else "white")
    fig.colorbar(im, ax=ax, label="Condition CHR = FP/(TP+FP)")
    ax.set_title(f"Hallucination RATE by condition — {dataset} / {strategy}")
    _save(fig, out_dir, "hallucination_rate_heatmap")
    made.append("hallucination_rate_heatmap")

    # FIGURE 7: top hallucinated conditions (total FP across models). DENTEX
    # has 4 conditions, so all are shown, ranked; the "top 10" cap applies when
    # a dataset has more.
    totals = count.sum(axis=0)
    order = np.argsort(totals)[::-1][:10]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([cond_names[k] for k in order], [totals[k] for k in order],
                  color=OKABE[3])
    for rect, k in zip(bars, order):
        ax.annotate(f"{int(totals[k])}", (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_ylabel("Total FP findings across models")
    ax.set_title(f"Top hallucinated conditions — {dataset} / {strategy}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, out_dir, "top_hallucinated_conditions")
    made.append("top_hallucinated_conditions")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default="results/hallucination", type=Path)
    ap.add_argument("--out", default="figures/hallucination", type=Path)
    args = ap.parse_args()
    _style()

    summary = pd.read_csv(args.csv_dir / "model_hallucination_summary.csv")
    conditions = pd.read_csv(args.csv_dir / "condition_hallucination_summary.csv")
    bootstrap = pd.read_csv(args.csv_dir / "bootstrap_confidence_intervals.csv")

    made_all: list[str] = []
    for (dataset, strategy), sub in summary.groupby(["Dataset", "Strategy"], sort=False):
        out_dir = args.out / f"{dataset}_{strategy}"
        cond_sub = conditions[(conditions.Dataset == dataset) & (conditions.Strategy == strategy)]
        boot_sub = bootstrap[(bootstrap.Dataset == dataset) & (bootstrap.Strategy == strategy)]
        made = figures_for_run(sub, cond_sub, boot_sub, dataset, strategy, out_dir)
        made += condition_heatmaps(cond_sub, dataset, strategy, out_dir)
        print(f"{dataset}/{strategy}: {len(made)} figures -> {out_dir}")
        made_all.extend(f"{out_dir}/{s}" for s in made)

    # FIGURE 9 (precision vs CHR) is intentionally NOT generated: under the
    # binary matching rule CHR == 1 - Precision exactly, so the scatter is a
    # straight anti-diagonal line and adds no information beyond Figure 8.
    note = args.out / "FIGURE9_not_generated.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "Figure 9 (Precision vs CHR) was intentionally not generated.\n"
        "Under this project's binary finding-matching rule, CHR = FP/(TP+FP) = "
        "1 - Precision exactly, so a precision-vs-CHR scatter is a deterministic "
        "anti-diagonal (y = 1 - x) and carries no information beyond the "
        "precision bars (Figure 2) and the hallucination-vs-recall scatter "
        "(Figure 8). It is documented here rather than drawn.\n"
    )
    print(f"\nTotal figures written: {len(made_all)}")
    print("Figure 9 skipped (CHR == 1 - Precision); see", note)


if __name__ == "__main__":
    main()
