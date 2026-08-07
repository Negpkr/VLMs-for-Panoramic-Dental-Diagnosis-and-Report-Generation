#!/usr/bin/env python3
"""Turn a comparison_*.json produced by vlm_comparison.py into a table and figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = Path("/home/s222393187/Dental/Results/model_comparison")


def latest_results(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("comparison_*.json"))
    if not files:
        raise SystemExit(f"no comparison_*.json in {results_dir}")
    return files[-1]


def build_table(payload: dict) -> pd.DataFrame:
    """Build summary table from patient-mean 32-tooth metrics."""
    rows = []
    for entry in payload["models"].values():
        metrics = entry.get("metrics") or {}
        tooth = metrics.get("per_tooth")
        image = metrics.get("image_level") or {}
        if not tooth:
            rows.append({"Model": entry.get("display_name", entry["model_key"]), "Status": "failed"})
            continue
        rows.append(
            {
                "Model": entry["display_name"],
                "Cases": metrics["n_cases"],
                "Strategy": payload.get("strategy") or payload.get("prompt_name", ""),
                "Aggregation": metrics.get("aggregation", "unknown"),
                "Tooth F1": tooth["f1_score"],
                "Tooth Precision": tooth["precision"],
                "Tooth Recall": tooth["recall"],
                "Tooth Specificity": tooth.get("specificity", float("nan")),
                "Tooth Accuracy": tooth["accuracy"],
                "Image F1": image.get("f1_score", float("nan")),
                "Mean pred/case": metrics["mean_predicted_count"],
                "Mean GT/case": metrics["mean_ground_truth_count"],
                "Sec/case": metrics["mean_seconds_per_case"],
                "Status": "ok",
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, out_path: Path) -> None:
    ok = df[df["Status"] == "ok"]
    if ok.empty:
        return
    metrics = [
        "Tooth F1",
        "Tooth Precision",
        "Tooth Recall",
        "Tooth Specificity",
        "Tooth Accuracy",
    ]
    metrics = [m for m in metrics if m in ok.columns]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(ok), 1)
    strategy = ""
    if "Strategy" in ok.columns and ok["Strategy"].notna().any():
        strategy = str(ok["Strategy"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (_, row) in enumerate(ok.iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=row["Model"])
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (patient mean)")
    title = "Missing-teeth detection (32 teeth/patient, then mean across patients)"
    if strategy:
        title = f"{title} [{strategy}]"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(payload: dict, out_path: Path) -> None:
    entries = [e for e in payload["models"].values() if (e.get("metrics") or {}).get("per_tooth")]
    if not entries:
        return
    fig, axes = plt.subplots(1, len(entries), figsize=(4 * len(entries), 3.8))
    if len(entries) == 1:
        axes = [axes]
    for ax, entry in zip(axes, entries):
        cm = np.array(entry["metrics"]["per_tooth"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(entry["display_name"], fontsize=10)
        ax.set_xticks([0, 1], ["pred present", "pred missing"], fontsize=8)
        ax.set_yticks([0, 1], ["true present", "true missing"], fontsize=8)
    fig.suptitle("Per-tooth confusion matrices")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_counts(payload: dict, out_path: Path) -> None:
    entries = [e for e in payload["models"].values() if (e.get("metrics") or {}).get("per_tooth")]
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for entry in entries:
        ok = [r for r in entry["results"] if "error" not in r]
        gt = [len(r["ground_truth_teeth"]) for r in ok]
        pred = [len(r["predicted_teeth"]) for r in ok]
        ax.scatter(gt, pred, alpha=0.6, s=28, label=entry["display_name"])
    lim = 33
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect agreement")
    ax.set_xlabel("Ground-truth missing teeth per case")
    ax.set_ylabel("Predicted missing teeth per case")
    ax.set_title("Predicted vs actual missing-teeth counts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    path = args.results or latest_results(args.out_dir)
    payload = json.loads(path.read_text())
    stamp = payload.get("timestamp", "latest")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_table(payload)
    csv_path = args.out_dir / f"summary_{stamp}.csv"
    df.to_csv(csv_path, index=False)

    plot_metrics(df, args.out_dir / f"metrics_{stamp}.png")
    plot_confusion(payload, args.out_dir / f"confusion_{stamp}.png")
    plot_counts(payload, args.out_dir / f"counts_{stamp}.png")

    print(f"source     : {path.name}")
    print(
        f"cases      : {payload['n_cases']}  strategy={payload.get('strategy') or payload.get('prompt_name')}  "
        f"aggregation={payload.get('aggregation', 'n/a')}"
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {csv_path.name}, metrics/confusion/counts_{stamp}.png -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
