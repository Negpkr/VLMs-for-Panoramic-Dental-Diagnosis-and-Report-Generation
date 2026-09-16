#!/usr/bin/env python3
"""Publication figures for the Tufts + DENTEX VLM study.

Figure 1  End-to-end pipeline (no model run required)
Figure 2  Dataset / class support + DENTEX multi-label inset
Figure 3  Model × strategy Macro F1 (Tufts, DENTEX abnormality, DENTEX Macro-4)
Figure 4  DENTEX per-disease F1 heatmap (+ binary heatmaps)
Figure 5  Qualitative successes and failures
Figure 6  DENTEX abnormal vs no-finding 2×2 confusion matrices (12 model × strategy cells)

After the qed matrix finishes:
    python paper_figures.py --dentex-split qed
    # writes to Main Job Outputs/figures/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import eval_metrics  # noqa: E402

ROOT = Path("/home/s222393187/Dental")
TUFTS_DIR = ROOT / "Results" / "model_comparison"
DENTEX_DIR = ROOT / "Results" / "dentex_comparison"
VAL_JSON = ROOT / "DENTEX" / "validation_triple.json"
TRAIN_JSON = (
    ROOT
    / "DENTEX"
    / "training_data"
    / "quadrant-enumeration-disease"
    / "train_quadrant_enumeration_disease.json"
)
VAL_XRAYS = ROOT / "DENTEX" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TRAIN_XRAYS = ROOT / "DENTEX" / "training_data" / "quadrant-enumeration-disease" / "xrays"
TUFTS_XRAYS = ROOT / "Tufts_Dental_Database" / "Radiographs"
OUT_DEFAULT = ROOT / "Main Job Outputs" / "figures"

DISEASES = ["Impacted", "Caries", "Deep Caries", "Periapical Lesion"]
DISEASE_SHORT = {
    "Impacted": "Impacted",
    "Caries": "Caries",
    "Deep Caries": "Deep Caries",
    "Periapical Lesion": "Periapical",
}
MODEL_ORDER = ["llava", "llava_med", "huatuogpt_vision", "dentvlm"]
MODEL_LABEL = {
    "llava": "LLaVA-1.5-7B",
    "llava_med": "LLaVA-Med",
    "huatuogpt_vision": "HuatuoGPT-Vision",
    "dentvlm": "DentVLM",
}
STRAT_ORDER = ["zero_shot", "few_shot", "cot"]
STRAT_LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "cot": "CoT"}
TUFTS_JSON_FALLBACK = {
    "zero_shot": TUFTS_DIR / "comparison_zero_shot_full_20260820_103338.json",
    "few_shot": TUFTS_DIR / "comparison_few_shot_full_20260820_120337.json",
    "cot": TUFTS_DIR / "comparison_cot_full_20260820_134750.json",
}

OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
DPI = 300


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


def _caption(out_dir: Path, stem: str, text: str) -> Path:
    path = out_dir / f"{stem}_caption.txt"
    path.write_text(text.strip() + "\n")
    return path


def load_qed_findings(json_path: Path) -> tuple[dict[str, dict[int, list[str]]], dict]:
    payload = json.loads(json_path.read_text())
    c1 = {c["id"]: str(c["name"]) for c in payload["categories_1"]}
    c2 = {c["id"]: str(c["name"]) for c in payload["categories_2"]}
    c3 = {c["id"]: str(c["name"]) for c in payload["categories_3"]}
    by_img: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in payload["annotations"]:
        fdi = int(f"{c1[a['category_id_1']]}{c2[a['category_id_2']]}")
        disease = c3[a["category_id_3"]]
        by_img[a["image_id"]][fdi].add(disease)
    id_to_name = {im["id"]: im["file_name"] for im in payload["images"]}
    cases = {}
    for im in payload["images"]:
        stem = Path(im["file_name"]).stem
        teeth = by_img.get(im["id"], {})
        cases[stem] = {int(fdi): sorted(labs) for fdi, labs in teeth.items()}
    meta = {"id_to_name": id_to_name, "n_images_json": len(payload["images"]), "images": payload["images"]}
    return cases, meta


def dentex_support_stats() -> dict[str, Any]:
    val, val_meta = load_qed_findings(VAL_JSON)
    train, train_meta = load_qed_findings(TRAIN_JSON)
    all_cases = {**train, **val}
    n_images = len(all_cases)
    n_slots = n_images * 32
    disease_n = Counter()
    n_abnormal = 0
    multi = Counter()
    for teeth in all_cases.values():
        n_abnormal += len(teeth)
        for labs in teeth.values():
            multi[len(labs)] += 1
            for lab in labs:
                disease_n[lab] += 1
    return {
        "n_images": n_images,
        "n_slots": n_slots,
        "n_abnormal": n_abnormal,
        "n_no_finding": n_slots - n_abnormal,
        "disease_n": dict(disease_n),
        "multi": dict(multi),
        "n_train": len(train),
        "n_val": len(val),
        "val_cases": val,
        "val_meta": val_meta,
        "train_meta": train_meta,
    }


def latest_tufts_json(strategy: str) -> Path:
    files = sorted(TUFTS_DIR.glob(f"comparison_{strategy}_full_*.json"))
    done = []
    for p in files:
        log = p.with_suffix(".log")
        if log.exists() and "COMPARISON_DONE" in log.read_text(errors="ignore"):
            done.append(p)
    if done:
        return done[-1]
    return TUFTS_JSON_FALLBACK[strategy]


def tufts_support_from_bbox() -> dict[str, int]:
    path = ROOT / "Tufts_Dental_Database" / "Segmentation" / "teeth_bbox.json"
    records = json.loads(path.read_text())
    if isinstance(records, dict):
        records = list(records.values())
    missing = present = 0
    n = 0
    for record in records:
        detected: set[int] = set()
        for obj in record.get("Label", {}).get("objects", []):
            for tok in re.findall(r"\d+", str(obj.get("title", ""))):
                if 1 <= int(tok) <= 32:
                    detected.add(int(tok))
        missing += 32 - len(detected)
        present += len(detected)
        n += 1
    return {"missing": missing, "present": present, "n_cases": n}


def latest_dentex_json(split: str, strategy: str) -> Path | None:
    files = sorted(DENTEX_DIR.glob(f"dentex_{strategy}_{split}_*.json"))
    files = [p for p in files if p.with_suffix(".log").exists()]
    done = []
    for p in files:
        log = p.with_suffix(".log")
        if "DENTEX_COMPARISON_DONE" in log.read_text(errors="ignore"):
            done.append(p)
    if done:
        return done[-1]
    return files[-1] if files else None


def load_dentex_matrix(split: str) -> dict[str, dict[str, dict]] | None:
    matrix: dict[str, dict[str, dict]] = {}
    for strat in STRAT_ORDER:
        path = latest_dentex_json(split, strat)
        if path is None:
            return None
        payload = json.loads(path.read_text())
        if len(payload.get("models") or {}) < 4:
            return None
        if not all((payload["models"].get(k) or {}).get("metrics", {}).get("abnormal_tooth") for k in MODEL_ORDER):
            return None
        matrix[strat] = payload
    return matrix


def tufts_pooled_macro_f1(entry: dict) -> dict[str, float]:
    yt, yp = [], []
    for r in entry.get("results") or []:
        if "error" in r:
            continue
        yt.extend(r["y_true_32"])
        yp.extend(r["y_pred_32"])
    block = eval_metrics.pooled_metrics(yt, yp, labels=(0, 1), names=("present", "missing"))
    pc = block["per_class"]
    return {
        "macro_f1": float(block["macro"]["f1_score"]),
        "missing_f1": float(pc["missing"]["f1_score"]),
        "present_f1": float(pc["present"]["f1_score"]),
        "abnormal_f1": float(pc["missing"]["f1_score"]),
        "no_finding_f1": float(pc["present"]["f1_score"]),
    }


def dentex_scores(entry: dict) -> dict[str, float]:
    m = entry["metrics"]
    ab = m["abnormal_tooth"]
    path = eval_metrics.flatten_pathology_report(m.get("disease_multiclass") or {}, DISEASES)
    bin_row = eval_metrics.flatten_binary_report(ab, "abnormal", eval_metrics.NO_ANNOTATED_FINDING)
    out = {
        "macro_f1_abn": float(bin_row["Macro_F1"] or 0),
        "abnormal_f1": float(bin_row["abnormal_F1"] or 0),
        "no_finding_f1": float(bin_row[eval_metrics.csv_class_key(eval_metrics.NO_ANNOTATED_FINDING) + "_F1"] or 0),
        "macro4_f1": float(path["Macro4_F1"] or 0),
    }
    for d in DISEASES:
        val = path[eval_metrics.csv_class_key(d) + "_F1"]
        out[d] = float("nan") if val == "N/A" else float(val or 0)
    return out


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def figure1(out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.set_title("End-to-end evaluation pipeline (pooled 32-slot scoring)", pad=8, fontweight="bold")

    def box(x, y, w, h, text, fc="#F7F7F7", ec="#333333", size=9, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=fc, edgecolor=ec, linewidth=1.1,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                linewidth=1.2, color="#333333",
            )
        )

    box(3.35, 6.35, 4.5, 0.85, "Panoramic radiograph", fc="#DEEBF7", weight="bold")
    arrow(5.6, 6.35, 5.6, 5.95)
    box(0.4, 4.85, 4.9, 1.05, "Tufts: 32 Universal slots  (1–32)\nMissing vs Present", fc="#E8F5E9")
    box(5.9, 4.85, 4.9, 1.05, "DENTEX: 32 FDI slots  (11–48)\nAbnormal vs No annotated finding", fc="#FFF3E0")
    arrow(5.6, 4.85, 5.6, 4.45)
    box(1.6, 3.45, 8.0, 0.95,
        "VLM  ·  LLaVA-1.5-7B  ·  LLaVA-Med  ·  HuatuoGPT-Vision  ·  DentVLM\n"
        "Prompting strategy  ·  zero-shot  ·  few-shot  ·  chain-of-thought",
        fc="#F3E5F5", weight="bold", size=9)
    arrow(5.6, 3.45, 5.6, 3.05)
    box(2.2, 2.15, 6.8, 0.85,
        "Parser → tooth-level predictions\n"
        "Abnormality from parsed FDI/Universal findings  (not from Total-count lines)",
        fc="#F7F7F7", size=8.5)
    arrow(5.6, 2.15, 5.6, 1.75)

    box(0.25, 0.25, 3.4, 1.4,
        "Table A / Tufts binary\nPooled TP/FP/FN/TN\nHeadline: Macro F1\nAlso: Acc, BalAcc",
        fc="#E8F5E9", size=8.2)
    box(3.9, 0.25, 3.4, 1.4,
        "DENTEX Table A\nAbnormal vs no finding\nHeadline: Macro F1\nAlso: Acc, BalAcc",
        fc="#FFF3E0", size=8.2)
    box(7.55, 0.25, 3.4, 1.4,
        "DENTEX Table B  (multi-label)\n4 independent one-vs-rest\nImpacted · Caries\nDeep Caries · Periapical\nHeadline: Macro-4 F1",
        fc="#FFECB3", size=8.0)

    # Multi-label callout
    ax.add_patch(
        FancyBboxPatch(
            (7.55, 1.75), 3.4, 0.95, boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor="#FFFDE7", edgecolor="#D55E00", linewidth=1.2, linestyle="--",
        )
    )
    ax.text(
        9.25, 2.22,
        "One tooth, two findings\nFDI 46: Caries + Periapical\n→ yes in both OvR classifiers",
        ha="center", va="center", fontsize=7.6, color="#333333",
    )
    paths = _save(fig, out_dir, "Figure1_evaluation_pipeline")
    _caption(
        out_dir,
        "Figure1_evaluation_pipeline",
        "Evaluation pipeline. Each panoramic radiograph is scored over 32 tooth slots "
        "(Universal 1–32 on Tufts; FDI 11–48 on DENTEX). Four VLMs are run under zero-shot, "
        "few-shot, and chain-of-thought prompts. Predictions are parsed to tooth-level findings; "
        "abnormality is derived from those findings, not from a Total abnormal count. "
        "Tufts and DENTEX Table A are binary pooled tasks (headline Macro F1, plus Accuracy "
        "and Balanced Accuracy). DENTEX Table B "
        "scores Impacted, Caries, Deep Caries, and Periapical Lesion as independent one-vs-rest "
        "classifiers because one tooth may carry more than one diagnosis (headline Macro-4 F1).",
    )
    return paths


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def figure2(out_dir: Path) -> list[Path]:
    counts_path = ROOT / "runs" / "dataset_counts.json"
    if not counts_path.exists():
        import dataset_inventory

        dataset_inventory.build_inventory()
    inv = json.loads(counts_path.read_text())
    tufts = inv["tufts"]["paper_evaluation"]
    dx = inv["dentex"]["qed_paper_evaluation"]
    disease_n = dx["disease_support"]
    multi = {int(k): v for k, v in (dx.get("multi_label_teeth") or {}).items()}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.15), gridspec_kw={"width_ratios": [1.05, 1.35, 1.0]})

    ax = axes[0]
    vals = [tufts["missing_slots"], tufts["present_slots"]]
    bars = ax.bar(["Missing", "Present"], vals, color=[OKABE[3], OKABE[0]], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Tooth slots (pooled)")
    ax.set_title(f"Tufts paper set  ({tufts['n_cases']} cases × 32)")
    ax.set_ylim(0, max(vals) * 1.15)

    ax = axes[1]
    names = [
        "Unique\nabnormal\nteeth",
        "No annotated\nfinding",
        "Impacted\nGT+ slots",
        "Caries\nGT+ slots",
        "Deep Caries\nGT+ slots",
        "Periapical\nGT+ slots",
    ]
    counts_v = [dx["n_abnormal_teeth"], dx["n_no_finding_slots"]] + [disease_n.get(d, 0) for d in DISEASES]
    colors = [OKABE[3], OKABE[0], OKABE[2], OKABE[1], OKABE[5], OKABE[4]]
    bars = ax.bar(names, counts_v, color=colors, width=0.72)
    for b, v in zip(bars, counts_v):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(f"DENTEX QED scored  ({dx['n_images']} images; 3 train exemplars held out)")
    ax.set_ylabel("Count  (disease bars = GT-positive slots)")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylim(0, max(counts_v) * 1.22)
    ax.axvline(1.5, color="#666666", linewidth=0.8, linestyle="--", alpha=0.7)

    ax = axes[2]
    xs = [1, 2, 3]
    ys = [multi.get(k, 0) for k in xs]
    bars = ax.bar(xs, ys, color=OKABE[1], width=0.55)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("Diagnoses on the same tooth")
    ax.set_ylabel("Annotated teeth")
    ax.set_title("DENTEX is multi-label")
    ax.set_ylim(0, max(ys) * 1.2 if max(ys) else 1)

    fig.suptitle("Dataset and annotation support  (why accuracy is not the headline)", y=1.02, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, out_dir, "Figure2_dataset_support")
    _caption(
        out_dir,
        "Figure2_dataset_support",
        "Class support for the paper evaluation sets. Tufts: all 1000 usable cases × 32 Universal "
        "slots (the earlier 100-case runs are archive only). DENTEX: QED train+val minus 3 held-out "
        f"train text-exemplars ({dx['n_images']} images). The first two DENTEX bars are mutually "
        f"exclusive 32-slot counts: unique abnormal teeth {dx['n_abnormal_teeth']:,} + no-finding "
        f"slots {dx['n_no_finding_slots']:,} = {dx['n_slots']:,}. Disease bars are GT-positive "
        "tooth-slot support (n) under one-vs-rest scoring: "
        f"Impacted {disease_n.get('Impacted', 0)}, Caries {disease_n.get('Caries', 0)}, "
        f"Deep Caries {disease_n.get('Deep Caries', 0)}, Periapical Lesion {disease_n.get('Periapical Lesion', 0)} "
        f"(sum {dx.get('diagnosis_positive_slots_sum') or sum(disease_n.values())}). "
        "These four n do not sum to unique abnormal teeth because a multi-label tooth is counted in "
        "every disease it carries. 2290/610 were unofficial sums of the DENTEX paper's train totals "
        "plus val; canonical unique (image, FDI) on-disk QED before held-out exclusion is Caries 2287 "
        "and Deep Caries 607.",
    )
    return paths


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def figure3(out_dir: Path, dentex_matrix: dict) -> list[Path]:
    tufts_vals = np.zeros((len(MODEL_ORDER), len(STRAT_ORDER)))
    abn_vals = np.zeros_like(tufts_vals)
    mac4_vals = np.zeros_like(tufts_vals)
    tufts_ci = np.full((len(MODEL_ORDER), len(STRAT_ORDER), 2), np.nan)
    abn_ci = np.full_like(tufts_ci, np.nan)
    mac4_ci = np.full_like(tufts_ci, np.nan)

    def _ci(entry: dict, key: str) -> tuple[float, float]:
        boot = ((entry.get("metrics") or {}).get("bootstrap") or {}).get(key) or {}
        return float(boot["low"]) if boot.get("low") is not None else float("nan"), (
            float(boot["high"]) if boot.get("high") is not None else float("nan")
        )

    for j, strat in enumerate(STRAT_ORDER):
        tufts = json.loads(latest_tufts_json(strat).read_text())
        for i, mk in enumerate(MODEL_ORDER):
            tufts_vals[i, j] = tufts_pooled_macro_f1(tufts["models"][mk])["macro_f1"]
            tufts_ci[i, j] = _ci(tufts["models"][mk], "macro_f1_ci95")
            sc = dentex_scores(dentex_matrix[strat]["models"][mk])
            abn_vals[i, j] = sc["macro_f1_abn"]
            mac4_vals[i, j] = sc["macro4_f1"]
            abn_ci[i, j] = _ci(dentex_matrix[strat]["models"][mk], "abnormal_macro_f1_ci95")
            mac4_ci[i, j] = _ci(dentex_matrix[strat]["models"][mk], "pathology_macro4_f1_ci95")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), sharey=True)
    panels = [
        (tufts_vals, tufts_ci, "Tufts Macro F1\n(Missing / Present, 1000 cases)"),
        (abn_vals, abn_ci, "DENTEX abnormality Macro F1\n(Abnormal / No annotated finding)"),
        (mac4_vals, mac4_ci, "DENTEX pathology Macro-4 F1\n(four independent OvR classes)"),
    ]
    x = np.arange(len(STRAT_ORDER))
    width = 0.18
    for ax, (mat, ci, title) in zip(axes, panels):
        for i, mk in enumerate(MODEL_ORDER):
            xpos = x + (i - 1.5) * width
            yerr = None
            if np.isfinite(ci[i]).all():
                low = np.clip(mat[i] - ci[i, :, 0], 0, None)
                high = np.clip(ci[i, :, 1] - mat[i], 0, None)
                yerr = np.vstack([low, high])
            ax.bar(
                xpos, mat[i], width, label=MODEL_LABEL[mk], color=OKABE[i],
                yerr=yerr, capsize=2, error_kw={"linewidth": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([STRAT_LABEL[s] for s in STRAT_ORDER])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pooled F1")
    axes[2].legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.suptitle("Model × prompting strategy  (12 combinations; error bars = patient-level 95% CI)", y=1.03, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, out_dir, "Figure3_model_strategy_macroF1")
    _caption(
        out_dir,
        "Figure3_model_strategy_macroF1",
        "Grouped bars for four models and three prompting strategies. Point estimates are pooled-slot "
        "Macro F1 (Tufts: 1000 cases; DENTEX: QED minus 3 held-out text exemplars). Error bars are "
        "95% CIs from 5000 patient-level bootstrap replicates (resample patients, keep all of that "
        "patient's images and their 32 slots; on Tufts and DENTEX QED each image is one patient). "
        "recompute pooled Macro F1). Right panel is Macro-4 F1 over Impacted, Caries, Deep Caries, "
        "and Periapical Lesion.",
    )
    return paths


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

def _heatmap(ax, matrix: np.ndarray, row_labels, col_labels, title: str, cmap="YlGnBu") -> None:
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7.5)
    ax.set_title(title, fontsize=10)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if np.isnan(v):
                ax.text(c, r, "N/A", ha="center", va="center", fontsize=7)
            else:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.55 else "black")
    return im


def figure4(out_dir: Path, dentex_matrix: dict) -> list[Path]:
    rows = [f"{MODEL_LABEL[m]}\n{STRAT_LABEL[s]}" for s in STRAT_ORDER for m in MODEL_ORDER]
    disease = np.zeros((12, 4))
    binary = np.zeros((12, 4))  # tufts miss/pres + dentex abn/nofind
    k = 0
    for s in STRAT_ORDER:
        tufts = json.loads(latest_tufts_json(s).read_text())
        for m in MODEL_ORDER:
            ts = tufts_pooled_macro_f1(tufts["models"][m])
            ds = dentex_scores(dentex_matrix[s]["models"][m])
            disease[k] = [ds[d] for d in DISEASES]
            binary[k] = [ts["missing_f1"], ts["present_f1"], ds["abnormal_f1"], ds["no_finding_f1"]]
            k += 1

    fig = plt.figure(figsize=(11.6, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.15], wspace=0.32)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    im0 = _heatmap(ax0, disease, rows, [DISEASE_SHORT[d] for d in DISEASES],
                   "DENTEX pathology F1  (one-vs-rest)")
    im1 = _heatmap(ax1, binary, rows,
                   ["Tufts Missing", "Tufts Present", "DENTEX Abnormal", "DENTEX No finding"],
                   "Binary-task F1")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03, label="F1")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03, label="F1")
    fig.suptitle("Per-class pooled F1  (minority classes can fail despite high accuracy)", y=0.98, fontweight="bold")
    paths = _save(fig, out_dir, "Figure4_per_class_F1_heatmaps")
    _caption(
        out_dir,
        "Figure4_per_class_F1_heatmaps",
        "Left: DENTEX one-vs-rest F1 for Impacted, Caries, Deep Caries, and Periapical Lesion. "
        "Right: binary F1 for Tufts Missing/Present and DENTEX Abnormal/No annotated finding. "
        "Rows are the 12 model × prompting combinations. Support counts are reported in Figure 2, not in cells.",
    )
    return paths


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

def _draw_pano(path: Path, overlays: list[dict], title: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_s = font
    for ov in overlays:
        x, y, w, h = ov["bbox"]
        color = ov.get("color", (213, 94, 0, 180))
        draw.rectangle([x, y, x + w, y + h], outline=color[:3] + (255,), width=6)
        label = ov.get("label", "")
        if label:
            draw.rectangle([x, max(0, y - 36), x + min(w, 8 + 14 * len(label)), y], fill=color[:3] + (220,))
            draw.text((x + 4, max(0, y - 34)), label, fill="white", font=font_s)
    draw.rectangle([0, 0, im.width, 48], fill=(0, 0, 0, 150))
    draw.text((12, 10), title, fill="white", font=font)
    return im


def _pano_path(case_id: str) -> Path | None:
    candidates = [
        VAL_XRAYS / f"{case_id}.png",
        TRAIN_XRAYS / f"{case_id}.png",
        TUFTS_XRAYS / f"{case_id}.JPG",
        TUFTS_XRAYS / f"{case_id}.jpg",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _boxed_from_json(json_path: Path) -> dict[tuple[str, int], list]:
    payload_raw = json.loads(json_path.read_text())
    c1 = {c["id"]: str(c["name"]) for c in payload_raw["categories_1"]}
    c2 = {c["id"]: str(c["name"]) for c in payload_raw["categories_2"]}
    c3 = {c["id"]: str(c["name"]) for c in payload_raw["categories_3"]}
    id_to_name = {im["id"]: im["file_name"] for im in payload_raw["images"]}
    boxed: dict[tuple[str, int], list] = defaultdict(list)
    for a in payload_raw["annotations"]:
        fdi = int(f"{c1[a['category_id_1']]}{c2[a['category_id_2']]}")
        stem = Path(id_to_name[a["image_id"]]).stem
        boxed[(stem, fdi)].append((a["bbox"], c3[a["category_id_3"]]))
    return boxed


def _find_tufts_example(payload: dict) -> tuple[str, dict] | None:
    """First sorted case with a missing-tooth FN. Model order is locked in the protocol."""
    for mk in ("huatuogpt_vision", "llava", "llava_med", "dentvlm"):
        entry = payload["models"].get(mk) or {}
        rows = [r for r in entry.get("results") or [] if "error" not in r]
        rows.sort(key=lambda r: (not str(r["case_id"]).isdigit(), int(r["case_id"]) if str(r["case_id"]).isdigit() else r["case_id"]))
        for r in rows:
            if _pano_path(str(r["case_id"])) is None:
                continue
            yt, yp = r.get("y_true_32") or [], r.get("y_pred_32") or []
            if any(t == 1 and p == 0 for t, p in zip(yt, yp)):
                missed = [i + 1 for i, (t, p) in enumerate(zip(yt, yp)) if t == 1 and p == 0]
                return mk, {"case_id": r["case_id"], "missed": missed, "gt": r["ground_truth_teeth"],
                            "pred": r["predicted_teeth"]}
    return None


def _find_dentex_examples(payload: dict) -> dict[str, Any]:
    """First sorted matches per locked Figure 5 category. No cherry-picking."""
    entry = None
    mk = None
    for cand in ("huatuogpt_vision", "llava", "llava_med", "dentvlm"):
        e = payload["models"].get(cand)
        if e and (e.get("metrics") or {}).get("abnormal_tooth"):
            entry, mk = e, cand
            break
    if entry is None:
        return {}
    boxed = _boxed_from_json(TRAIN_JSON)
    boxed.update(_boxed_from_json(VAL_JSON))

    picked = {"model": mk, "display": entry.get("display_name", mk)}
    rows = [r for r in entry.get("results") or [] if "error" not in r]
    rows.sort(key=lambda r: r["case_id"])
    for r in rows:
        cid = r["case_id"]
        if _pano_path(cid) is None:
            continue
        gt = {int(k): eval_metrics.labels_on_tooth(v) for k, v in (r.get("ground_truth_findings") or {}).items()}
        pred = {int(k): eval_metrics.labels_on_tooth(v) for k, v in (r.get("predicted_findings") or {}).items()}
        if "correct" not in picked and gt == pred and gt:
            picked["correct"] = {"case_id": cid, "gt": gt, "pred": pred}
        missed_abn = [t for t, labs in gt.items() if labs and t not in pred]
        if "missed_abn" not in picked and missed_abn:
            picked["missed_abn"] = {"case_id": cid, "tooth": missed_abn[0], "gt": gt, "pred": pred}
        if "multi_ok" not in picked:
            for t, labs in gt.items():
                if len(labs) >= 2 and set(pred.get(t) or []) == set(labs):
                    picked["multi_ok"] = {"case_id": cid, "tooth": t, "gt": gt, "pred": pred}
                    break
        if "multi_partial" not in picked:
            for t, labs in gt.items():
                inter = set(pred.get(t) or []) & set(labs)
                if len(labs) >= 2 and inter and set(pred.get(t) or []) != set(labs):
                    picked["multi_partial"] = {"case_id": cid, "tooth": t, "gt": gt, "pred": pred}
                    break
    picked["boxed"] = boxed
    picked["protocol"] = str(OUT_DEFAULT / "Figure5_selection_protocol.md")
    return picked


def figure5(out_dir: Path, dentex_matrix: dict) -> list[Path]:
    tufts_payload = json.loads(latest_tufts_json("zero_shot").read_text())
    dentex_payload = dentex_matrix["zero_shot"]
    panels: list[tuple[Image.Image, str]] = []

    ex = _find_dentex_examples(dentex_payload)
    boxed = ex.get("boxed") or {}

    def overlays_for(case_id: str, teeth: dict[int, list[str]], highlight: int | None = None) -> list[dict]:
        ovs = []
        for t, labs in teeth.items():
            hits = boxed.get((case_id, t)) or []
            if not hits:
                continue
            bbox = hits[0][0]
            color = (0, 158, 115, 200) if highlight is None or t == highlight else (0, 114, 178, 160)
            if highlight is not None and t == highlight:
                color = (213, 94, 0, 220)
            ovs.append({"bbox": bbox, "label": f"{t}: {', '.join(labs)}", "color": color})
        return ovs

    if ex.get("correct"):
        c = ex["correct"]
        path = _pano_path(c["case_id"])
        if path is not None:
            im = _draw_pano(
                path,
                overlays_for(c["case_id"], c["gt"]),
                f"A  Completely correct   {ex['display']}  {c['case_id']}",
            )
            panels.append((im, "correct"))

    missed = _find_tufts_example(tufts_payload)
    if missed:
        mk, info = missed
        img_path = _pano_path(str(info["case_id"]))
        if img_path is not None:
            im = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            except OSError:
                font = ImageFont.load_default()
            draw.rectangle([0, 0, im.width, 48], fill=(0, 0, 0))
            draw.text(
                (12, 10),
                f"B  Missed missing tooth   {MODEL_LABEL.get(mk, mk)}  case {info['case_id']}  "
                f"missed {info['missed'][:8]}",
                fill="white",
                font=font,
            )
            panels.append((im, "missed_missing"))

    if ex.get("missed_abn"):
        c = ex["missed_abn"]
        path = _pano_path(c["case_id"])
        if path is not None:
            im = _draw_pano(
                path,
                overlays_for(c["case_id"], c["gt"], highlight=c["tooth"]),
                f"C  Abnormal predicted as no finding   {ex['display']}  "
                f"{c['case_id']} FDI {c['tooth']} GT={c['gt'].get(c['tooth'])} pred={c['pred'].get(c['tooth'], [])}",
            )
            panels.append((im, "missed_abn"))

    if ex.get("multi_ok"):
        c = ex["multi_ok"]
        path = _pano_path(c["case_id"])
        if path is not None:
            im = _draw_pano(
                path,
                overlays_for(c["case_id"], {c["tooth"]: c["gt"][c["tooth"]]}, highlight=c["tooth"]),
                f"D  Correct multi-label   {ex['display']}  {c['case_id']} "
                f"FDI {c['tooth']} {c['gt'][c['tooth']]}",
            )
            panels.append((im, "multi_ok"))

    if ex.get("multi_partial"):
        c = ex["multi_partial"]
        path = _pano_path(c["case_id"])
        if path is not None:
            im = _draw_pano(
                path,
                overlays_for(c["case_id"], {c["tooth"]: c["gt"][c["tooth"]]}, highlight=c["tooth"]),
                f"E  Partial multi-label failure   {ex['display']}  {c['case_id']} "
                f"FDI {c['tooth']}  GT={c['gt'][c['tooth']]}  pred={c['pred'].get(c['tooth'], [])}",
            )
            panels.append((im, "multi_partial"))

    log = {
        "protocol": str(OUT_DEFAULT / "Figure5_selection_protocol.md"),
        "source_model": ex.get("model"),
        "source_strategy": "zero_shot",
        "picks": {
            "A_correct": None if not ex.get("correct") else {"case_id": ex["correct"]["case_id"]},
            "B_missing_tooth_fn": None if not missed else {"model": missed[0], "case_id": missed[1]["case_id"]},
            "C_abnormality_fn": None if not ex.get("missed_abn") else {
                "case_id": ex["missed_abn"]["case_id"], "tooth": ex["missed_abn"]["tooth"]
            },
            "D_correct_multilabel": None if not ex.get("multi_ok") else {
                "case_id": ex["multi_ok"]["case_id"], "tooth": ex["multi_ok"]["tooth"]
            },
            "E_partial_multilabel": None if not ex.get("multi_partial") else {
                "case_id": ex["multi_partial"]["case_id"], "tooth": ex["multi_partial"]["tooth"]
            },
        },
        "omitted": [
            name for name, val in {
                "A_correct": ex.get("correct"),
                "B_missing_tooth_fn": missed,
                "C_abnormality_fn": ex.get("missed_abn"),
                "D_correct_multilabel": ex.get("multi_ok"),
                "E_partial_multilabel": ex.get("multi_partial"),
            }.items() if not val
        ],
        "note": "First sorted case_id match per locked category. No substitutions.",
    }
    (out_dir / "Figure5_selection_log.json").write_text(json.dumps(log, indent=2))

    if not panels:
        return []

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11.4, 2.35 * n))
    if n == 1:
        axes = [axes]
    for ax, (im, _) in zip(axes, panels):
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("Representative successes and failures", y=1.01, fontweight="bold")
    fig.tight_layout()
    paths = _save(fig, out_dir, "Figure5_qualitative_examples")
    _caption(
        out_dir,
        "Figure5_qualitative_examples",
        "Qualitative cases selected by a protocol locked before inspecting outcomes "
        "(Main Job Outputs/figures/Figure5_selection_protocol.md). Each panel is the first matching "
        "case in sorted case_id order from the declared model/strategy. A: completely correct "
        "DENTEX film. B: missed missing tooth on Tufts (Universal numbering). C: DENTEX abnormal "
        "tooth predicted as no annotated finding. D: correctly recovered multi-label tooth. "
        "E: partial multi-label failure (prediction shares ≥1 GT label and misses ≥1). "
        "A missing category is omitted rather than replaced. Boxes are DENTEX QED annotations.",
    )
    return paths


# ---------------------------------------------------------------------------
# Figure 6  DENTEX 2×2 confusion (abnormal vs no finding)
# ---------------------------------------------------------------------------

def figure6(out_dir: Path, dentex_matrix: dict) -> list[Path]:
    """4 models × 3 strategies of 2×2 CMs. Colour is row-normalised so TN mass does not hide FN."""
    cms = []
    for mk in MODEL_ORDER:
        row = []
        for strat in STRAT_ORDER:
            ab = dentex_matrix[strat]["models"][mk]["metrics"]["abnormal_tooth"]
            cm = np.array(ab["confusion_matrix"], dtype=float)
            row.append(cm)
        cms.append(row)

    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(STRAT_ORDER),
        figsize=(11.6, 8.6),
    )
    xticklabels = ["Pred.\nno finding", "Pred.\nabnormal"]
    yticklabels = ["True\nno finding", "True\nabnormal"]

    last_im = None
    for i, mk in enumerate(MODEL_ORDER):
        for j, strat in enumerate(STRAT_ORDER):
            ax = axes[i, j]
            cm = cms[i][j]
            row_sum = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
            rates = cm / row_sum
            last_im = ax.imshow(rates, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
            for r in range(2):
                for c in range(2):
                    n = int(cm[r, c])
                    p = rates[r, c]
                    ax.text(
                        c, r, f"{n:,}\n{p:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if p > 0.55 else "black",
                    )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(xticklabels, fontsize=7.5)
            ax.set_yticklabels(yticklabels, fontsize=7.5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_color("#555555")
            if i == 0:
                ax.set_title(STRAT_LABEL[strat], fontsize=11, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[mk], fontsize=10, fontweight="bold", labelpad=8)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.07, wspace=0.22, hspace=0.38)
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("Share of true class", fontsize=9)
    fig.suptitle(
        "DENTEX abnormality confusion  (pooled 24,064 FDI slots; 3,627 abnormal / 20,437 no finding)",
        y=0.97, fontweight="bold",
    )
    paths = _save(fig, out_dir, "Figure6_dentex_abnormal_confusion")
    _caption(
        out_dir,
        "Figure6_dentex_abnormal_confusion",
        "Two-by-two confusion matrices for Abnormal vs No annotated finding on DENTEX QED "
        "(752 cases × 32 FDI slots = 24,064; 3,627 abnormal, 20,437 no finding). Rows are the "
        "true class, columns the predicted class: [[TN, FP], [FN, TP]]. Cell colour is "
        "row-normalised (share of that true class) so the majority true-negative class does not "
        "dominate the colormap; numbers are raw slot counts and the same row percentage. "
        "High specificity with low sensitivity appears as a dark top-left cell and a pale "
        "bottom-right cell. PPV is undefined when a model never predicts abnormal (TP+FP = 0).",
    )
    return paths


def write_status(out_dir: Path, made: dict[str, list[Path]], pending: list[str]) -> Path:
    lines = ["# Main-paper figures", ""]
    for name, paths in made.items():
        lines.append(f"- **{name}**: " + ", ".join(p.name for p in paths))
    if pending:
        lines.append("")
        lines.append("Pending until DENTEX qed matrix is complete:")
        for p in pending:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("ROC/PR curves are not in the main set.")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--dentex-split", default="qed")
    args = ap.parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    made: dict[str, list[Path]] = {}
    print("Figure 1 pipeline…")
    made["Figure 1"] = figure1(args.out_dir)
    print("Figure 2 dataset support…")
    made["Figure 2"] = figure2(args.out_dir)

    pending = []
    matrix = load_dentex_matrix(args.dentex_split)
    if matrix is None:
        pending = [
            "Figure 3 model × strategy Macro F1 (needs DENTEX qed JSON for all 3 strategies)",
            "Figure 4 per-class F1 heatmaps",
            "Figure 5 qualitative examples (first sorted match per locked category)",
            "Figure 6 DENTEX abnormal vs no-finding confusion matrices",
        ]
        print("DENTEX qed matrix not complete — skipping Figures 3–6")
    else:
        print("Figure 3 model × strategy…")
        made["Figure 3"] = figure3(args.out_dir, matrix)
        print("Figure 4 per-class heatmaps…")
        made["Figure 4"] = figure4(args.out_dir, matrix)
        print("Figure 5 qualitative…")
        made["Figure 5"] = figure5(args.out_dir, matrix)
        print("Figure 6 DENTEX confusion…")
        made["Figure 6"] = figure6(args.out_dir, matrix)

    write_status(args.out_dir, made, pending)
    print(f"wrote {args.out_dir}")
    for name, paths in made.items():
        print(f"  {name}: {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
