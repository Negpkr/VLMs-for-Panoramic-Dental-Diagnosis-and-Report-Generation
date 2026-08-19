#!/usr/bin/env python3
"""Add balanced accuracy to existing comparison JSONs and rewrite summary CSVs.

balanced_accuracy = (recall + specificity) / 2
Does not rerun models. Uses already-saved patient-level recall and specificity.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

TUFTS = [
    Path("/home/s222393187/Dental/Results/model_comparison/comparison_zero_shot_100_20260807_131943.json"),
    Path("/home/s222393187/Dental/Results/model_comparison/comparison_few_shot_100_20260807_141518.json"),
    Path("/home/s222393187/Dental/Results/model_comparison/comparison_cot_100_20260807_145219.json"),
]
DENTEX = [
    Path("/home/s222393187/Dental/Results/dentex_comparison/dentex_zero_shot_val50_20260807_134046.json"),
    Path("/home/s222393187/Dental/Results/dentex_comparison/dentex_few_shot_val50_20260819_220141.json"),
]
COMBINED = Path("/home/s222393187/Dental/Results/metrics_with_balanced_accuracy.csv")


def bal(recall, spec) -> float | None:
    if recall is None or spec is None:
        return None
    return 0.5 * (float(recall) + float(spec))


def patch_block(block: dict) -> None:
    if not block:
        return
    if "recall" in block and "specificity" in block:
        block["balanced_accuracy"] = bal(block["recall"], block["specificity"])


def patch_payload(payload: dict) -> None:
    for entry in payload.get("models", {}).values():
        metrics = entry.get("metrics") or {}
        for key in ("per_tooth", "image_level", "abnormal_tooth"):
            patch_block(metrics.get(key) or {})
        for block in (metrics.get("per_disease") or {}).values():
            patch_block(block)
        for sample in metrics.get("per_sample") or []:
            patch_block(sample)
        for result in entry.get("results") or []:
            patch_block(result.get("sample_metrics") or {})


def tufts_row(payload, entry) -> dict:
    m = entry.get("metrics") or {}
    t = m.get("per_tooth") or {}
    return {
        "Dataset": "Tufts",
        "Task": "missing_teeth",
        "Model": entry.get("display_name"),
        "Strategy": payload.get("strategy") or payload.get("prompt_name"),
        "Cases": m.get("n_cases"),
        "F1": t.get("f1_score"),
        "Precision": t.get("precision"),
        "Recall": t.get("recall"),
        "Specificity": t.get("specificity"),
        "Balanced_Accuracy": t.get("balanced_accuracy"),
        "Accuracy": t.get("accuracy"),
        "Mean_pred_per_case": m.get("mean_predicted_count"),
        "Mean_GT_per_case": m.get("mean_ground_truth_count"),
    }


def dentex_row(payload, entry) -> dict:
    m = entry.get("metrics") or {}
    t = m.get("abnormal_tooth") or {}
    return {
        "Dataset": "DENTEX",
        "Task": "abnormal_tooth",
        "Model": entry.get("display_name"),
        "Strategy": payload.get("strategy"),
        "Cases": m.get("n_cases"),
        "F1": t.get("f1_score"),
        "Precision": t.get("precision"),
        "Recall": t.get("recall"),
        "Specificity": t.get("specificity"),
        "Balanced_Accuracy": t.get("balanced_accuracy"),
        "Accuracy": t.get("accuracy"),
        "Mean_pred_per_case": m.get("mean_predicted_count"),
        "Mean_GT_per_case": m.get("mean_ground_truth_count"),
    }


def write_summary_tufts(payload: dict, json_path: Path) -> None:
    out = json_path.with_name("summary_" + json_path.stem.replace("comparison_", "") + ".csv")
    # keep original naming: summary_TIMESTAMP.csv next to comparison_STRATEGY_...
    stamp = json_path.stem.replace("comparison_", "")
    out = json_path.parent / f"summary_{stamp}.csv"
    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        t = m.get("per_tooth")
        if not t:
            continue
        rows.append(
            {
                "Model": entry["display_name"],
                "Cases": m.get("n_cases"),
                "Strategy": payload.get("strategy") or payload.get("prompt_name"),
                "Aggregation": m.get("aggregation"),
                "F1": t.get("f1_score"),
                "Precision": t.get("precision"),
                "Recall": t.get("recall"),
                "Specificity": t.get("specificity"),
                "Balanced_Accuracy": t.get("balanced_accuracy"),
                "Accuracy": t.get("accuracy"),
                "Mean_pred_per_case": m.get("mean_predicted_count"),
                "Mean_GT_per_case": m.get("mean_ground_truth_count"),
                "Sec_per_case": m.get("mean_seconds_per_case"),
                "Status": "ok",
            }
        )
    if not rows:
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out.name}")


def write_summary_dentex(payload: dict, json_path: Path) -> None:
    out = json_path.with_name(json_path.stem + "_summary.csv")
    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        t = m.get("abnormal_tooth")
        if not t:
            continue
        row = {
            "Model": entry["display_name"],
            "Cases": m.get("n_cases"),
            "Strategy": payload.get("strategy"),
            "Abnormal F1": t.get("f1_score"),
            "Abnormal Precision": t.get("precision"),
            "Abnormal Recall": t.get("recall"),
            "Abnormal Specificity": t.get("specificity"),
            "Abnormal Balanced Acc": t.get("balanced_accuracy"),
            "Abnormal Accuracy": t.get("accuracy"),
            "Mean pred/case": m.get("mean_predicted_count"),
            "Mean GT/case": m.get("mean_ground_truth_count"),
            "Sec/case": m.get("mean_seconds_per_case"),
            "Status": "ok",
        }
        for disease, block in (m.get("per_disease") or {}).items():
            row[f"{disease} F1"] = block.get("f1_score")
            row[f"{disease} BalAcc"] = block.get("balanced_accuracy")
        rows.append(row)
    if not rows:
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out.name}")


def main() -> None:
    combined = []
    for path in TUFTS:
        payload = json.loads(path.read_text())
        patch_payload(payload)
        path.write_text(json.dumps(payload, indent=2))
        print(f"patched {path.name}")
        write_summary_tufts(payload, path)
        for entry in payload["models"].values():
            if (entry.get("metrics") or {}).get("per_tooth"):
                combined.append(tufts_row(payload, entry))
    for path in DENTEX:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        patch_payload(payload)
        path.write_text(json.dumps(payload, indent=2))
        print(f"patched {path.name}")
        write_summary_dentex(payload, path)
        for entry in payload["models"].values():
            if (entry.get("metrics") or {}).get("abnormal_tooth"):
                combined.append(dentex_row(payload, entry))

    with COMBINED.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        w.writeheader()
        w.writerows(combined)
    print(f"wrote {COMBINED}")


if __name__ == "__main__":
    main()
