#!/usr/bin/env python3
"""Run the finding-level hallucination analysis over the saved VLM runs.

This is additive: it only READS the existing run JSONs under Results/ and
WRITES new CSVs under results/hallucination/. It never modifies predictions,
ground truth, or any existing metric file.

Usage (from the repo root):
    python -m hallucination.run_analysis
    python -m hallucination.run_analysis --results-dir Results --out results/hallucination

Outputs (see OUTPUT STRUCTURE in the brief):
    results/hallucination/model_hallucination_summary.csv
    results/hallucination/condition_hallucination_summary.csv
    results/hallucination/image_level_hallucination_results.csv
    results/hallucination/location_analysis.csv          (DENTEX only)
    results/hallucination/bootstrap_confidence_intervals.csv
    results/hallucination/model_pairwise_comparisons.csv
    results/hallucination/verification_sample.txt        (10 audited cases/config)
    results/hallucination/README_run_context.json        (provenance / matching rule)

Because the project evaluates two datasets (DENTEX, Tufts) x three prompting
strategies, every CSV carries Dataset + Strategy columns; the model summary is
one row per (Dataset, Strategy, Model). Figures are produced separately by
hallucination/figures.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd

from hallucination.metrics import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    ImageResult,
    aggregate_metrics,
    bootstrap_metrics,
    condition_metrics,
    findings_from_dentex,
    findings_from_tufts,
    location_breakdown,
    paired_comparison,
    serialise_findings,
)

# Preserve the repository's canonical model ordering (from paper_figures.py).
MODEL_ORDER = ["llava", "llava_med", "huatuogpt_vision", "dentvlm"]
STRATEGY_ORDER = ["zero_shot", "few_shot", "cot"]
VERIFY_SEED = 42  # fixed seed for the "10 random cases" audit


def find_run_files(results_dir: Path) -> list[Path]:
    """Locate the per-run comparison JSONs (DENTEX + Tufts, all strategies)."""
    patterns = [
        results_dir / "dentex_comparison" / "dentex_*_qed_*.json",
        results_dir / "model_comparison" / "comparison_*_full_*.json",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(Path(p) for p in glob.glob(str(pat)))
    # Drop non-run sidecars (summaries/supplementary are CSV, but be safe).
    return sorted(f for f in files if f.suffix == ".json")


def load_run(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def run_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise dataset/strategy/label metadata across the two JSON schemas."""
    rc = payload.get("run_config") or {}
    dataset = payload.get("dataset") or rc.get("dataset")
    strategy = payload.get("strategy") or rc.get("strategy")
    if dataset == "DENTEX":
        conditions = list(payload.get("diseases") or [])
        fdi = list(payload.get("fdi_teeth") or [])
        n_teeth = len(fdi)
        kind = "dentex"
    else:  # Tufts missing-tooth task
        conditions = ["Missing"]
        n_teeth = 32  # Tufts is a fixed 32-slot universal-numbering grid
        kind = "tufts"
    return {
        "dataset": dataset,
        "strategy": strategy,
        "kind": kind,
        "conditions": conditions,
        "n_teeth": n_teeth,
        "n_slots_per_image": n_teeth * len(conditions),
        "timestamp": payload.get("timestamp"),
    }


def build_image_results(
    model_results: list[dict[str, Any]],
    ctx: dict[str, Any],
) -> list[ImageResult]:
    """Convert saved per-case records into finding-level ImageResult objects."""
    kind = ctx["kind"]
    n_slots = ctx["n_slots_per_image"]
    out: list[ImageResult] = []
    for rec in model_results:
        image_id = str(rec.get("case_id") or rec.get("file_name") or "unknown")
        if kind == "dentex":
            gt = findings_from_dentex(rec.get("ground_truth_findings"))
            pred = findings_from_dentex(rec.get("predicted_findings"))
        else:
            gt = findings_from_tufts(rec.get("ground_truth_teeth"))
            pred = findings_from_tufts(rec.get("predicted_teeth"))
        out.append(ImageResult(image_id=image_id, gt=gt, pred=pred, n_slots=n_slots))
    return out


def ordered_models(models: dict[str, Any]) -> list[str]:
    """Canonical order first, then any extras, preserving reproducibility."""
    known = [m for m in MODEL_ORDER if m in models]
    extras = [m for m in models if m not in MODEL_ORDER]
    return known + sorted(extras)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_files = find_run_files(results_dir)
    if not run_files:
        raise SystemExit(f"No run JSONs found under {results_dir}")

    summary_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    location_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    verification_blocks: list[str] = []
    provenance: list[dict[str, Any]] = []

    for path in run_files:
        payload = load_run(path)
        ctx = run_context(payload)
        dataset, strategy, kind = ctx["dataset"], ctx["strategy"], ctx["kind"]
        conditions, n_teeth = ctx["conditions"], ctx["n_teeth"]
        models = payload["models"]
        provenance.append({"file": str(path), **ctx, "models": list(models)})
        print(f"\n=== {dataset} / {strategy}  ({path.name}) ===")

        # Per-model finding-level results, kept for pairwise comparison.
        per_model_images: dict[str, list[ImageResult]] = {}

        for mk in ordered_models(models):
            entry = models[mk]
            display = entry.get("display_name", mk)
            images = build_image_results(entry["results"], ctx)
            per_model_images[mk] = images

            agg = aggregate_metrics(images)
            summary_rows.append({
                "Dataset": dataset, "Strategy": strategy,
                "Model": mk, "Model_display": display, **agg,
            })
            print(f"  {display:22s} TP={agg['TP']:5d} FP={agg['FP']:5d} "
                  f"FN={agg['FN']:5d}  CHR={agg['CHR']:.3f}  "
                  f"P={agg['Precision']:.3f} R={agg['Recall']:.3f} F1={agg['F1']:.3f}")

            # Per-condition breakdown.
            cmet = condition_metrics(images, conditions, n_teeth)
            for cond, row in cmet.items():
                condition_rows.append({
                    "Dataset": dataset, "Strategy": strategy,
                    "Model": mk, "Model_display": display,
                    "Condition": cond, **row,
                })

            # Per-image rows.
            for im in images:
                image_rows.append({
                    "Dataset": dataset, "Strategy": strategy,
                    "image_id": im.image_id, "model": mk, "model_display": display,
                    "ground_truth_findings": serialise_findings(im.gt),
                    "predicted_findings": serialise_findings(im.pred),
                    "correctly_predicted_findings": serialise_findings(im.tp_set),
                    "hallucinated_findings": serialise_findings(im.fp_set),
                    "missed_findings": serialise_findings(im.fn_set),
                    "TP": im.tp, "FP": im.fp, "FN": im.fn,
                    "number_of_predictions": im.n_pred,
                    "number_of_hallucinations": im.fp,
                    "hallucination_rate_for_image": im.image_chr,
                })

            # Location-aware breakdown: only where GT is tooth-specific AND
            # multiple conditions exist (DENTEX). For Tufts the tooth IS the
            # finding, so a wrong tooth is simply an ordinary FP.
            if kind == "dentex":
                lb = location_breakdown(images)
                location_rows.append({
                    "Dataset": dataset, "Strategy": strategy,
                    "Model": mk, "Model_display": display, **lb.as_row(),
                })

            # Image-level bootstrap CIs (seed fixed for reproducibility).
            boot = bootstrap_metrics(images, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            brow: dict[str, Any] = {
                "Dataset": dataset, "Strategy": strategy,
                "Model": mk, "Model_display": display,
                "n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "image",
            }
            for metric, ci in boot.items():
                brow[f"{metric}_point"] = ci["point"]
                brow[f"{metric}_CI95_low"] = ci["low"]
                brow[f"{metric}_CI95_high"] = ci["high"]
            bootstrap_rows.append(brow)

        # Pairwise paired-bootstrap comparisons (models share the same images).
        aligned, note = _aligned_models(per_model_images)
        model_keys = ordered_models(models)
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                a, b = model_keys[i], model_keys[j]
                if aligned:
                    cmp = paired_comparison(
                        per_model_images[a], per_model_images[b],
                        n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                    )
                else:
                    cmp = {}
                pairwise_rows.append({
                    "Dataset": dataset, "Strategy": strategy,
                    "Model_A": a, "Model_B": b,
                    "paired": aligned, "pairing_note": note, **cmp,
                })

        # Manual verification: 10 randomly-selected audited cases for this run,
        # deliberately seeded and salted with hard cases (see _verification).
        verification_blocks.append(
            _verification(dataset, strategy, ctx, models, per_model_images)
        )

    # --- write CSVs -------------------------------------------------------
    _write_csv(summary_rows, out_dir / "model_hallucination_summary.csv")
    _write_csv(condition_rows, out_dir / "condition_hallucination_summary.csv")
    _write_csv(image_rows, out_dir / "image_level_hallucination_results.csv")
    _write_csv(location_rows, out_dir / "location_analysis.csv")
    _write_csv(bootstrap_rows, out_dir / "bootstrap_confidence_intervals.csv")
    _write_csv(pairwise_rows, out_dir / "model_pairwise_comparisons.csv")

    (out_dir / "verification_sample.txt").write_text("\n\n".join(verification_blocks))
    (out_dir / "README_run_context.json").write_text(json.dumps({
        "matching_rule": (
            "A predicted finding (FDI tooth, condition) is a TP iff an identical "
            "(tooth, condition) exists in the ground truth; otherwise it is a FP "
            "(hallucination candidate). A GT finding with no matching prediction "
            "is a FN. TN = tooth x condition slots negative in both. Labels are "
            "used exactly as normalised by the project's own parser."
        ),
        "chr_definition": "CHR = FP / (TP + FP) = 1 - Precision (binary matching); NaN if TP+FP==0.",
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "image"},
        "iou": "Not computed: no bounding boxes or segmentation masks exist in the saved runs.",
        "runs": provenance,
    }, indent=2))

    print(f"\nWrote CSVs + audit to {out_dir}")


def _aligned_models(per_model_images: dict[str, list[ImageResult]]) -> tuple[bool, str]:
    """Check every model was evaluated on exactly the same image IDs, in order."""
    keys = list(per_model_images)
    if len(keys) < 2:
        return False, "fewer than two models"
    ref = [im.image_id for im in per_model_images[keys[0]]]
    for k in keys[1:]:
        ids = [im.image_id for im in per_model_images[k]]
        if ids != ref:
            return False, f"image IDs differ between {keys[0]} and {k}"
    return True, "identical image set/order across models"


def _verification(
    dataset: str, strategy: str, ctx: dict[str, Any],
    models: dict[str, Any], per_model_images: dict[str, list[ImageResult]],
) -> str:
    """Human-auditable dump of 10 cases: random draw + guaranteed hard cases.

    Hard cases (per the brief) are sought on the first model: empty GT but a
    prediction, empty prediction, over-prediction, and (DENTEX) a wrong-tooth
    case. The random remainder uses VERIFY_SEED for reproducibility.
    """
    lines = [f"########## VERIFICATION: {dataset} / {strategy} ##########"]
    keys = list(per_model_images)
    ref = per_model_images[keys[0]]
    idx_by_id = {im.image_id: k for k, im in enumerate(ref)}

    # Locate hard cases on the reference model.
    hard: dict[str, int] = {}
    for k, im in enumerate(ref):
        if "empty_gt_pred" not in hard and not im.gt and im.pred:
            hard["empty_gt_pred"] = k
        if "empty_pred" not in hard and im.gt and not im.pred:
            hard["empty_pred"] = k
        if "over_predict" not in hard and im.n_pred > len(im.gt) and im.gt:
            hard["over_predict"] = k
        if ctx["kind"] == "dentex" and "wrong_tooth" not in hard:
            gt_conditions = {c for _, c in im.gt}
            if any(c in gt_conditions and (t, c) not in im.gt for (t, c) in im.pred):
                hard["wrong_tooth"] = k

    rng = random.Random(VERIFY_SEED)
    chosen = list(dict.fromkeys(hard.values()))  # unique, keep order
    pool = [k for k in range(len(ref)) if k not in chosen]
    rng.shuffle(pool)
    chosen += pool[: max(0, 10 - len(chosen))]
    chosen = chosen[:10]

    labels = {v: name for name, v in hard.items()}
    for k in chosen:
        # Audit against the reference model (first in canonical order).
        im = ref[k]
        tag = f"  [hard: {labels[k]}]" if k in labels else ""
        lines.append(f"\n--- image_id={im.image_id}  (model={keys[0]}){tag}")
        lines.append(f"  Ground truth       : {serialise_findings(im.gt) or '(none)'}")
        lines.append(f"  Model prediction   : {serialise_findings(im.pred) or '(none)'}")
        lines.append(f"  Matched TP         : {serialise_findings(im.tp_set) or '(none)'}")
        lines.append(f"  FP / hallucinated  : {serialise_findings(im.fp_set) or '(none)'}")
        lines.append(f"  FN / missed        : {serialise_findings(im.fn_set) or '(none)'}")
        lines.append(f"  TP={im.tp} FP={im.fp} FN={im.fn} "
                     f"CHR_img={im.image_chr:.3f}" if im.n_pred
                     else f"  TP={im.tp} FP={im.fp} FN={im.fn} CHR_img=NaN (no predictions)")
    return "\n".join(lines)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")  # keep an explicit empty artefact
        print(f"  (no rows) {path}")
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="Results", type=Path,
                    help="Directory holding the saved run JSONs (default: Results)")
    ap.add_argument("--out", default="results/hallucination", type=Path,
                    help="Output directory for the hallucination CSVs")
    args = ap.parse_args()
    analyse(args.results_dir, args.out)


if __name__ == "__main__":
    main()
