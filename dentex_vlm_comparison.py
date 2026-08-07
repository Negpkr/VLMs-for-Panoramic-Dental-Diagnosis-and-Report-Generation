#!/usr/bin/env python3
"""DENTEX disease + FDI tooth enumeration zero-shot comparison.

Second dataset next to Tufts missing-teeth evaluation. Uses the DENTEX
validation split (50 images) with ground truth from validation_triple.json.

Metrics follow the same patient-level protocol as Tufts:
  - for each sample, score all 32 permanent FDI teeth (11-18, 21-28, 31-38, 41-48)
    as independent binary decisions (abnormal vs not)
  - average Acc / Prec / Recall / Spec / F1 across patients

Usage:
    CUDA_VISIBLE_DEVICES=0 python dentex_vlm_comparison.py --split val --strategy zero_shot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_models  # noqa: E402

DENTEX_ROOT = Path("/home/s222393187/Dental/DENTEX")
VAL_JSON = DENTEX_ROOT / "validation_triple.json"
VAL_XRAYS = DENTEX_ROOT / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TEST_INPUT = DENTEX_ROOT / "disease" / "input"
TEST_LABEL = DENTEX_ROOT / "disease" / "label"
RESULTS_DIR = Path("/home/s222393187/Dental/Results/dentex_comparison")

# Permanent dentition in FDI (32 positions).
FDI_TEETH: list[int] = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)
DISEASES = ["Impacted", "Caries", "Deep Caries", "Periapical Lesion"]

ZERO_SHOT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use the FDI tooth numbering system ONLY (11-18, 21-28, 31-38, 41-48).\n"
    "Do NOT use Universal numbers 1-32.\n\n"
    "Task: find abnormal teeth and assign ONE diagnosis from:\n"
    "Impacted, Caries, Deep Caries, Periapical Lesion.\n"
    "Only list teeth that clearly show one of these abnormalities.\n"
    "If none, write None.\n\n"
    "Answer in exactly this format:\n"
    "Findings:\n"
    "<FDI>: <diagnosis>\n"
    "<FDI>: <diagnosis>\n"
    "Total abnormal: <count>\n"
    "Confidence: High/Medium/Low\n\n"
    "Example:\n"
    "Findings:\n"
    "18: Impacted\n"
    "16: Caries\n"
    "36: Deep Caries\n"
    "Total abnormal: 3\n"
    "Confidence: Medium"
)

FEW_SHOT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use FDI numbering ONLY (11-18, 21-28, 31-38, 41-48).\n"
    "Diagnoses allowed: Impacted, Caries, Deep Caries, Periapical Lesion.\n\n"
    "Example 1 (impacted wisdom teeth):\n"
    "Findings:\n"
    "18: Impacted\n"
    "28: Impacted\n"
    "38: Impacted\n"
    "48: Impacted\n"
    "Total abnormal: 4\n"
    "Confidence: High\n\n"
    "Example 2 (caries):\n"
    "Findings:\n"
    "16: Caries\n"
    "36: Deep Caries\n"
    "Total abnormal: 2\n"
    "Confidence: Medium\n\n"
    "Example 3 (none):\n"
    "Findings:\n"
    "None\n"
    "Total abnormal: 0\n"
    "Confidence: High\n\n"
    "Now answer for THIS radiograph in the same format."
)

COT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use FDI numbering ONLY (11-18, 21-28, 31-38, 41-48).\n"
    "Diagnoses: Impacted, Caries, Deep Caries, Periapical Lesion.\n\n"
    "Think step by step:\n"
    "1. Inspect each quadrant (1=UR, 2=UL, 3=LL, 4=LR).\n"
    "2. Note impacted third molars, caries, deep caries, periapical lesions.\n"
    "3. Assign FDI IDs and one diagnosis each.\n\n"
    "Write Reasoning, then the final answer:\n"
    "Reasoning: <brief>\n"
    "Findings:\n"
    "<FDI>: <diagnosis>\n"
    "Total abnormal: <count>\n"
    "Confidence: High/Medium/Low"
)

PROMPT_REGISTRY = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT,
    "cot": COT_PROMPT,
}

# Turkish label tokens in DENTEX test LabelMe files → English disease.
TEST_DISEASE_MAP = {
    "gömülü": "Impacted",
    "gomulu": "Impacted",
    "çürük": "Caries",
    "curuk": "Caries",
    "küretaj": "Caries",  # curettage often marks treated/caries-related; keep as Caries proxy
    "kuretaj": "Caries",
    "lezyon": "Periapical Lesion",
    "kanal": "Periapical Lesion",
    "çekim": None,  # extraction — not in DENTEX disease taxonomy for QED
    "cekim": None,
    "kırık": None,
    "kirik": None,
    "saglam": None,  # healthy
}


def _specificity(y_true: list[int], y_pred: list[int], zero_division: float = 0.0) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _fn, _tp = cm.ravel()
    denom = tn + fp
    if denom == 0:
        return float(zero_division)
    return float(tn / denom)


def sample_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity(y_true, y_pred, zero_division=0),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def to_binary_fdi(abnormal_fdis: set[int] | list[int]) -> list[int]:
    abnormal = set(abnormal_fdis)
    return [1 if t in abnormal else 0 for t in FDI_TEETH]


class DentexParser:
    """Parse FDI + disease findings from model text."""

    fdi_pattern = re.compile(r"\b([1-4][1-8])\b")
    line_pattern = re.compile(
        r"\b([1-4][1-8])\b\s*[:\-–]\s*(Impacted|Caries|Deep\s*Caries|Periapical\s*Lesion)",
        re.IGNORECASE,
    )
    # Also accept "18 Impacted" without colon.
    loose_pattern = re.compile(
        r"\b([1-4][1-8])\b[^\n]{0,20}?\b(Impacted|Deep\s*Caries|Caries|Periapical\s*Lesion)\b",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> dict[int, str]:
        findings: dict[int, str] = {}
        if re.search(r"Findings:\s*None\b", text, re.IGNORECASE):
            return {}
        for pattern in (self.line_pattern, self.loose_pattern):
            for m in pattern.finditer(text):
                fdi = int(m.group(1))
                disease = re.sub(r"\s+", " ", m.group(2)).strip().title()
                if disease == "Deep Caries" or disease.lower() == "deep caries":
                    disease = "Deep Caries"
                elif disease.lower() == "periapical lesion":
                    disease = "Periapical Lesion"
                elif disease.lower() == "impacted":
                    disease = "Impacted"
                elif disease.lower() == "caries":
                    disease = "Caries"
                if disease in DISEASES and fdi in FDI_TEETH:
                    findings.setdefault(fdi, disease)
        # If diseases missing but FDI numbers present under Findings block, mark as Caries default? No — skip.
        return findings

    def confidence(self, text: str) -> str:
        m = re.search(r"Confidence:\s*(High|Medium|Low)", text, re.IGNORECASE)
        return m.group(1).title() if m else "Medium"


def load_val_cases() -> list[dict[str, Any]]:
    payload = json.loads(VAL_JSON.read_text())
    c1 = {c["id"]: str(c["name"]) for c in payload["categories_1"]}
    c2 = {c["id"]: str(c["name"]) for c in payload["categories_2"]}
    c3 = {c["id"]: str(c["name"]) for c in payload["categories_3"]}

    by_img: dict[int, dict[int, str]] = defaultdict(dict)
    for a in payload["annotations"]:
        fdi = int(f"{c1[a['category_id_1']]}{c2[a['category_id_2']]}")
        disease = c3[a["category_id_3"]]
        # If same tooth has multiple anns, keep first (rare).
        by_img[a["image_id"]].setdefault(fdi, disease)

    cases = []
    for im in payload["images"]:
        path = VAL_XRAYS / im["file_name"]
        if not path.exists():
            continue
        gt = by_img.get(im["id"], {})
        cases.append(
            {
                "case_id": Path(im["file_name"]).stem,
                "file_name": im["file_name"],
                "image_path": path,
                "gt_findings": {int(k): v for k, v in gt.items()},
            }
        )
    cases.sort(key=lambda c: (not c["case_id"].split("_")[-1].isdigit(),
                              int(c["case_id"].split("_")[-1]) if c["case_id"].split("_")[-1].isdigit() else c["case_id"]))
    return cases


def load_test_cases() -> list[dict[str, Any]]:
    """Optional: parse LabelMe test labels into FDI→disease (best-effort)."""
    cases = []
    for img in sorted(TEST_INPUT.glob("test_*.png")):
        label_path = TEST_LABEL / f"{img.stem}.json"
        gt: dict[int, str] = {}
        if label_path.exists():
            data = json.loads(label_path.read_text())
            for shape in data.get("shapes", []):
                lab = str(shape.get("label", ""))
                # e.g. 1-çürük-15 or 0-saglam-11
                parts = lab.split("-")
                if len(parts) < 3:
                    continue
                token = parts[1].strip().lower()
                tooth = parts[-1].strip()
                if not tooth.isdigit():
                    continue
                fdi = int(tooth)
                disease = TEST_DISEASE_MAP.get(token)
                if disease and fdi in FDI_TEETH:
                    gt.setdefault(fdi, disease)
        cases.append(
            {
                "case_id": img.stem,
                "file_name": img.name,
                "image_path": img,
                "gt_findings": gt,
            }
        )
    return cases


def image_is_readable(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n_cases": 0, "aggregation": "patient_mean_of_32fdi"}

    per_sample = []
    disease_rows: dict[str, list[dict[str, float]]] = {d: [] for d in DISEASES}

    for r in ok:
        gt_set = set(r["ground_truth_findings"])
        pred_set = set(r["predicted_findings"])
        y_true = to_binary_fdi(gt_set)
        y_pred = to_binary_fdi(pred_set)
        m = sample_metrics(y_true, y_pred)
        m["case_id"] = r["case_id"]
        per_sample.append(m)
        r["sample_metrics"] = {k: m[k] for k in ("accuracy", "precision", "recall", "specificity", "f1_score", "tp", "fp", "fn", "tn")}
        r["y_true_32"] = y_true
        r["y_pred_32"] = y_pred

        # Per-disease patient metrics (binary: that disease vs not on each FDI tooth).
        gt_map = r["ground_truth_findings"]
        pred_map = r["predicted_findings"]
        for disease in DISEASES:
            yt = [1 if gt_map.get(t) == disease else 0 for t in FDI_TEETH]
            yp = [1 if pred_map.get(t) == disease else 0 for t in FDI_TEETH]
            disease_rows[disease].append(sample_metrics(yt, yp))

    def mean_block(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            "accuracy": float(np.mean([x["accuracy"] for x in rows])),
            "precision": float(np.mean([x["precision"] for x in rows])),
            "recall": float(np.mean([x["recall"] for x in rows])),
            "specificity": float(np.mean([x["specificity"] for x in rows])),
            "f1_score": float(np.mean([x["f1_score"] for x in rows])),
        }

    per_disease = {d: mean_block(rows) for d, rows in disease_rows.items() if rows}

    return {
        "n_cases": len(ok),
        "n_failed": len(results) - len(ok),
        "aggregation": "patient_mean_of_32fdi",
        "abnormal_tooth": mean_block(per_sample),  # primary: any abnormality
        "per_disease": per_disease,
        "per_sample": per_sample,
        "mean_predicted_count": sum(len(r["predicted_findings"]) for r in ok) / len(ok),
        "mean_ground_truth_count": sum(len(r["ground_truth_findings"]) for r in ok) / len(ok),
        "mean_seconds_per_case": sum(r["seconds"] for r in ok) / len(ok),
    }


def evaluate_model(
    model_key: str,
    cases: list[dict[str, Any]],
    parser: DentexParser,
    *,
    prompt: str,
    load_in_4bit: bool = False,
    max_new_tokens: int = 350,
    log=print,
) -> dict[str, Any]:
    spec = vlm_models.MODEL_REGISTRY[model_key]
    log(f"=== {spec.display_name} ({spec.repo_id}) ===")
    bundle = None
    results: list[dict[str, Any]] = []
    try:
        bundle = vlm_models.load_model(model_key, load_in_4bit=load_in_4bit)
        log(f"loaded in {bundle.load_seconds:.1f}s on {bundle.device}")
        for i, case in enumerate(cases, 1):
            path = case["image_path"]
            if not image_is_readable(path):
                results.append({"case_id": case["case_id"], "error": "unreadable image"})
                continue
            t0 = time.time()
            try:
                reply = vlm_models.generate(
                    bundle,
                    path,
                    prompt,
                    seed_key=f"dentex:{model_key}:{case['case_id']}",
                    max_new_tokens=max_new_tokens,
                )
                findings = parser.parse(reply)
                results.append(
                    {
                        "case_id": case["case_id"],
                        "file_name": case["file_name"],
                        "response": reply,
                        "predicted_findings": {str(k): v for k, v in findings.items()},
                        "ground_truth_findings": {str(k): v for k, v in case["gt_findings"].items()},
                        # int-key copies for metrics
                        "_pred": findings,
                        "_gt": case["gt_findings"],
                        "confidence": parser.confidence(reply),
                        "seconds": time.time() - t0,
                    }
                )
            except Exception as e:
                results.append({"case_id": case["case_id"], "error": f"{type(e).__name__}: {e}"})
                log(f"  case {case['case_id']} failed: {type(e).__name__}: {e}")
            if i % 10 == 0 or i == len(cases):
                log(f"  {i}/{len(cases)} cases done")
            vlm_models.free_vram()
    finally:
        vlm_models.unload(bundle)

    # Normalize finding dicts to int keys for metrics.
    for r in results:
        if "error" in r:
            continue
        r["predicted_findings"] = {int(k): v for k, v in r.pop("_pred").items()}
        r["ground_truth_findings"] = {int(k): v for k, v in r.pop("_gt").items()}

    metrics = compute_metrics(results)
    # JSON-serialize finding keys as strings again for dump friendliness.
    for r in results:
        if "error" in r:
            continue
        r["predicted_findings"] = {str(k): v for k, v in r["predicted_findings"].items()}
        r["ground_truth_findings"] = {str(k): v for k, v in r["ground_truth_findings"].items()}

    ab = metrics.get("abnormal_tooth") or {}
    log(
        f"  patient-avg abnormal F1={ab.get('f1_score', 0):.3f} "
        f"P={ab.get('precision', 0):.3f} R={ab.get('recall', 0):.3f} "
        f"Spec={ab.get('specificity', 0):.3f} Acc={ab.get('accuracy', 0):.3f}"
    )
    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "repo_id": spec.repo_id,
        "metrics": metrics,
        "results": results,
    }


def write_summary_csv(payload: dict[str, Any], out_path: Path) -> None:
    import csv

    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        ab = m.get("abnormal_tooth") or {}
        if not ab:
            rows.append({"Model": entry.get("display_name", entry.get("model_key")), "Status": "failed"})
            continue
        row = {
            "Model": entry["display_name"],
            "Cases": m.get("n_cases"),
            "Strategy": payload.get("strategy"),
            "Abnormal F1": ab.get("f1_score"),
            "Abnormal Precision": ab.get("precision"),
            "Abnormal Recall": ab.get("recall"),
            "Abnormal Specificity": ab.get("specificity"),
            "Abnormal Accuracy": ab.get("accuracy"),
            "Mean pred/case": m.get("mean_predicted_count"),
            "Mean GT/case": m.get("mean_ground_truth_count"),
            "Sec/case": m.get("mean_seconds_per_case"),
            "Status": "ok",
        }
        for disease, block in (m.get("per_disease") or {}).items():
            row[f"{disease} F1"] = block.get("f1_score")
        rows.append(row)

    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--cases", type=int, default=0, help="0 = all in split")
    ap.add_argument("--models", nargs="+", default=list(vlm_models.MODEL_REGISTRY))
    ap.add_argument("--strategy", choices=list(PROMPT_REGISTRY), default="zero_shot")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    prompt = PROMPT_REGISTRY[args.strategy]
    max_new_tokens = 450 if args.strategy == "cot" else 350

    cases = load_val_cases() if args.split == "val" else load_test_cases()
    if args.cases > 0:
        cases = cases[: args.cases]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.strategy}_{args.split}"
    stem = f"dentex_{tag}_{stamp}"
    log_path = args.out_dir / f"{stem}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(
        f"DENTEX split={args.split} cases={len(cases)} models={args.models} "
        f"strategy={args.strategy} aggregation=patient_mean_of_32fdi"
    )
    parser = DentexParser()
    payload = {
        "timestamp": stamp,
        "dataset": "DENTEX",
        "split": args.split,
        "tag": tag,
        "n_cases": len(cases),
        "case_ids": [c["case_id"] for c in cases],
        "strategy": args.strategy,
        "prompt": prompt,
        "aggregation": "patient_mean_of_32fdi",
        "fdi_teeth": FDI_TEETH,
        "diseases": DISEASES,
        "models": {},
    }

    for key in args.models:
        try:
            payload["models"][key] = evaluate_model(
                key,
                cases,
                parser,
                prompt=prompt,
                load_in_4bit=args.load_in_4bit,
                max_new_tokens=max_new_tokens,
                log=log,
            )
        except Exception as e:
            log(f"MODEL {key} FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            payload["models"][key] = {"model_key": key, "error": f"{type(e).__name__}: {e}"}

        out_json = args.out_dir / f"{stem}.json"
        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        log(f"checkpoint -> {out_json.name}")

    log("=== SUMMARY (patient-mean abnormal-tooth) ===")
    log(f"{'model':<24}{'F1':>8}{'Prec':>8}{'Recall':>8}{'Spec':>8}{'Acc':>8}")
    for key, entry in payload["models"].items():
        ab = (entry.get("metrics") or {}).get("abnormal_tooth")
        if not ab:
            log(f"{key:<24}{'ERROR':>8}")
            continue
        log(
            f"{entry['display_name']:<24}"
            f"{ab['f1_score']:>8.3f}{ab['precision']:>8.3f}"
            f"{ab['recall']:>8.3f}{ab['specificity']:>8.3f}{ab['accuracy']:>8.3f}"
        )
    log("DENTEX_COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
