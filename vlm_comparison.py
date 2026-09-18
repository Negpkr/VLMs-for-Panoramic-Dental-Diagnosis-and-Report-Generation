#!/usr/bin/env python3
"""Four-way missing-teeth comparison: LLaVA-1.5 vs LLaVA-Med vs HuatuoGPT-Vision vs DentVLM.

Primary evaluation is pooled-slot: concatenate all 32 tooth decisions across
patients, accumulate TP/FP/TN/FN per class, then compute P/R/F1. Headline is
Macro F1. Patient-mean scores are stored as a secondary JSON record.
Missing vs Present is derived from the parsed Missing teeth list, never from
the Total missing count.

Usage:
    python vlm_comparison.py --cases 0 --strategy zero_shot
    python vlm_comparison.py --cases 0 --strategy few_shot
    python vlm_comparison.py --cases 0 --strategy cot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
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
import eval_metrics  # noqa: E402
import vlm_models  # noqa: E402

DATA_ROOT = Path("/home/s222393187/Dental/Tufts_Dental_Database")
PANOS_DIR = DATA_ROOT / "Radiographs"
TEETH_BBOX_JSON = DATA_ROOT / "Segmentation" / "teeth_bbox.json"
RESULTS_DIR = Path("/home/s222393187/Dental/Results/model_comparison")
PARSER_VERSION = "tufts-strict-missing-line-v1"

# ---------------------------------------------------------------------------
# Prompting strategies
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use ONLY the Universal Numbering System: teeth are numbered 1-32.\n"
    "- Teeth 1-16: upper jaw (patient's right to left)\n"
    "- Teeth 17-32: lower jaw (patient's left to right)\n"
    "Do NOT use FDI numbers (11-48).\n\n"
    "Identify which teeth are missing (gaps or empty sockets).\n"
    "The Total missing line is formatting only and is not used for scoring.\n\n"
    "Answer in exactly this format and nothing else:\n"
    "Missing teeth: <comma-separated Universal numbers 1-32, or None>\n"
    "Total missing: <count>\n"
    "Confidence: High/Medium/Low"
)

FEW_SHOT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use ONLY the Universal Numbering System (teeth 1-32). Do NOT use FDI (11-48).\n"
    "- Teeth 1-16: upper jaw (patient's right to left)\n"
    "- Teeth 17-32: lower jaw (patient's left to right)\n\n"
    "Task: list missing teeth (gaps or empty sockets).\n"
    "The Total missing line is formatting only and is not used for scoring.\n\n"
    "Examples of correct answers:\n"
    "Example 1 (no missing teeth):\n"
    "Missing teeth: None\n"
    "Total missing: 0\n"
    "Confidence: High\n\n"
    "Example 2 (a few missing):\n"
    "Missing teeth: 1, 16, 17, 32\n"
    "Total missing: 4\n"
    "Confidence: High\n\n"
    "Example 3 (several missing):\n"
    "Missing teeth: 3, 14, 19, 30\n"
    "Total missing: 4\n"
    "Confidence: Medium\n\n"
    "Now answer for THIS radiograph in exactly the same format:\n"
    "Missing teeth: <comma-separated Universal numbers 1-32, or None>\n"
    "Total missing: <count>\n"
    "Confidence: High/Medium/Low"
)

COT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use ONLY the Universal Numbering System (teeth 1-32). Do NOT use FDI (11-48).\n"
    "- Teeth 1-16: upper jaw (patient's right to left)\n"
    "- Teeth 17-32: lower jaw (patient's left to right)\n\n"
    "Think step by step before answering:\n"
    "1. Scan the upper arch (teeth 1-16). Note empty spaces / missing crowns.\n"
    "2. Scan the lower arch (teeth 17-32). Note empty spaces / missing crowns.\n"
    "3. Combine the missing tooth numbers.\n"
    "4. Double-check that every listed number is a Universal ID from 1-32.\n\n"
    "The Total missing line is formatting only and is not used for scoring.\n"
    "Write a short Reasoning section, then give the final answer in exactly this format:\n"
    "Reasoning: <brief steps>\n"
    "Missing teeth: <comma-separated Universal numbers 1-32, or None>\n"
    "Total missing: <count>\n"
    "Confidence: High/Medium/Low"
)

# Legacy aliases kept for older CLI / notebook reproduction.
COMPACT_PROMPT = ZERO_SHOT_PROMPT
MISSING_TEETH_PROMPT = (
    "You are a dental radiologist. Look at this panoramic X-ray and identify missing teeth.\n\n"
    "Teeth are numbered 1-32:\n"
    "- Teeth 1-16: Upper jaw (right to left)\n"
    "- Teeth 17-32: Lower jaw (left to right)\n\n"
    "Look for gaps or empty spaces where teeth should be.\n\n"
    "Answer in this exact format:\n"
    "Missing teeth: [list tooth numbers, e.g., 3, 14, 19]\n"
    "Total missing: [count]\n"
    "Confidence: High/Medium/Low\n\n"
    "Examples:\n"
    "- No missing: Missing teeth: None, Total missing: 0, Confidence: High\n"
    "- Some missing: Missing teeth: 3, 14, 19, Total missing: 3, Confidence: High\n"
    "- Many missing: Missing teeth: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, "
    "17, 18, 19, 20, Total missing: 20, Confidence: High\n\n"
    "Be thorough - look at the entire dental arch."
)

PROMPT_REGISTRY: dict[str, str] = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT,
    "cot": COT_PROMPT,
    "compact": ZERO_SHOT_PROMPT,
    "notebook": MISSING_TEETH_PROMPT,
}


def recover_json_array(path: Path) -> list[dict]:
    """Read as many complete records as possible from a JSON array."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    start = raw.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    idx = start + 1
    items: list[dict] = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, idx = decoder.raw_decode(raw, idx)
        except ValueError:
            break
        items.append(obj)
    return items


class GroundTruthExtractor:
    """Missing teeth per case, derived from teeth_bbox.json."""

    def __init__(self, bbox_json: Path = TEETH_BBOX_JSON):
        self.truncated = False
        try:
            records = json.loads(bbox_json.read_text())
        except json.JSONDecodeError:
            records = recover_json_array(bbox_json)
            self.truncated = True
        if isinstance(records, dict):
            records = list(records.values())

        self.by_case: dict[str, Any] = {}
        for record in records:
            external_id = record.get("External ID")
            if external_id:
                self.by_case[Path(external_id).stem] = record

    def extract_missing_teeth(self, case_id: str) -> list[int]:
        record = self.by_case.get(case_id)
        if not record:
            return []
        detected: set[int] = set()
        for obj in record.get("Label", {}).get("objects", []):
            for n in re.findall(r"\d+", str(obj.get("title", ""))):
                if 1 <= int(n) <= 32:
                    detected.add(int(n))
        return sorted(set(range(1, 33)) - detected)

    def available_cases(self) -> list[str]:
        return sorted(self.by_case, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))


class MissingTeethParser:
    """Parse model replies into Universal tooth IDs 1-32.

    ``strict`` reads only the ``Missing teeth:`` line (preferred).
    ``loose`` collects every 1-32 integer in the reply (legacy).
    """

    tooth_pattern = re.compile(r"\b([1-9]|[12][0-9]|3[0-2])\b")
    missing_line = re.compile(r"Missing teeth:\s*([^\n]+)", re.IGNORECASE)
    confidence_pattern = re.compile(r"Confidence:\s*(High|Medium|Low)", re.IGNORECASE)

    def __init__(self, max_missing_teeth: int = 28, mode: str = "strict"):
        self.max_missing_teeth = max_missing_teeth
        self.mode = mode

    def extract_missing_teeth(self, text: str) -> list[int]:
        """Parse Missing teeth: only. The Total missing line is never scored."""
        text = re.sub(r"Total missing:\s*[^\n]*", " ", text, flags=re.IGNORECASE)
        source = text
        if self.mode == "strict":
            match = self.missing_line.search(text)
            if not match:
                return []
            source = match.group(1)
            if re.search(r"\bnone\b", source, re.IGNORECASE):
                return []
        teeth = sorted({int(t) for t in self.tooth_pattern.findall(source) if 1 <= int(t) <= 32})
        return teeth[: self.max_missing_teeth]

    def confidence(self, text: str) -> str:
        match = self.confidence_pattern.search(text)
        return match.group(1).title() if match else "Medium"


def find_image(case_id: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = PANOS_DIR / f"{case_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_is_readable(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def usable_cases(gt: "GroundTruthExtractor") -> list[str]:
    out = []
    for case_id in gt.available_cases():
        path = find_image(case_id)
        if path is not None and image_is_readable(path):
            out.append(case_id)
    return out


def to_binary_32(missing_teeth: list[int] | set[int]) -> list[int]:
    """Fixed-length 32 binary labels: 1 = missing, 0 = not missing."""
    missing = set(missing_teeth)
    return [1 if tooth in missing else 0 for tooth in range(1, 33)]


def to_position_array(missing_teeth: list[int] | set[int]) -> list[int]:
    """Fixed-length 32 array: tooth number if missing else 0."""
    missing = set(missing_teeth)
    return [tooth if tooth in missing else 0 for tooth in range(1, 33)]


def _specificity(y_true: list[int], y_pred: list[int], zero_division: float = 0.0) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _fn, _tp = cm.ravel()
    denom = tn + fp
    if denom == 0:
        return float(zero_division)
    return float(tn / denom)


def _balanced_accuracy(recall: float, specificity: float) -> float:
    """(Recall + Specificity) / 2 — accuracy that is not inflated by class imbalance."""
    return float(0.5 * (recall + specificity))


def sample_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Metrics for one patient's 32 tooth decisions."""
    assert len(y_true) == 32 and len(y_pred) == 32
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = _specificity(y_true, y_pred, zero_division=0)
    classes = eval_metrics.per_class_scores(
        y_true, y_pred, labels=(0, 1), names=("present", "missing")
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": rec,
        "specificity": spec,
        "balanced_accuracy": _balanced_accuracy(rec, spec),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "confusion_matrix": cm.tolist(),
        "per_class": classes["per_class"],
        "macro": classes["macro"],
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled 32-tooth metrics across the evaluation split.

    Primary: concatenate all 32 slots from all patients, then score.
    Secondary: mean of per-patient 32-tooth scores (previous protocol).
    """
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n_cases": 0, "aggregation": "pooled_slots"}

    per_sample: list[dict[str, Any]] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    for r in ok:
        y_true = to_binary_32(r["ground_truth_teeth"])
        y_pred = to_binary_32(r["predicted_teeth"])
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        m = sample_metrics(y_true, y_pred)
        m["case_id"] = r["case_id"]
        m["y_true_32"] = y_true
        m["y_pred_32"] = y_pred
        m["gt_position_array"] = to_position_array(r["ground_truth_teeth"])
        m["pred_position_array"] = to_position_array(r["predicted_teeth"])
        per_sample.append(m)
        # Attach onto the result dict for downstream inspection.
        r["sample_metrics"] = {
            k: m[k]
            for k in (
                "accuracy",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "f1_score",
                "tp",
                "fp",
                "fn",
                "tn",
            )
        }
        r["per_class"] = m["per_class"]
        r["macro"] = m["macro"]
        r["y_true_32"] = y_true
        r["y_pred_32"] = y_pred

    def mean_of(key: str) -> float:
        return float(np.mean([s[key] for s in per_sample]))

    total_cm = np.zeros((2, 2), dtype=int)
    for s in per_sample:
        total_cm += np.array(s["confusion_matrix"], dtype=int)

    class_mean = eval_metrics.mean_per_class(per_sample)
    patient_avg = {
        "aggregation": "patient_mean_of_32tooth",
        "accuracy": mean_of("accuracy"),
        "precision": mean_of("precision"),
        "recall": mean_of("recall"),
        "specificity": mean_of("specificity"),
        "balanced_accuracy": mean_of("balanced_accuracy"),
        "f1_score": mean_of("f1_score"),
        "confusion_matrix": total_cm.tolist(),
        "per_class": class_mean.get("per_class", {}),
        "macro": class_mean.get("macro", {}),
    }
    pooled = eval_metrics.pooled_metrics(
        y_true_all, y_pred_all, labels=(0, 1), names=("present", "missing")
    )
    yt_patients = [s["y_true_32"] for s in per_sample]
    yp_patients = [s["y_pred_32"] for s in per_sample]
    mac_ci = eval_metrics.bootstrap_binary_macro_f1(
        yt_patients,
        yp_patients,
        names=("present", "missing"),
        n_boot=eval_metrics.PAPER_BOOTSTRAP_N,
        seed=eval_metrics.PAPER_BOOTSTRAP_SEED,
        cluster_ids=[eval_metrics.patient_id_for_image(s["case_id"]) for s in per_sample],
    )
    pooled.setdefault("macro", {})["f1_ci95"] = mac_ci

    # Image-level: does the case have any missing teeth? (secondary)
    y_true_img = [1 if r["ground_truth_teeth"] else 0 for r in ok]
    y_pred_img = [1 if r["predicted_teeth"] else 0 for r in ok]
    rec_img = float(recall_score(y_true_img, y_pred_img, zero_division=0))
    spec_img = _specificity(y_true_img, y_pred_img, zero_division=0)
    image_level = {
        "accuracy": float(accuracy_score(y_true_img, y_pred_img)),
        "precision": float(precision_score(y_true_img, y_pred_img, zero_division=0)),
        "recall": rec_img,
        "specificity": spec_img,
        "balanced_accuracy": _balanced_accuracy(rec_img, spec_img),
        "f1_score": float(f1_score(y_true_img, y_pred_img, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_img, y_pred_img, labels=[0, 1]).tolist(),
    }

    return {
        "n_cases": len(ok),
        "n_failed": len(results) - len(ok),
        "aggregation": "pooled_slots",
        "per_tooth": pooled,
        "per_tooth_patient_mean": patient_avg,
        "image_level": image_level,
        "bootstrap": {
            "n_boot": eval_metrics.PAPER_BOOTSTRAP_N,
            "seed": eval_metrics.PAPER_BOOTSTRAP_SEED,
            "unit": "patient",
            "n_patients": mac_ci.get("n_patients"),
            "n_images": mac_ci.get("n_images"),
            "patient_equals_image": mac_ci.get("patient_equals_image"),
            "note": mac_ci.get("note"),
            "macro_f1_ci95": mac_ci,
        },
        "per_sample": per_sample,
        "mean_predicted_count": sum(len(r["predicted_teeth"]) for r in ok) / len(ok),
        "mean_ground_truth_count": sum(len(r["ground_truth_teeth"]) for r in ok) / len(ok),
        "mean_seconds_per_case": sum(r["seconds"] for r in ok) / len(ok),
    }


def evaluate_model(
    model_key: str,
    case_ids: list[str],
    gt: GroundTruthExtractor,
    parser: MissingTeethParser,
    *,
    prompt: str = ZERO_SHOT_PROMPT,
    load_in_4bit: bool = False,
    max_new_tokens: int | None = None,
    log=print,
) -> dict[str, Any]:
    """Load one model, score every case, then release the GPU."""
    spec = vlm_models.MODEL_REGISTRY[model_key]
    log(f"=== {spec.display_name} ({spec.repo_id}) ===")

    gen_overrides: dict[str, Any] = {}
    if max_new_tokens is not None:
        gen_overrides["max_new_tokens"] = max_new_tokens

    bundle = None
    results: list[dict[str, Any]] = []
    try:
        bundle = vlm_models.load_model(model_key, load_in_4bit=load_in_4bit)
        log(f"loaded in {bundle.load_seconds:.1f}s on {bundle.device}")

        for i, case_id in enumerate(case_ids, 1):
            image_path = find_image(case_id)
            if image_path is None:
                results.append({"case_id": case_id, "error": "image not found"})
                continue
            t0 = time.time()
            try:
                reply = vlm_models.generate(
                    bundle,
                    image_path,
                    prompt,
                    seed_key=f"{model_key}:{case_id}",
                    **gen_overrides,
                )
                predicted = parser.extract_missing_teeth(reply)
                results.append(
                    {
                        "case_id": case_id,
                        "response": reply,
                        "raw_model_output": reply,
                        "parser_version": PARSER_VERSION,
                        "predicted_teeth": predicted,
                        "ground_truth_teeth": gt.extract_missing_teeth(case_id),
                        "confidence": parser.confidence(reply),
                        "seconds": time.time() - t0,
                    }
                )
            except Exception as e:
                results.append({"case_id": case_id, "error": f"{type(e).__name__}: {e}"})
                log(f"  case {case_id} failed: {type(e).__name__}: {e}")

            if i % 10 == 0 or i == len(case_ids):
                log(f"  {i}/{len(case_ids)} cases done")
            vlm_models.free_vram()
    finally:
        vlm_models.unload(bundle)

    metrics = compute_metrics(results)
    pt = metrics.get("per_tooth") or {}
    log(
        f"  pooled missing F1={pt.get('f1_score', 0):.3f} "
        f"P={pt.get('precision', 0):.3f} R={pt.get('recall', 0):.3f} "
        f"Spec={pt.get('specificity', 0):.3f} BalAcc={pt.get('balanced_accuracy', 0):.3f} "
        f"Acc={pt.get('accuracy', 0):.3f} MacroF1={(pt.get('macro') or {}).get('f1_score', 0):.3f}"
    )
    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "repo_id": spec.repo_id,
        "metrics": metrics,
        "results": results,
    }


def _model_complete(entry: dict[str, Any]) -> bool:
    return bool((entry.get("metrics") or {}).get("per_tooth"))


def gt_slot_support(case_ids: list[str], gt: GroundTruthExtractor) -> dict[str, int]:
    missing = 0
    for case_id in case_ids:
        missing += len(gt.extract_missing_teeth(case_id))
    n_slots = 32 * len(case_ids)
    return {"missing": missing, "present": n_slots - missing, "n_slots": n_slots, "n_cases": len(case_ids)}


def write_summary_csv(payload: dict[str, Any], out_path: Path) -> None:
    import csv

    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        pt = m.get("per_tooth")
        if not pt:
            rows.append({"Model": entry.get("display_name", entry.get("model_key")), "Status": "failed"})
            continue
        row = {
            "Model": entry["display_name"],
            "Cases": m.get("n_cases"),
            "Strategy": payload.get("strategy") or payload.get("prompt_name"),
            "Aggregation": m.get("aggregation"),
            **eval_metrics.flatten_binary_main(pt, "missing", "present"),
            "Mean_pred_per_case": m.get("mean_predicted_count"),
            "Mean_GT_per_case": m.get("mean_ground_truth_count"),
            "Sec_per_case": m.get("mean_seconds_per_case"),
            "Status": "ok",
        }
        rows.append(row)
    if not rows:
        return
    fieldnames = []
    for r in rows:
        for kk in r.keys():
            if kk not in fieldnames:
                fieldnames.append(kk)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_supplementary_csv(payload: dict[str, Any], out_path: Path) -> None:
    import csv

    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        pt = m.get("per_tooth")
        if not pt:
            continue
        rows.append(
            {
                "Model": entry["display_name"],
                "Cases": m.get("n_cases"),
                "Strategy": payload.get("strategy") or payload.get("prompt_name"),
                **eval_metrics.flatten_binary_report(pt, "missing", "present"),
            }
        )
    if not rows:
        return
    fieldnames = []
    for r in rows:
        for kk in r.keys():
            if kk not in fieldnames:
                fieldnames.append(kk)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases", type=int, default=0, help="number of cases to evaluate (0 = all usable)"
    )
    ap.add_argument("--models", nargs="+", default=list(vlm_models.MODEL_REGISTRY))
    ap.add_argument("--parse-mode", choices=["loose", "strict"], default="strict")
    ap.add_argument(
        "--strategy",
        "--prompt",
        dest="strategy",
        choices=list(PROMPT_REGISTRY),
        default="zero_shot",
        help="prompting strategy (zero_shot / few_shot / cot); legacy aliases: compact, notebook",
    )
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="optional tag appended to output filenames (e.g. zero_shot_full)",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="continue an interrupted run; skip models that already have metrics",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print case count, models, and prompt length, then exit",
    )
    args = ap.parse_args()

    prompt = PROMPT_REGISTRY[args.strategy]
    max_new_tokens = 400 if args.strategy == "cot" else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or args.strategy
    stem = f"comparison_{tag}_{stamp}"
    log_path = args.out_dir / f"{stem}.log"
    resume_payload: dict[str, Any] | None = None
    skip_models: list[str] = []

    if args.resume_from:
        resume_payload = json.loads(args.resume_from.read_text())
        stem = args.resume_from.stem
        log_path = args.resume_from.with_suffix(".log")
        skip_models = [
            k for k, entry in resume_payload.get("models", {}).items() if _model_complete(entry)
        ]
        args.models = [k for k in args.models if k not in skip_models]
        if resume_payload.get("strategy") and resume_payload["strategy"] != args.strategy:
            raise SystemExit(
                f"--resume-from strategy={resume_payload['strategy']} "
                f"does not match --strategy {args.strategy}"
            )

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    gt = GroundTruthExtractor()
    parser = MissingTeethParser(mode=args.parse_mode)

    all_cases = usable_cases(gt)
    case_ids = all_cases if args.cases <= 0 else all_cases[: args.cases]
    if args.resume_from and resume_payload and resume_payload.get("case_ids"):
        case_ids = list(resume_payload["case_ids"])

    if args.dry_run:
        print(
            f"Tufts dry-run cases={len(case_ids)} strategy={args.strategy} "
            f"models={args.models} parse_mode={args.parse_mode} "
            f"max_new_tokens={max_new_tokens} prompt_chars={len(prompt)}"
        )
        if skip_models:
            print(f"would skip completed models: {skip_models}")
        print("first_case", case_ids[0] if case_ids else None)
        return 0

    if gt.truncated:
        log(
            f"WARNING teeth_bbox.json is truncated: only {len(gt.by_case)} of ~1000 "
            f"annotations recovered; {len(all_cases)} of those have a readable radiograph"
        )
    support = gt_slot_support(case_ids, gt)
    log(
        f"cases={len(case_ids)} models={args.models} strategy={args.strategy} "
        f"parse_mode={args.parse_mode} aggregation=pooled_slots "
        f"scoring=parsed_Missing_teeth_line_only"
    )
    log(
        f"GT slot support: Missing n={support['missing']}  "
        f"Present n={support['present']}  slots={support['n_slots']}"
    )
    if skip_models:
        log(f"resume from {args.resume_from.name}; skipping completed models: {skip_models}")

    payload = resume_payload or {
        "timestamp": stamp,
        "tag": tag,
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "parse_mode": args.parse_mode,
        "prompt_name": args.strategy,
        "strategy": args.strategy,
        "prompt": prompt,
        "aggregation": "pooled_slots",
        "parser_version": PARSER_VERSION,
        "run_config": {
            "dataset": "Tufts",
            "n_cases_paper": len(case_ids),
            "n_cases_note": "Paper evaluation is all usable Tufts cases (1000). The Aug 2026 100-case JSONs are archive only.",
            "strategy": args.strategy,
            "parser_version": PARSER_VERSION,
            "parse_source": "Missing teeth line only; Total missing ignored",
            "aggregation": "pooled_slots",
            "bootstrap_n": eval_metrics.PAPER_BOOTSTRAP_N,
            "bootstrap_seed": eval_metrics.PAPER_BOOTSTRAP_SEED,
            "bootstrap_unit": "patient",
            "bootstrap_patient_equals_image": True,
            "bootstrap_note": eval_metrics.BOOTSTRAP_UNIT_NOTE,
            "few_shot_uses_exemplar_images": False,
            "few_shot_examples": "synthetic text templates, not Tufts case IDs",
        },
        "gt_support": support,
        "models": {},
    }
    payload["gt_support"] = support

    if not args.models:
        log("nothing left to run; all requested models already complete")
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        write_supplementary_csv(payload, args.out_dir / f"{stem}_supplementary.csv")
        log("COMPARISON_DONE")
        return 0

    for key in args.models:
        try:
            payload["models"][key] = evaluate_model(
                key,
                case_ids,
                gt,
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
        write_supplementary_csv(payload, args.out_dir / f"{stem}_supplementary.csv")
        log(f"checkpoint -> {out_json.name}")

    csv_path = args.out_dir / f"{stem}_summary.csv"
    supp_path = args.out_dir / f"{stem}_supplementary.csv"
    write_summary_csv(payload, csv_path)
    write_supplementary_csv(payload, supp_path)
    log("=== GT SUPPORT (not a paper table) ===")
    log(f"  Missing n={support['missing']}  Present n={support['present']}  cases={support['n_cases']}  slots={support['n_slots']}")
    log("=== TABLE A missing vs present (pooled; per-class F1 + Macro F1 95% CI + Acc + BalAcc) ===")
    log(
        f"{'model':<22}{'MissF1':>8}{'PresF1':>8}{'MacF1':>8}{'Acc':>8}{'BalAcc':>8}{'CI95':>19}"
    )
    for key, entry in payload["models"].items():
        pt = entry.get("metrics", {}).get("per_tooth")
        if not pt:
            log(f"{key:<22}{'ERROR':>8}")
            continue
        row = eval_metrics.flatten_binary_main(pt, "missing", "present")
        lo, hi = row.get("Macro_F1_CI95_low"), row.get("Macro_F1_CI95_high")
        ci_s = f"[{lo:.3f},{hi:.3f}]" if lo is not None and hi is not None else "n/a"
        log(
            f"{entry['display_name']:<22}"
            f"{(row['missing_F1'] or 0):>8.3f}"
            f"{(row['present_F1'] or 0):>8.3f}"
            f"{(row['Macro_F1'] or 0):>8.3f}"
            f"{(row['Accuracy'] or 0):>8.3f}"
            f"{(row['Balanced_Accuracy'] or 0):>8.3f}"
            f"{ci_s:>19}"
        )
    log(f"summary csv -> {csv_path.name}  supplementary P/R -> {supp_path.name}")
    log("COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
