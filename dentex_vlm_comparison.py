#!/usr/bin/env python3
"""DENTEX disease + FDI tooth enumeration comparison.

Splits
  train  705 images, official QED labels
  val     50 images, official QED labels
  test   250 images, LabelMe labels (best-effort; no Deep Caries class)
  qed    train + val on disk (755), minus 3 held-out text exemplars → 752 scored
  all    train + val + test, with the same 3 train IDs excluded from scoring

DENTEX annotates diagnosis only for abnormal teeth. Unannotated FDI slots
are scored as "No annotated finding", not "Normal". QED labels are
multi-label: one tooth may have more than one diagnosis. Each disease is
scored independently (pooled one-vs-rest). Table A is abnormal vs
no-finding. Table B is the four diagnoses only; Macro-4 is their
unweighted mean. Abnormality is derived from parsed FDI findings, never
from the model's Total abnormal count. A model that predicts no finding
on every slot has Macro F1 = no-finding F1 / 2 (the no-finding-only
baseline, not a mathematical floor) and accuracy equal to the no-finding
prior. Summary-CSV mean pred/GT counts are unique abnormal teeth per
case, not diagnosis-label counts.

Do not launch from this module. Use runs/run_dentex_strategies.sh when ready.

    python dentex_vlm_comparison.py --split all --strategy zero_shot --dry-run
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
import eval_metrics  # noqa: E402
import vlm_models  # noqa: E402
import dataset_inventory  # noqa: E402

DENTEX_ROOT = Path("/home/s222393187/Dental/DENTEX")
VAL_JSON = DENTEX_ROOT / "validation_triple.json"
VAL_XRAYS = DENTEX_ROOT / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TRAIN_JSON = (
    DENTEX_ROOT
    / "training_data"
    / "quadrant-enumeration-disease"
    / "train_quadrant_enumeration_disease.json"
)
TRAIN_XRAYS = DENTEX_ROOT / "training_data" / "quadrant-enumeration-disease" / "xrays"
TEST_INPUT = DENTEX_ROOT / "disease" / "input"
TEST_LABEL = DENTEX_ROOT / "disease" / "label"
RESULTS_DIR = Path("/home/s222393187/Dental/Results/dentex_comparison")
SPLIT_CHOICES = ("train", "val", "test", "qed", "all")

# Permanent dentition in FDI (32 positions).
FDI_TEETH: list[int] = (
    list(range(11, 19))
    + list(range(21, 29))
    + list(range(31, 39))
    + list(range(41, 49))
)
PARSER_VERSION = "dentex-multilabel-v1"
BOOTSTRAP_N = eval_metrics.PAPER_BOOTSTRAP_N
BOOTSTRAP_SEED = eval_metrics.PAPER_BOOTSTRAP_SEED
NO_FINDING = eval_metrics.NO_ANNOTATED_FINDING
BINARY_NAMES = (NO_FINDING, "abnormal")
DISEASES = ["Impacted", "Caries", "Deep Caries", "Periapical Lesion"]

ZERO_SHOT_PROMPT = (
    "This is a panoramic dental X-ray.\n"
    "Use the FDI tooth numbering system ONLY (11-18, 21-28, 31-38, 41-48).\n"
    "Do NOT use Universal numbers 1-32.\n\n"
    "Task: find abnormal teeth and list every diagnosis that applies from:\n"
    "Impacted, Caries, Deep Caries, Periapical Lesion.\n"
    "One tooth may have more than one diagnosis. If so, write a separate line "
    "for each diagnosis on that same FDI.\n"
    "Only list teeth that clearly show one or more of these abnormalities.\n"
    "If none, write None.\n"
    "The Total abnormal line is formatting only and is not used for scoring.\n\n"
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
    "Diagnoses allowed: Impacted, Caries, Deep Caries, Periapical Lesion.\n"
    "One tooth may have more than one diagnosis; write one line per diagnosis.\n"
    "The Total abnormal line is formatting only and is not used for scoring.\n\n"
    "Example 1 (impacted wisdom teeth):\n"
    "Findings:\n"
    "18: Impacted\n"
    "28: Impacted\n"
    "38: Impacted\n"
    "48: Impacted\n"
    "Total abnormal: 4\n"
    "Confidence: High\n\n"
    "Example 2 (caries, including one tooth with two findings):\n"
    "Findings:\n"
    "16: Caries\n"
    "36: Deep Caries\n"
    "46: Caries\n"
    "46: Periapical Lesion\n"
    "Total abnormal: 3\n"
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
    "Diagnoses: Impacted, Caries, Deep Caries, Periapical Lesion.\n"
    "One tooth may have more than one diagnosis; write one line per diagnosis.\n"
    "The Total abnormal line is formatting only and is not used for scoring.\n\n"
    "Think step by step:\n"
    "1. Inspect each quadrant (1=UR, 2=UL, 3=LL, 4=LR).\n"
    "2. Note impacted third molars, caries, deep caries, periapical lesions.\n"
    "3. Assign FDI IDs and every diagnosis that applies to each tooth.\n\n"
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


def _findings_block(gt: dict) -> str:
    items = {int(k): list(v) for k, v in (gt or {}).items()}
    if not items:
        return "Findings:\nNone\nTotal abnormal: 0\nConfidence: High"
    lines = ["Findings:"]
    for fdi in sorted(items):
        for lab in items[fdi]:
            lines.append(f"{fdi}: {lab}")
    lines.append(f"Total abnormal: {len(items)}")
    lines.append("Confidence: Medium")
    return "\n".join(lines)


def few_shot_prompt_from_heldout() -> str:
    """Text exemplars from held-out TRAIN cases. Those cases are not scored."""
    held = dataset_inventory.heldout_payload()
    titles = {
        "impacted_only": "Example 1 (impacted teeth)",
        "multi_label_or_mixed": "Example 2 (mixed findings, including a tooth with two diagnoses)",
        "no_annotated_finding": "Example 3 (none)",
    }
    header = (
        "This is a panoramic dental X-ray.\n"
        "Use FDI numbering ONLY (11-18, 21-28, 31-38, 41-48).\n"
        "Diagnoses allowed: Impacted, Caries, Deep Caries, Periapical Lesion.\n"
        "One tooth may have more than one diagnosis; write one line per diagnosis.\n"
        "The Total abnormal line is formatting only and is not used for scoring.\n\n"
    )
    blocks = []
    for i, ex in enumerate(held.get("exemplars") or [], 1):
        title = titles.get(ex.get("role"), f"Example {i}")
        blocks.append(f"{title}:\n{_findings_block(ex.get('gt') or {})}")
    return header + "\n\n".join(blocks) + "\n\nNow answer for THIS radiograph in the same format."


def prompt_for_strategy(strategy: str) -> str:
    if strategy == "few_shot":
        return few_shot_prompt_from_heldout()
    return PROMPT_REGISTRY[strategy]

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


def _balanced_accuracy(recall: float, specificity: float) -> float:
    """(Recall + Specificity) / 2 — accuracy that is not inflated by class imbalance."""
    return float(0.5 * (recall + specificity))


def sample_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = _specificity(y_true, y_pred, zero_division=0)
    classes = eval_metrics.per_class_scores(
        y_true, y_pred, labels=(0, 1), names=BINARY_NAMES
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
        "per_class": classes["per_class"],
        "macro": classes["macro"],
    }


def to_binary_fdi(abnormal_fdis: set[int] | list[int]) -> list[int]:
    abnormal = set(abnormal_fdis)
    return [1 if t in abnormal else 0 for t in FDI_TEETH]


class DentexParser:
    """Parse FDI + disease findings from model text."""

    fdi_pattern = re.compile(r"\b([1-4][1-8])\b")
    line_pattern = re.compile(
        r"\b([1-4][1-8])\b\s*[:\-–]\s*(Impacted|Deep\s*Caries|Caries|Periapical\s*Lesion)",
        re.IGNORECASE,
    )
    # Also accept "18 Impacted" without colon.
    loose_pattern = re.compile(
        r"\b([1-4][1-8])\b[^\n]{0,20}?\b(Impacted|Deep\s*Caries|Caries|Periapical\s*Lesion)\b",
        re.IGNORECASE,
    )
    extra_disease = re.compile(
        r"(Impacted|Deep\s*Caries|Caries|Periapical\s*Lesion)",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalize_disease(raw: str) -> str | None:
        key = re.sub(r"\s+", " ", raw).strip().lower()
        mapping = {
            "impacted": "Impacted",
            "caries": "Caries",
            "deep caries": "Deep Caries",
            "periapical lesion": "Periapical Lesion",
        }
        return mapping.get(key)

    def _add_finding(self, findings: dict[int, list[str]], fdi: int, disease: str | None) -> None:
        if not disease or fdi not in FDI_TEETH:
            return
        slot = findings.setdefault(fdi, [])
        if disease not in slot:
            slot.append(disease)

    def parse(self, text: str) -> dict[int, list[str]]:
        """Parse FDI→diagnoses. Ignore the Total abnormal line; it is not scored."""
        findings: dict[int, list[str]] = {}
        if re.search(r"Findings:\s*None\b", text, re.IGNORECASE):
            return {}
        for pattern in (self.line_pattern, self.loose_pattern):
            for m in pattern.finditer(text):
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.start())
                if line_end < 0:
                    line_end = len(text)
                if re.search(r"Total\s+abnormal", text[line_start:line_end], re.IGNORECASE):
                    continue
                fdi = int(m.group(1))
                self._add_finding(findings, fdi, self._normalize_disease(m.group(2)))
                rest = text[m.end() : line_end]
                for extra in self.extra_disease.finditer(rest):
                    self._add_finding(findings, fdi, self._normalize_disease(extra.group(1)))
        return findings

    def confidence(self, text: str) -> str:
        m = re.search(r"Confidence:\s*(High|Medium|Low)", text, re.IGNORECASE)
        return m.group(1).title() if m else "Medium"


def _sort_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(case: dict[str, Any]):
        stem = case["case_id"]
        tail = stem.split("_")[-1]
        return (stem.rsplit("_", 1)[0], not tail.isdigit(), int(tail) if tail.isdigit() else stem)

    return sorted(cases, key=key)


def load_qed_json(json_path: Path, xray_dir: Path, split_name: str) -> list[dict[str, Any]]:
    """Load DENTEX hierarchical JSON (train or val) into FDI→list-of-diseases cases."""
    payload = json.loads(json_path.read_text())
    c1 = {c["id"]: str(c["name"]) for c in payload["categories_1"]}
    c2 = {c["id"]: str(c["name"]) for c in payload["categories_2"]}
    c3 = {c["id"]: str(c["name"]) for c in payload["categories_3"]}

    by_img: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for a in payload["annotations"]:
        fdi = int(f"{c1[a['category_id_1']]}{c2[a['category_id_2']]}")
        disease = c3[a["category_id_3"]]
        by_img[a["image_id"]][fdi].add(disease)

    cases = []
    for im in payload["images"]:
        path = xray_dir / im["file_name"]
        if not path.exists():
            continue
        gt = {int(fdi): sorted(labels) for fdi, labels in by_img.get(im["id"], {}).items()}
        cases.append(
            {
                "case_id": Path(im["file_name"]).stem,
                "file_name": im["file_name"],
                "image_path": path,
                "split": split_name,
                "gt_findings": gt,
            }
        )
    return _sort_cases(cases)


def load_val_cases() -> list[dict[str, Any]]:
    return load_qed_json(VAL_JSON, VAL_XRAYS, "val")


def load_train_cases() -> list[dict[str, Any]]:
    return load_qed_json(TRAIN_JSON, TRAIN_XRAYS, "train")


def load_test_cases() -> list[dict[str, Any]]:
    """Optional: parse LabelMe test labels into FDI→disease (best-effort)."""
    cases = []
    for img in sorted(TEST_INPUT.glob("test_*.png")):
        label_path = TEST_LABEL / f"{img.stem}.json"
        gt: dict[int, list[str]] = {}
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
                    slot = gt.setdefault(fdi, [])
                    if disease not in slot:
                        slot.append(disease)
        cases.append(
            {
                "case_id": img.stem,
                "file_name": img.name,
                "image_path": img,
                "split": "test",
                "gt_findings": gt,
            }
        )
    return cases


def load_split(split: str) -> list[dict[str, Any]]:
    if split == "val":
        cases = load_val_cases()
    elif split == "train":
        cases = load_train_cases()
    elif split == "test":
        cases = load_test_cases()
    elif split == "qed":
        cases = load_train_cases() + load_val_cases()
    elif split == "all":
        cases = load_train_cases() + load_val_cases() + load_test_cases()
    else:
        raise ValueError(f"unknown split: {split}")
    if split in {"qed", "train", "all"}:
        held = set(dataset_inventory.heldout_case_ids())
        cases = [c for c in cases if c["case_id"] not in held]
        remaining = [c["case_id"] for c in cases if c["case_id"] in held]
        if remaining:
            raise RuntimeError(f"few-shot exemplars leaked into scored split: {remaining}")
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


def count_gt_disease_support(cases: list[dict[str, Any]]) -> dict[str, int]:
    return eval_metrics.gt_disease_support(
        [c.get("gt_findings") or {} for c in cases], DISEASES
    )


def log_gt_disease_support(log, cases: list[dict[str, Any]]) -> dict[str, Any]:
    findings = [c.get("gt_findings") or {} for c in cases]
    n_unique_abn = sum(len(gt) for gt in findings)
    table_a = eval_metrics.table_a_binary_support(len(cases), n_unique_abn)
    disease_n = eval_metrics.gt_disease_support(findings, DISEASES)
    diag_sum = int(sum(disease_n.values()))
    missing = [name for name, n in disease_n.items() if n == 0]
    log(
        "=== TABLE A binary support (unique abnormal teeth vs no-finding slots; "
        "these partition n_images × 32) ==="
    )
    log(f"  n_images                 {table_a['n_images']}")
    log(f"  n_slots                  {table_a['n_slots']}")
    log(f"  Abnormal                 n={table_a['abnormal']}")
    log(f"  No annotated finding     n={table_a['no_annotated_finding']}")
    log(
        f"  check {table_a['abnormal']} + {table_a['no_annotated_finding']} = "
        f"{table_a['sum']}  {'OK' if table_a['consistent'] else 'MISMATCH'} "
        f"(expected {table_a['n_slots']})"
    )
    if not table_a["consistent"]:
        log("WARNING: Table A binary supports do not sum to n_images × 32")
    log("=== TABLE B GT-positive tooth-slot support n (one-vs-rest; not unique teeth) ===")
    for name in DISEASES:
        note = "  [undefined: no GT examples; excluded from Macro-4]" if disease_n[name] == 0 else ""
        log(f"  {name:<22} n={disease_n[name]}{note}")
    log(
        f"  diagnosis-positive sum   n={diag_sum}  "
        f"(can exceed unique abnormal teeth={n_unique_abn} when teeth are multi-label)"
    )
    if missing:
        used = [name for name in DISEASES if disease_n[name] > 0]
        log(
            f"WARNING: {', '.join(missing)} ha{'s' if len(missing)==1 else 've'} "
            f"zero GT support. Macro-4 uses: {', '.join(used) or 'none'}."
        )
    else:
        log("All four DENTEX diagnoses have GT support; Macro-4 uses all four classes.")
    return {
        "n_images": table_a["n_images"],
        "n_slots": table_a["n_slots"],
        "table_a": table_a,
        "table_b_gt_positive_slots": disease_n,
        "diagnosis_positive_slots_sum": diag_sum,
        **disease_n,
    }


def _findings_map(raw: dict | None) -> dict[int, list[str]]:
    return {int(k): eval_metrics.labels_on_tooth(v) for k, v in (raw or {}).items()}


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n_cases": 0, "aggregation": "pooled_slots"}

    per_sample = []
    disease_rows: dict[str, list[dict[str, float]]] = {d: [] for d in DISEASES}
    exclusive_rows: list[dict[str, Any]] = []
    gt_maps = []
    pred_maps = []

    for r in ok:
        gt_map = _findings_map(r["ground_truth_findings"])
        pred_map = _findings_map(r["predicted_findings"])
        gt_maps.append(gt_map)
        pred_maps.append(pred_map)
        # Abnormality is derived only from parsed FDI findings, never from
        # the model's "Total abnormal" count.
        y_true = to_binary_fdi(gt_map)
        y_pred = to_binary_fdi(pred_map)
        m = sample_metrics(y_true, y_pred)
        m["case_id"] = r["case_id"]
        per_sample.append(m)
        r["sample_metrics"] = {k: m[k] for k in ("accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1_score", "tp", "fp", "fn", "tn")}
        r["per_class"] = m["per_class"]
        r["macro"] = m["macro"]
        r["y_true_32"] = y_true
        r["y_pred_32"] = y_pred

        for disease in DISEASES:
            yt = [1 if eval_metrics.tooth_has_disease(gt_map, t, disease) else 0 for t in FDI_TEETH]
            yp = [1 if eval_metrics.tooth_has_disease(pred_map, t, disease) else 0 for t in FDI_TEETH]
            disease_rows[disease].append(sample_metrics(yt, yp))
        yt5, yp5, names5 = eval_metrics.disease_class_ids(gt_map, pred_map, FDI_TEETH, DISEASES)
        exclusive_rows.append(
            eval_metrics.per_class_scores(
                yt5, yp5, labels=list(range(len(names5))), names=names5
            )
        )

    def mean_block(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = ["accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1_score"]
        return {k: float(np.mean([x[k] for x in rows])) for k in keys if k in rows[0]}

    per_disease_patient = {d: mean_block(rows) for d, rows in disease_rows.items() if rows}
    abnormal_patient = mean_block(per_sample)
    class_mean = eval_metrics.mean_per_class(per_sample)
    abnormal_patient["per_class"] = class_mean.get("per_class", {})
    abnormal_patient["macro"] = class_mean.get("macro", {})
    exclusive_patient = eval_metrics.mean_per_class(exclusive_rows)
    if exclusive_patient.get("per_class"):
        exclusive_patient["macro_all_classes"] = exclusive_patient.get("macro", {})
        exclusive_patient["macro"] = eval_metrics.macro_from_classes(
            exclusive_patient["per_class"], DISEASES
        )

    ovr_patient_pc: dict[str, dict[str, float]] = {}
    for d, rows in disease_rows.items():
        if not rows:
            continue
        pos = [row["per_class"]["abnormal"] for row in rows]
        ovr_patient_pc[d] = {
            "precision": float(np.mean([x["precision"] for x in pos])),
            "recall": float(np.mean([x["recall"] for x in pos])),
            "f1_score": float(np.mean([x["f1_score"] for x in pos])),
            "support": float(np.mean([x["support"] for x in pos])),
        }
    ovr_patient = {
        "per_class": ovr_patient_pc,
        "macro": eval_metrics.macro_from_classes(ovr_patient_pc, DISEASES) if ovr_patient_pc else {},
    }

    y_abn_true: list[int] = []
    y_abn_pred: list[int] = []
    y_ex_true: list[int] = []
    y_ex_pred: list[int] = []
    names5 = [NO_FINDING, *DISEASES]
    for r, gt_map, pred_map in zip(ok, gt_maps, pred_maps):
        y_abn_true.extend(r["y_true_32"])
        y_abn_pred.extend(r["y_pred_32"])
        yt5, yp5, names5 = eval_metrics.disease_class_ids(gt_map, pred_map, FDI_TEETH, DISEASES)
        y_ex_true.extend(yt5)
        y_ex_pred.extend(yp5)

    abnormal_pooled = eval_metrics.pooled_metrics(
        y_abn_true, y_abn_pred, labels=(0, 1), names=BINARY_NAMES
    )
    disease_ovr = eval_metrics.pooled_multilabel_diseases(
        gt_maps, pred_maps, FDI_TEETH, DISEASES
    )
    exclusive_pooled = eval_metrics.pooled_metrics(
        y_ex_true,
        y_ex_pred,
        labels=list(range(len(names5))),
        names=names5,
        primary_macro_names=DISEASES,
    )
    yt_patients = [r["y_true_32"] for r in ok]
    yp_patients = [r["y_pred_32"] for r in ok]
    cluster_ids = [eval_metrics.patient_id_for_image(r["case_id"]) for r in ok]
    abn_ci = eval_metrics.bootstrap_binary_macro_f1(
        yt_patients,
        yp_patients,
        names=BINARY_NAMES,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
        cluster_ids=cluster_ids,
    )
    mac4_ci = eval_metrics.bootstrap_macro4_f1(
        gt_maps,
        pred_maps,
        FDI_TEETH,
        DISEASES,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
        cluster_ids=cluster_ids,
    )
    abnormal_pooled.setdefault("macro", {})["f1_ci95"] = abn_ci
    disease_ovr.setdefault("macro", {})["f1_ci95"] = mac4_ci

    return {
        "n_cases": len(ok),
        "n_failed": len(results) - len(ok),
        "aggregation": "pooled_slots",
        "pathology_scoring": "multi_label_one_vs_rest",
        "abnormal_tooth": abnormal_pooled,
        "abnormal_tooth_patient_mean": abnormal_patient,
        "per_disease": disease_ovr.get("per_disease") or {},
        "per_disease_patient_mean": per_disease_patient,
        "disease_multiclass": disease_ovr,
        "disease_multiclass_patient_mean": ovr_patient,
        "disease_exclusive_multiclass": exclusive_pooled,
        "disease_exclusive_multiclass_patient_mean": exclusive_patient,
        "gt_support": {
            name: int((disease_ovr.get("per_class") or {}).get(name, {}).get("support") or 0)
            for name in DISEASES
        },
        "bootstrap": {
            "n_boot": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "unit": "patient",
            "n_patients": abn_ci.get("n_patients"),
            "n_images": abn_ci.get("n_images"),
            "patient_equals_image": abn_ci.get("patient_equals_image"),
            "note": abn_ci.get("note"),
            "abnormal_macro_f1_ci95": abn_ci,
            "pathology_macro4_f1_ci95": mac4_ci,
        },
        "per_sample": per_sample,
        # Unique abnormal teeth (mapping keys). Not diagnosis-label counts.
        "mean_predicted_count": _mean_abnormal_teeth(ok, "predicted_findings"),
        "mean_ground_truth_count": _mean_abnormal_teeth(ok, "ground_truth_findings"),
        "mean_predicted_abnormal_teeth": _mean_abnormal_teeth(ok, "predicted_findings"),
        "mean_ground_truth_abnormal_teeth": _mean_abnormal_teeth(ok, "ground_truth_findings"),
        "mean_predicted_disease_labels": _mean_disease_labels(ok, "predicted_findings"),
        "mean_ground_truth_disease_labels": _mean_disease_labels(ok, "ground_truth_findings"),
        "mean_seconds_per_case": sum(r["seconds"] for r in ok) / len(ok),
    }


def _mean_abnormal_teeth(rows: list[dict[str, Any]], key: str) -> float:
    return sum(len(r.get(key) or {}) for r in rows) / len(rows)


def _mean_disease_labels(rows: list[dict[str, Any]], key: str) -> float:
    return sum(
        sum(len(eval_metrics.labels_on_tooth(v)) for v in (r.get(key) or {}).values())
        for r in rows
    ) / len(rows)


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
                        "raw_model_output": reply,
                        "parser_version": PARSER_VERSION,
                        "predicted_findings": {
                            str(k): eval_metrics.labels_on_tooth(v) for k, v in findings.items()
                        },
                        "ground_truth_findings": {
                            str(k): eval_metrics.labels_on_tooth(v)
                            for k, v in case["gt_findings"].items()
                        },
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
        r["predicted_findings"] = {
            str(k): eval_metrics.labels_on_tooth(v) for k, v in r["predicted_findings"].items()
        }
        r["ground_truth_findings"] = {
            str(k): eval_metrics.labels_on_tooth(v) for k, v in r["ground_truth_findings"].items()
        }

    ab = metrics.get("abnormal_tooth") or {}
    mac = ab.get("macro") or {}
    log(
        f"  pooled abnormal F1={ab.get('f1_score', 0):.3f} "
        f"macroF1={mac.get('f1_score', 0):.3f} "
        f"BalAcc={ab.get('balanced_accuracy', 0):.3f} Acc={ab.get('accuracy', 0):.3f}"
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
            **eval_metrics.flatten_binary_main(ab, "abnormal", NO_FINDING),
            **eval_metrics.flatten_pathology_main(
                m.get("disease_multiclass") or {}, DISEASES
            ),
            "Mean pred abnormal teeth/case": m.get(
                "mean_predicted_abnormal_teeth", m.get("mean_predicted_count")
            ),
            "Mean GT abnormal teeth/case": m.get(
                "mean_ground_truth_abnormal_teeth", m.get("mean_ground_truth_count")
            ),
            "Mean pred disease labels/case": m.get("mean_predicted_disease_labels"),
            "Mean GT disease labels/case": m.get("mean_ground_truth_disease_labels"),
            "Sec/case": m.get("mean_seconds_per_case"),
            "Status": "ok",
        }
        rows.append(row)

    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_supplementary_csv(payload: dict[str, Any], out_path: Path) -> None:
    import csv

    rows = []
    for entry in payload["models"].values():
        m = entry.get("metrics") or {}
        ab = m.get("abnormal_tooth") or {}
        if not ab:
            continue
        row = {
            "Model": entry["display_name"],
            "Strategy": payload.get("strategy"),
            "Cases": m.get("n_cases"),
            **eval_metrics.flatten_binary_report(ab, "abnormal", NO_FINDING),
            **eval_metrics.flatten_pathology_report(
                m.get("disease_multiclass") or {}, DISEASES
            ),
        }
        rows.append(row)
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _model_complete(entry: dict[str, Any]) -> bool:
    return bool((entry.get("metrics") or {}).get("abnormal_tooth"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLIT_CHOICES, default="val")
    ap.add_argument("--cases", type=int, default=0, help="0 = all in split")
    ap.add_argument("--models", nargs="+", default=list(vlm_models.MODEL_REGISTRY))
    ap.add_argument("--strategy", choices=list(PROMPT_REGISTRY), default="zero_shot")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="continue an interrupted run; skip models that already have metrics",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print split, case count, models, and prompt length, then exit",
    )
    args = ap.parse_args()

    prompt = prompt_for_strategy(args.strategy)
    max_new_tokens = 450 if args.strategy == "cot" else 350

    cases = load_split(args.split)
    if args.cases > 0:
        cases = cases[: args.cases]
    held_ids = dataset_inventory.heldout_case_ids() if args.split in {"qed", "train", "all"} else []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.strategy}_{args.split}{'' if args.cases <= 0 else args.cases}"
    stem = f"dentex_{tag}_{stamp}"
    log_path = args.out_dir / f"{stem}.log"
    resume_payload: dict[str, Any] | None = None
    skip_models: list[str] = []

    if args.resume_from:
        resume_payload = json.loads(args.resume_from.read_text())
        stem = args.resume_from.stem
        log_path = args.resume_from.with_suffix(".log")
        skip_models = [k for k, entry in resume_payload.get("models", {}).items() if _model_complete(entry)]
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

    if args.dry_run:
        print(
            f"DENTEX dry-run split={args.split} cases={len(cases)} "
            f"strategy={args.strategy} models={args.models} "
            f"max_new_tokens={max_new_tokens} prompt_chars={len(prompt)} "
            f"heldout_excluded={held_ids}"
        )
        if skip_models:
            print(f"would skip completed models: {skip_models}")
        print("first_case", cases[0]["case_id"] if cases else None)
        print("mean_gt", round(sum(len(c["gt_findings"]) for c in cases) / max(len(cases), 1), 3))
        return 0

    log(
        f"DENTEX split={args.split} cases={len(cases)} models={args.models} "
        f"strategy={args.strategy} aggregation=pooled_slots "
        f"none_label={NO_FINDING!r} disease_macro=4 pathology=multi_label_ovr "
        f"parser={PARSER_VERSION} heldout_excluded={held_ids} "
        f"max_new_tokens={max_new_tokens}"
    )
    if held_ids:
        log(
            "Few-shot uses TEXT from held-out train cases only; those case_ids are "
            "excluded from zero-shot, few-shot, and CoT scoring so all strategies "
            f"share the same images: {held_ids}"
        )
    gt_support = log_gt_disease_support(log, cases)
    if skip_models:
        log(f"resume from {args.resume_from.name}; skipping completed models: {skip_models}")
    parser = DentexParser()
    payload = resume_payload or {
        "timestamp": stamp,
        "dataset": "DENTEX",
        "split": args.split,
        "tag": tag,
        "n_cases": len(cases),
        "case_ids": [c["case_id"] for c in cases],
        "strategy": args.strategy,
        "prompt": prompt,
        "aggregation": "pooled_slots",
        "fdi_teeth": FDI_TEETH,
        "diseases": DISEASES,
        "none_label": NO_FINDING,
        "disease_macro_classes": DISEASES,
        "pathology_scoring": "multi_label_one_vs_rest",
        "heldout_exemplars": held_ids,
        "parser_version": PARSER_VERSION,
        "run_config": {
            "dataset": "DENTEX",
            "split": args.split,
            "strategy": args.strategy,
            "parser_version": PARSER_VERSION,
            "parse_source": "Findings FDI lines only; Total abnormal ignored",
            "pathology_scoring": "multi_label_one_vs_rest",
            "aggregation": "pooled_slots",
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "patient",
            "bootstrap_patient_equals_image": True,
            "bootstrap_note": eval_metrics.BOOTSTRAP_UNIT_NOTE,
            "heldout_exemplars": held_ids,
            "few_shot_uses_exemplar_images": False,
        },
        "gt_support": gt_support,
        "models": {},
    }
    payload["gt_support"] = gt_support

    if not args.models:
        log("nothing left to run; all requested models already complete")
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        write_supplementary_csv(payload, args.out_dir / f"{stem}_supplementary.csv")
        log("DENTEX_COMPARISON_DONE")
        return 0

    out_json = args.out_dir / f"{stem}.json"
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

        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        write_supplementary_csv(payload, args.out_dir / f"{stem}_supplementary.csv")
        log(f"checkpoint -> {out_json.name}")

    support = payload.get("gt_support") or {}
    ta = support.get("table_a") or {}
    tb = support.get("table_b_gt_positive_slots") or {d: support.get(d, 0) for d in DISEASES}
    log("=== TABLE A binary support ===")
    if ta:
        log(
            f"  Abnormal n={ta.get('abnormal')}  No annotated finding n={ta.get('no_annotated_finding')}  "
            f"slots={ta.get('n_slots')}  "
            f"{'OK' if ta.get('consistent') else 'MISMATCH'}: "
            f"{ta.get('abnormal')} + {ta.get('no_annotated_finding')} = {ta.get('sum')}"
        )
    log("=== TABLE B GT-positive slots n (OvR; not unique abnormal teeth) ===")
    log(f"{'disease':<22}{'n':>6}")
    for name in DISEASES:
        log(f"{name:<22}{tb.get(name, 0):>6}")
    log(f"{'diagnosis-positive sum':<22}{support.get('diagnosis_positive_slots_sum', sum(tb.values())):>6}")
    log("=== TABLE A abnormality (per-class F1 + Macro F1 95% CI + Acc + BalAcc) ===")
    log(f"{'model':<22}{'abnF1':>8}{'noFndF1':>8}{'MacroF1':>8}{'Acc':>8}{'BalAcc':>8}{'CI95':>19}")
    for key, entry in payload["models"].items():
        ab = (entry.get("metrics") or {}).get("abnormal_tooth")
        if not ab:
            log(f"{key:<22}{'ERROR':>8}")
            continue
        row = eval_metrics.flatten_binary_main(ab, "abnormal", NO_FINDING)
        no_key = eval_metrics.csv_class_key(NO_FINDING) + "_F1"
        lo, hi = row.get("Macro_F1_CI95_low"), row.get("Macro_F1_CI95_high")
        ci_s = f"[{lo:.3f},{hi:.3f}]" if lo is not None and hi is not None else "n/a"
        log(
            f"{entry['display_name']:<22}"
            f"{(row['abnormal_F1'] or 0):>8.3f}"
            f"{(row.get(no_key) or 0):>8.3f}"
            f"{(row['Macro_F1'] or 0):>8.3f}"
            f"{(row['Accuracy'] or 0):>8.3f}"
            f"{(row['Balanced_Accuracy'] or 0):>8.3f}"
            f"{ci_s:>19}"
        )
    log("=== TABLE B pathology OvR (main paper: F1 only; P/R in supplementary) ===")
    log(f"{'model':<22}{'ImpF1':>8}{'CarF1':>8}{'DeepF1':>8}{'PerF1':>8}{'Mac4F1':>8}{'CI95':>19}")

    def _fmt(val: Any) -> str:
        if val in (None, "N/A"):
            return f"{'N/A':>8}"
        return f"{float(val):>8.3f}"

    for key, entry in payload["models"].items():
        if not (entry.get("metrics") or {}).get("abnormal_tooth"):
            log(f"{key:<22}{'ERROR':>8}")
            continue
        path = eval_metrics.flatten_pathology_main(
            (entry.get("metrics") or {}).get("disease_multiclass") or {}, DISEASES
        )
        lo, hi = path.get("Macro4_F1_CI95_low"), path.get("Macro4_F1_CI95_high")
        ci = f"[{lo:.3f},{hi:.3f}]" if lo is not None and hi is not None else "n/a"
        log(
            f"{entry.get('display_name', key):<22}"
            f"{_fmt(path['Impacted_F1'])}"
            f"{_fmt(path['Caries_F1'])}"
            f"{_fmt(path['Deep_Caries_F1'])}"
            f"{_fmt(path['Periapical_Lesion_F1'])}"
            f"{_fmt(path['Macro4_F1'])}"
            f"{ci:>19}"
        )
    log("DENTEX_COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
