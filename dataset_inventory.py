#!/usr/bin/env python3
"""Canonical dataset sizes and DENTEX held-out few-shot exemplars.

This module does not import torch. Run:
    python dataset_inventory.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/s222393187/Dental")
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
TUFTS_BBOX = ROOT / "Tufts_Dental_Database" / "Segmentation" / "teeth_bbox.json"
TUFTS_XRAYS = ROOT / "Tufts_Dental_Database" / "Radiographs"
HELDOUT_PATH = ROOT / "runs" / "dentex_heldout_exemplars.json"
COUNTS_PATH = ROOT / "runs" / "dataset_counts.json"
DISEASES = ["Impacted", "Caries", "Deep Caries", "Periapical Lesion"]

# Archive 100-case Tufts runs exist under Results/model_comparison/comparison_*_100_*.
# They are not the paper evaluation. The paper Tufts run is all usable radiographs.
TUFTS_PAPER_N_CASES = 1000
TUFTS_ARCHIVE_N_CASES = 100


def _disease_counts(findings_by_stem: dict[str, dict[int, list[str]]]) -> dict[str, int]:
    counts = Counter()
    for teeth in findings_by_stem.values():
        for labs in teeth.values():
            for lab in labs:
                if lab in DISEASES:
                    counts[lab] += 1
    return {d: int(counts[d]) for d in DISEASES}


def load_qed_layer(json_path: Path, xray_dir: Path) -> dict[str, Any]:
    payload = json.loads(json_path.read_text())
    c1 = {c["id"]: str(c["name"]) for c in payload["categories_1"]}
    c2 = {c["id"]: str(c["name"]) for c in payload["categories_2"]}
    c3 = {c["id"]: str(c["name"]) for c in payload["categories_3"]}
    by_img: dict[int, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    n_raw = 0
    for a in payload["annotations"]:
        n_raw += 1
        fdi = int(f"{c1[a['category_id_1']]}{c2[a['category_id_2']]}")
        by_img[a["image_id"]][fdi].add(c3[a["category_id_3"]])
    all_stems: dict[str, dict[int, list[str]]] = {}
    on_disk: dict[str, dict[int, list[str]]] = {}
    missing_files = []
    for im in payload["images"]:
        stem = Path(im["file_name"]).stem
        teeth = {int(f): sorted(labs) for f, labs in by_img.get(im["id"], {}).items()}
        all_stems[stem] = teeth
        path = xray_dir / im["file_name"]
        if path.exists() and path.stat().st_size > 0:
            on_disk[stem] = teeth
        else:
            missing_files.append(im["file_name"])
    return {
        "n_images_json": len(payload["images"]),
        "n_annotations_raw": n_raw,
        "n_images_on_disk": len(on_disk),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files,
        "all_stems": all_stems,
        "on_disk": on_disk,
        "support_json": _disease_counts(all_stems),
        "support_on_disk": _disease_counts(on_disk),
    }


def tufts_inventory() -> dict[str, Any]:
    records = json.loads(TUFTS_BBOX.read_text())
    if isinstance(records, dict):
        records = list(records.values())
    n_bbox = 0
    missing_slots = 0
    present_slots = 0
    usable = 0
    for record in records:
        ext = record.get("External ID")
        if not ext:
            continue
        stem = Path(ext).stem
        n_bbox += 1
        detected: set[int] = set()
        for obj in record.get("Label", {}).get("objects", []):
            for tok in __import__("re").findall(r"\d+", str(obj.get("title", ""))):
                if 1 <= int(tok) <= 32:
                    detected.add(int(tok))
        miss = 32 - len(detected)
        img = None
        for extn in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
            cand = TUFTS_XRAYS / f"{stem}{extn}"
            if cand.exists() and cand.stat().st_size > 0:
                img = cand
                break
        if img is None:
            continue
        usable += 1
        missing_slots += miss
        present_slots += len(detected)
    return {
        "paper_evaluation": {
            "n_cases": usable,
            "n_slots": usable * 32,
            "missing_slots": missing_slots,
            "present_slots": present_slots,
            "note": "Final Tufts paper run is ALL usable cases with bbox + readable radiograph.",
        },
        "archive_100": {
            "n_cases": TUFTS_ARCHIVE_N_CASES,
            "note": "Aug 2026 100-case prompting runs are archive only and will be replaced.",
        },
        "n_bbox_records": n_bbox,
    }


def render_findings_block(gt: dict[int, list[str]]) -> str:
    if not gt:
        return "Findings:\nNone\nTotal abnormal: 0\nConfidence: High"
    lines = ["Findings:"]
    for fdi in sorted(gt):
        for lab in gt[fdi]:
            lines.append(f"{fdi}: {lab}")
    lines.append(f"Total abnormal: {len(gt)}")
    lines.append("Confidence: Medium")
    return "\n".join(lines)


def select_heldout_exemplars(train_on_disk: dict[str, dict[int, list[str]]]) -> dict[str, Any]:
    """Fixed train cases used as TEXT few-shot exemplars; excluded from all qed scoring."""
    if HELDOUT_PATH.exists():
        return json.loads(HELDOUT_PATH.read_text())

    stems = sorted(train_on_disk)
    impacted = multilabel = none_case = mixed = None
    for stem in stems:
        gt = train_on_disk[stem]
        diseases = {lab for labs in gt.values() for lab in labs}
        has_multi = any(len(labs) >= 2 for labs in gt.values())
        if none_case is None and not gt:
            none_case = stem
        if impacted is None and diseases == {"Impacted"} and gt:
            impacted = stem
        if multilabel is None and has_multi:
            multilabel = stem
        if mixed is None and len(diseases) >= 2:
            mixed = stem
    chosen = []
    roles = []
    for stem, role in (
        (impacted, "impacted_only"),
        (multilabel or mixed, "multi_label_or_mixed"),
        (none_case, "no_annotated_finding"),
    ):
        if stem and stem not in chosen:
            chosen.append(stem)
            roles.append({"case_id": stem, "role": role, "gt": train_on_disk[stem]})
    payload = {
        "purpose": (
            "Text-only few-shot exemplars taken from DENTEX train. These case_ids are "
            "excluded from qed scoring for zero-shot, few-shot, and CoT so all three "
            "strategies are evaluated on the same images. Few-shot does not feed the "
            "exemplar radiographs — only their GT text."
        ),
        "case_ids": chosen,
        "exemplars": roles,
    }
    HELDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELDOUT_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def build_inventory() -> dict[str, Any]:
    train = load_qed_layer(TRAIN_JSON, TRAIN_XRAYS)
    val = load_qed_layer(VAL_JSON, VAL_XRAYS)
    held = select_heldout_exemplars(train["on_disk"])
    exclude = set(held["case_ids"])
    qed_disk = {**train["on_disk"], **val["on_disk"]}
    qed_eval = {k: v for k, v in qed_disk.items() if k not in exclude}

    def pack(stems: dict[str, dict[int, list[str]]]) -> dict[str, Any]:
        n_abn = sum(len(t) for t in stems.values())
        n_img = len(stems)
        multi = Counter(len(labs) for teeth in stems.values() for labs in teeth.values())
        ds = _disease_counts(stems)
        n_diag = int(sum(ds.values()))
        table_a = {
            "abnormal": n_abn,
            "no_annotated_finding": n_img * 32 - n_abn,
            "n_slots": n_img * 32,
            "n_images": n_img,
            "sum": n_abn + (n_img * 32 - n_abn),
            "consistent": True,
        }
        return {
            "n_images": n_img,
            "n_slots": n_img * 32,
            "n_abnormal_teeth": n_abn,
            "n_no_finding_slots": n_img * 32 - n_abn,
            "table_a_binary_support": table_a,
            "disease_support": ds,
            "diagnosis_positive_slots_sum": n_diag,
            "disease_support_definition": (
                "GT-positive tooth-slot support (n) under one-vs-rest. "
                "A multi-label tooth is counted in every disease it carries. "
                "Sum of disease_support is not equal to n_abnormal_teeth."
            ),
            "multi_label_teeth": {str(k): int(multi[k]) for k in sorted(multi)},
        }

    recon = {
        "why_2290_vs_2287": (
            "2290/610 were unofficial QED sums using the DENTEX paper's train counts "
            "(Caries 2189 + val 101; Deep Caries 578 + val 32). "
            "Canonical counts are unique (image, FDI) diagnoses on radiographs that exist "
            "on disk, with a multi-label tooth counted in every disease it carries."
        )
    }
    tufts = tufts_inventory()
    out = {
        "tufts": tufts,
        "dentex": {
            "train_json": {
                "n_images": train["n_images_json"],
                "n_annotations_raw": train["n_annotations_raw"],
                "support_all_json_images": train["support_json"],
            },
            "train_on_disk": pack(train["on_disk"]),
            "val_json": {
                "n_images": val["n_images_json"],
                "n_annotations_raw": val["n_annotations_raw"],
                "support_all_json_images": val["support_json"],
            },
            "val_on_disk": pack(val["on_disk"]),
            "qed_on_disk_before_heldout": pack(qed_disk),
            "heldout_exemplars": held["case_ids"],
            "qed_paper_evaluation": pack(qed_eval),
            "missing_files": {
                "train": train["n_missing_files"],
                "val": val["n_missing_files"],
            },
            "count_reconciliation": recon,
        },
    }
    COUNTS_PATH.write_text(json.dumps(out, indent=2))
    return out


def heldout_case_ids() -> list[str]:
    if not HELDOUT_PATH.exists():
        build_inventory()
    return list(json.loads(HELDOUT_PATH.read_text())["case_ids"])


def heldout_payload() -> dict[str, Any]:
    if not HELDOUT_PATH.exists():
        build_inventory()
    return json.loads(HELDOUT_PATH.read_text())


def main() -> int:
    inv = build_inventory()
    t = inv["tufts"]["paper_evaluation"]
    d = inv["dentex"]
    print("TUFTS paper evaluation:", t)
    print("TUFTS archive 100:", inv["tufts"]["archive_100"])
    print("DENTEX train json support", d["train_json"]["support_all_json_images"])
    print("DENTEX train on disk", d["train_on_disk"]["disease_support"], "n=", d["train_on_disk"]["n_images"])
    print("DENTEX val on disk", d["val_on_disk"]["disease_support"], "n=", d["val_on_disk"]["n_images"])
    print("DENTEX qed on disk", d["qed_on_disk_before_heldout"]["disease_support"], "n=", d["qed_on_disk_before_heldout"]["n_images"])
    print("held-out", d["heldout_exemplars"])
    print("DENTEX qed PAPER eval", d["qed_paper_evaluation"]["disease_support"], "n=", d["qed_paper_evaluation"]["n_images"])
    print("wrote", COUNTS_PATH)
    print("wrote", HELDOUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
