#!/usr/bin/env python3
"""Pooled (dataset-level) per-class metrics for binary tooth tasks and DENTEX diseases.

Primary protocol: concatenate all valid tooth slots across patients, accumulate
TP/FP/TN/FN per class, then compute P/R/F1.

DENTEX pathology is multi-label. One tooth may carry more than one diagnosis,
so each disease is scored independently (one-vs-rest) rather than as a 5-way
exclusive class. Macro-4 is the unweighted mean of Impacted, Caries, Deep
Caries, and Periapical Lesion. "No annotated finding" belongs to the binary
abnormality task (Table A), not the four pathology classifiers (Table B).

If a future split has GT support 0 for a disease, that class is reported as
N/A and omitted from that split's supported-class macro. Full val and qed
have all four diseases, so Macro-4 uses all four with equal weight.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Patient-level CIs for the paper. Cheap CPU on saved predictions; not a GPU rerun.
PAPER_BOOTSTRAP_N = 5000
PAPER_BOOTSTRAP_SEED = 20260820


def per_class_scores(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    names: Sequence[str],
) -> dict[str, Any]:
    """Precision / recall / F1 for each class, plus unweighted macro averages."""
    label_list = list(labels)
    name_list = list(names)
    prec = precision_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=label_list, average=None, zero_division=0)
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(name_list):
        support = int(sum(int(v) == label_list[i] for v in y_true))
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1_score": float(f1[i]),
            "support": support,
        }
    return {
        "per_class": per_class,
        "macro": {
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
            "f1_score": float(np.mean(f1)),
        },
    }


NO_ANNOTATED_FINDING = "No annotated finding"
MACRO_KEYS = (
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def csv_class_key(name: str) -> str:
    return name.replace(" ", "_")


def macro_from_classes(
    per_class: dict[str, dict[str, Any]],
    names: Sequence[str],
    *,
    skip_zero_support: bool = True,
) -> dict[str, Any]:
    """Unweighted mean of selected classes.

    Classes with GT support 0 are undefined (not model failures). They are
    omitted from the macro when skip_zero_support is True.
    """
    used, skipped = [], []
    for name in names:
        if name not in per_class:
            skipped.append(name)
            continue
        support = int(per_class[name].get("support") or 0)
        if skip_zero_support and support == 0:
            skipped.append(name)
            continue
        used.append(name)
    selected = [per_class[name] for name in used]
    out: dict[str, Any] = {"classes_used": used, "classes_undefined": skipped}
    if not selected:
        for key in ("precision", "recall", "f1_score", "specificity", "accuracy", "balanced_accuracy"):
            out[key] = None
        return out
    for key in MACRO_KEYS:
        vals = [row[key] for row in selected if key in row]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def pooled_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    names: Sequence[str],
    primary_macro_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Dataset-level metrics from concatenated tooth slots.

    For each class k (one-vs-rest on the pooled vectors):
      P, R, F1, specificity, accuracy, balanced accuracy, and TP/FP/FN/TN.
    `macro` is the unweighted mean across `primary_macro_names` if given,
    otherwise across all classes. `macro_all_classes` always covers every class.
    Top-level `accuracy` is overall correctness. `balanced_accuracy` is the
    primary-macro recall (mean of the classes used for `macro`).
    """
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    label_list = list(labels)
    name_list = list(names)
    n = int(yt.size)
    prec = precision_score(yt, yp, labels=label_list, average=None, zero_division=0)
    rec = recall_score(yt, yp, labels=label_list, average=None, zero_division=0)
    f1 = f1_score(yt, yp, labels=label_list, average=None, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=label_list)

    per_class: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(name_list):
        k = label_list[i]
        true_k = yt == k
        pred_k = yp == k
        tp = int(np.sum(true_k & pred_k))
        fp = int(np.sum(~true_k & pred_k))
        fn = int(np.sum(true_k & ~pred_k))
        tn = int(np.sum(~true_k & ~pred_k))
        spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
        acc_k = float((tp + tn) / n) if n else 0.0
        rec_k = float(rec[i])
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": rec_k,
            "f1_score": float(f1[i]),
            "specificity": spec,
            "accuracy": acc_k,
            "balanced_accuracy": float(0.5 * (rec_k + spec)),
            "support": int(np.sum(true_k)),
            "undefined": bool(np.sum(true_k) == 0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    macro_keys = MACRO_KEYS
    macro_all = {key: float(np.mean([per_class[name][key] for name in name_list])) for key in macro_keys}
    if primary_macro_names:
        macro = macro_from_classes(per_class, primary_macro_names)
    else:
        macro = macro_all
    overall_acc = float(accuracy_score(yt, yp)) if n else 0.0
    bal = macro.get("recall")
    overall_bal = float(bal) if bal is not None else float(np.mean(rec) if len(rec) else 0.0)
    out: dict[str, Any] = {
        "aggregation": "pooled_slots",
        "n_slots": n,
        "per_class": per_class,
        "macro": macro,
        "macro_all_classes": macro_all,
        "accuracy": overall_acc,
        "balanced_accuracy": overall_bal,
        "confusion_matrix": cm.tolist(),
    }
    if len(name_list) == 2:
        pos = per_class[name_list[1]]
        out["precision"] = pos["precision"]
        out["recall"] = pos["recall"]
        out["f1_score"] = pos["f1_score"]
        out["specificity"] = pos["specificity"]
    else:
        out["precision"] = macro.get("precision", 0.0)
        out["recall"] = macro.get("recall", 0.0)
        out["f1_score"] = macro.get("f1_score", 0.0)
        out["specificity"] = macro.get("specificity", 0.0)
    return out


def _class_scores(block: dict[str, Any], class_name: str) -> dict[str, Any]:
    return dict((block.get("per_class") or {}).get(class_name) or {})


def _prefixed_class(scores: dict[str, Any], prefix: str) -> dict[str, Any]:
    key = csv_class_key(prefix)
    return {
        f"{key}_P": scores.get("precision"),
        f"{key}_R": scores.get("recall"),
        f"{key}_F1": scores.get("f1_score"),
    }


def flatten_binary_report(block: dict[str, Any], pos_name: str, neg_name: str) -> dict[str, Any]:
    """Paper binary table: per-class P/R/F1, macro P/R/F1, overall Acc and BalAcc.

    Do not report per-class Acc/Spec/BalAcc or macro Acc/Spec: Acc is already
    overall, BalAcc is already (sens+spec)/2, and Spec of one class is Recall
    of the other.
    """
    macro = block.get("macro") or {}
    return {
        **_prefixed_class(_class_scores(block, pos_name), pos_name),
        **_prefixed_class(_class_scores(block, neg_name), neg_name),
        "Macro_P": macro.get("precision"),
        "Macro_F1": macro.get("f1_score"),
        "Macro_F1_CI95_low": (macro.get("f1_ci95") or {}).get("low") if isinstance(macro.get("f1_ci95"), dict) else None,
        "Macro_F1_CI95_high": (macro.get("f1_ci95") or {}).get("high") if isinstance(macro.get("f1_ci95"), dict) else None,
        "Accuracy": block.get("accuracy"),
        "Balanced_Accuracy": block.get("balanced_accuracy"),
    }


def flatten_binary_main(block: dict[str, Any], pos_name: str, neg_name: str) -> dict[str, Any]:
    """Main-paper Table A: per-class F1, Macro F1 (95% CI), Accuracy, Balanced Accuracy.

    Precision/Recall stay in flatten_binary_report for the supplementary CSV.
    """
    macro = block.get("macro") or {}
    ci = (macro.get("f1_ci95") or {}) if isinstance(macro.get("f1_ci95"), dict) else {}
    return {
        f"{csv_class_key(pos_name)}_F1": _class_scores(block, pos_name).get("f1_score"),
        f"{csv_class_key(neg_name)}_F1": _class_scores(block, neg_name).get("f1_score"),
        "Macro_F1": macro.get("f1_score"),
        "Macro_F1_CI95_low": ci.get("low"),
        "Macro_F1_CI95_high": ci.get("high"),
        "Accuracy": block.get("accuracy"),
        "Balanced_Accuracy": block.get("balanced_accuracy"),
    }


def report_prf(scores: dict[str, Any]) -> tuple[Any, Any, Any]:
    """P/R/F1, or N/A when the class has no GT support."""
    if scores.get("undefined") or int(scores.get("support") or 0) == 0:
        return "N/A", "N/A", "N/A"
    return scores.get("precision"), scores.get("recall"), scores.get("f1_score")


def flatten_pathology_report(
    block: dict[str, Any],
    disease_names: Sequence[str],
) -> dict[str, Any]:
    """Supplementary Table B: four OvR P/R/F1 plus Macro-4 P/R/F1."""
    row: dict[str, Any] = {}
    per_class = block.get("per_class") or {}
    for name in disease_names:
        scores = per_class.get(name) or {}
        key = csv_class_key(name)
        p, r, f1 = report_prf(scores)
        row[f"{key}_P"] = p
        row[f"{key}_R"] = r
        row[f"{key}_F1"] = f1
    macro = block.get("macro") or {}
    row["Macro4_P"] = macro.get("precision")
    row["Macro4_R"] = macro.get("recall")
    row["Macro4_F1"] = macro.get("f1_score")
    ci = (macro.get("f1_ci95") or {}) if isinstance(macro.get("f1_ci95"), dict) else {}
    row["Macro4_F1_CI95_low"] = ci.get("low")
    row["Macro4_F1_CI95_high"] = ci.get("high")
    return row


def flatten_pathology_main(
    block: dict[str, Any],
    disease_names: Sequence[str],
) -> dict[str, Any]:
    """Main-paper Table B: per-disease F1 plus Macro-4 F1 only."""
    row: dict[str, Any] = {}
    per_class = block.get("per_class") or {}
    for name in disease_names:
        scores = per_class.get(name) or {}
        _p, _r, f1 = report_prf(scores)
        row[f"{csv_class_key(name)}_F1"] = f1
    macro = block.get("macro") or {}
    ci = (macro.get("f1_ci95") or {}) if isinstance(macro.get("f1_ci95"), dict) else {}
    row["Macro4_F1"] = macro.get("f1_score")
    row["Macro4_F1_CI95_low"] = ci.get("low")
    row["Macro4_F1_CI95_high"] = ci.get("high")
    return row


def flatten_multiclass_report(
    block: dict[str, Any],
    *,
    primary_names: Sequence[str] | None = None,
    optional_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per-class P/R/F1. Macro columns use `block['macro']` (primary classes)."""
    row: dict[str, Any] = {}
    per_class = block.get("per_class") or {}
    names = list(primary_names) if primary_names is not None else list(per_class)
    for name in names:
        scores = per_class.get(name) or {}
        key = csv_class_key(name)
        p, r, f1 = report_prf(scores)
        row[f"{key}_P"] = p
        row[f"{key}_R"] = r
        row[f"{key}_F1"] = f1
        row[f"{key}_support"] = int(scores.get("support") or 0)
        row[f"{key}_undefined"] = bool(scores.get("undefined") or (scores.get("support") == 0))
    for name in optional_names or ():
        scores = per_class.get(name) or {}
        key = csv_class_key(name)
        row[f"{key}_P"] = scores.get("precision")
        row[f"{key}_R"] = scores.get("recall")
        row[f"{key}_F1"] = scores.get("f1_score")
        row[f"{key}_support"] = scores.get("support")
    macro = block.get("macro") or {}
    row["Macro_P"] = macro.get("precision")
    row["Macro_R"] = macro.get("recall")
    row["Macro_F1"] = macro.get("f1_score")
    row["Macro_classes_used"] = ",".join(macro.get("classes_used") or [])
    row["Macro_classes_undefined"] = ",".join(macro.get("classes_undefined") or [])
    row["Balanced_Accuracy"] = block.get("balanced_accuracy", macro.get("recall"))
    return row


def mean_per_class(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average per-class / macro blocks across patients."""
    if not rows:
        return {}
    names = list(rows[0]["per_class"])
    out: dict[str, Any] = {"per_class": {}, "macro": {}}
    for name in names:
        out["per_class"][name] = {
            key: float(np.mean([row["per_class"][name][key] for row in rows]))
            for key in ("precision", "recall", "f1_score", "support")
        }
    out["macro"] = {
        key: float(np.mean([row["macro"][key] for row in rows]))
        for key in ("precision", "recall", "f1_score")
    }
    return out


def gt_disease_support(findings_list: Sequence[dict], diseases: Sequence[str]) -> dict[str, int]:
    """GT-positive tooth-slot support (n) per diagnosis (one-vs-rest).

    A multi-label tooth is counted in every disease it carries, so the four
    counts do not sum to the number of unique abnormal teeth.
    """
    counts = {name: 0 for name in diseases}
    for mapping in findings_list:
        for raw in mapping.values():
            for lab in labels_on_tooth(raw):
                if lab in counts:
                    counts[lab] += 1
    return counts


def table_a_binary_support(n_images: int, n_abnormal_teeth: int, *, slots_per_image: int = 32) -> dict[str, Any]:
    """DENTEX/Tufts Table A support: unique abnormal teeth vs no-finding slots.

    These two counts partition the pooled 32-slot grid, so they must sum to
    n_images × 32.
    """
    n_slots = int(n_images) * int(slots_per_image)
    n_none = n_slots - int(n_abnormal_teeth)
    return {
        "n_images": int(n_images),
        "n_slots": n_slots,
        "abnormal": int(n_abnormal_teeth),
        "no_annotated_finding": n_none,
        "sum": int(n_abnormal_teeth) + n_none,
        "consistent": (int(n_abnormal_teeth) + n_none) == n_slots,
    }


def patient_id_for_image(image_id: str) -> str:
    """Independent bootstrap sampling unit.

    Tufts External IDs and DENTEX file stems are unique, so patient = image.
    If a future dataset maps several radiographs to one person, return that
    shared patient ID so all of their images stay together in each replicate.
    """
    return str(image_id)


def labels_on_tooth(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    return [str(x) for x in raw]


def tooth_labels(mapping: dict, tooth: int) -> list[str]:
    return labels_on_tooth(mapping.get(tooth, mapping.get(str(tooth))))


def tooth_has_disease(mapping: dict, tooth: int, disease: str) -> bool:
    return disease in tooth_labels(mapping, tooth)


def pooled_multilabel_diseases(
    gt_maps: Sequence[dict],
    pred_maps: Sequence[dict],
    fdi_teeth: Sequence[int],
    diseases: Sequence[str],
) -> dict[str, Any]:
    """Independent pooled one-vs-rest for each DENTEX diagnosis, then Macro-4.

    A tooth with two findings is a positive for both diseases. Overall accuracy
    is not defined for this multi-label setup and is left unset.
    """
    teeth = list(fdi_teeth)
    disease_list = list(diseases)
    per_class: dict[str, dict[str, Any]] = {}
    per_disease: dict[str, dict[str, Any]] = {}
    for disease in disease_list:
        yt: list[int] = []
        yp: list[int] = []
        for gt_map, pred_map in zip(gt_maps, pred_maps):
            yt.extend(1 if tooth_has_disease(gt_map, t, disease) else 0 for t in teeth)
            yp.extend(1 if tooth_has_disease(pred_map, t, disease) else 0 for t in teeth)
        block = pooled_metrics(yt, yp, labels=(0, 1), names=("other", disease))
        per_disease[disease] = block
        per_class[disease] = dict((block.get("per_class") or {}).get(disease) or {})

    yt_nf: list[int] = []
    yp_nf: list[int] = []
    for gt_map, pred_map in zip(gt_maps, pred_maps):
        yt_nf.extend(0 if tooth_labels(gt_map, t) else 1 for t in teeth)
        yp_nf.extend(0 if tooth_labels(pred_map, t) else 1 for t in teeth)
    nf_block = pooled_metrics(
        yt_nf, yp_nf, labels=(0, 1), names=("has_finding", NO_ANNOTATED_FINDING)
    )
    per_class[NO_ANNOTATED_FINDING] = dict(
        (nf_block.get("per_class") or {}).get(NO_ANNOTATED_FINDING) or {}
    )

    macro = macro_from_classes(per_class, disease_list)
    n_slots = int(len(teeth) * len(gt_maps))
    return {
        "aggregation": "pooled_slots_one_vs_rest",
        "n_slots": n_slots,
        "per_class": per_class,
        "macro": macro,
        "per_disease": per_disease,
        "no_finding_binary": nf_block,
        "accuracy": None,
        "balanced_accuracy": macro.get("recall"),
        "precision": macro.get("precision"),
        "recall": macro.get("recall"),
        "f1_score": macro.get("f1_score"),
        "specificity": macro.get("specificity"),
    }


def disease_class_ids(
    gt_map: dict,
    pred_map: dict,
    fdi_teeth: Sequence[int],
    diseases: Sequence[str],
) -> tuple[list[int], list[int], list[str]]:
    """Exclusive 5-way ids (diagnostic only). Multi-label teeth keep the first disease in taxonomy order."""
    names = [NO_ANNOTATED_FINDING, *list(diseases)]
    index = {name: i for i, name in enumerate(names)}

    def to_id(mapping: dict, tooth: int) -> int:
        labs = tooth_labels(mapping, tooth)
        for name in diseases:
            if name in labs:
                return index[name]
        if len(labs) == 1 and labs[0] in index:
            return index[labs[0]]
        return 0

    y_true = [to_id(gt_map, tooth) for tooth in fdi_teeth]
    y_pred = [to_id(pred_map, tooth) for tooth in fdi_teeth]
    return y_true, y_pred, names


def _f1_from_tp_fp_fn(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def _binary_class_f1(yt: np.ndarray, yp: np.ndarray, k: int) -> float:
    tp = int(np.sum((yt == k) & (yp == k)))
    fp = int(np.sum((yt != k) & (yp == k)))
    fn = int(np.sum((yt == k) & (yp != k)))
    return _f1_from_tp_fp_fn(tp, fp, fn)


def _binary_macro_f1_fast(yt: np.ndarray, yp: np.ndarray) -> float:
    """Unweighted mean of class-0 and class-1 F1 on pooled slots."""
    return 0.5 * (_binary_class_f1(yt, yp, 0) + _binary_class_f1(yt, yp, 1))


def _positive_f1_fast(yt: np.ndarray, yp: np.ndarray) -> float | None:
    """One-vs-rest positive-class F1. None if the replicate has no GT positives."""
    if int(np.sum(yt)) == 0:
        return None
    return _binary_class_f1(yt, yp, 1)


def _stack_patient_slots(rows: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray([np.asarray(r, dtype=np.int8) for r in rows])


def _ovr_patient_stack(
    maps: Sequence[dict],
    fdi_teeth: Sequence[int],
    diseases: Sequence[str],
) -> np.ndarray:
    """Shape (n_patients, n_diseases, n_slots) with 1 = GT/pred positive for that disease."""
    teeth = list(fdi_teeth)
    out = np.zeros((len(maps), len(diseases), len(teeth)), dtype=np.int8)
    for i, mapping in enumerate(maps):
        for j, disease in enumerate(diseases):
            out[i, j] = np.fromiter(
                (1 if tooth_has_disease(mapping, t, disease) else 0 for t in teeth),
                dtype=np.int8,
                count=len(teeth),
            )
    return out


def _macro4_f1_from_stacks(yt: np.ndarray, yp: np.ndarray) -> float | None:
    """yt/yp shape (n_boot_patients, n_diseases, n_slots). Skip diseases with support 0."""
    f1s = []
    for j in range(yt.shape[1]):
        val = _positive_f1_fast(yt[:, j].ravel(), yp[:, j].ravel())
        if val is not None:
            f1s.append(val)
    if not f1s:
        return None
    return float(np.mean(f1s))


BOOTSTRAP_UNIT_NOTE = (
    "Independent sampling unit is the patient. All radiographs from the same "
    "patient are kept together in each replicate. On Tufts and DENTEX QED each "
    "image ID is unique and no multi-image patient identifier exists, so "
    "patient = image."
)


def _image_clusters(n: int, cluster_ids: Sequence[str] | None) -> list[list[int]]:
    if not cluster_ids:
        return [[i] for i in range(n)]
    if len(cluster_ids) != n:
        raise ValueError(f"cluster_ids length {len(cluster_ids)} != n_images {n}")
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for i, cid in enumerate(cluster_ids):
        key = str(cid)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [groups[k] for k in order]


def _bootstrap_unit_meta(n_images: int, clusters: list[list[int]]) -> dict[str, Any]:
    n_patients = len(clusters)
    sizes = [len(c) for c in clusters]
    return {
        "unit": "patient",
        "n_patients": n_patients,
        "n_images": n_images,
        "max_images_per_patient": max(sizes) if sizes else 0,
        "patient_equals_image": n_patients == n_images,
        "note": BOOTSTRAP_UNIT_NOTE,
    }


def _ci95(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"low": None, "high": None, "n_boot": 0, "unit": "patient"}
    return {
        "low": float(np.quantile(arr, 0.025)),
        "high": float(np.quantile(arr, 0.975)),
        "mean": float(np.mean(arr)),
        "n_boot": int(arr.size),
        "unit": "patient",
    }


def _resample_cluster_rows(clusters: list[list[int]], rng: np.random.Generator) -> np.ndarray:
    n_c = len(clusters)
    chosen = rng.integers(0, n_c, size=n_c)
    return np.concatenate([np.asarray(clusters[int(c)], dtype=int) for c in chosen])


def bootstrap_binary_macro_f1(
    y_true_patients: Sequence[Sequence[int]],
    y_pred_patients: Sequence[Sequence[int]],
    *,
    names: Sequence[str],
    n_boot: int = PAPER_BOOTSTRAP_N,
    seed: int = PAPER_BOOTSTRAP_SEED,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Patient-level bootstrap of pooled binary Macro F1.

    Resample patients with replacement. All images (and their 32 slots) from a
    sampled patient are kept together. Do not resample tooth slots.
    """
    n = len(y_true_patients)
    if n == 0:
        return {**_ci95([]), **_bootstrap_unit_meta(0, [])}
    clusters = _image_clusters(n, cluster_ids)
    yt_p = _stack_patient_slots(y_true_patients)
    yp_p = _stack_patient_slots(y_pred_patients)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample_cluster_rows(clusters, rng)
        stats.append(_binary_macro_f1_fast(yt_p[idx].ravel(), yp_p[idx].ravel()))
    return {**_ci95(stats), **_bootstrap_unit_meta(n, clusters)}


def bootstrap_macro4_f1(
    gt_maps: Sequence[dict],
    pred_maps: Sequence[dict],
    fdi_teeth: Sequence[int],
    diseases: Sequence[str],
    *,
    n_boot: int = PAPER_BOOTSTRAP_N,
    seed: int = PAPER_BOOTSTRAP_SEED,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Patient-level bootstrap of DENTEX pathology Macro-4 F1."""
    n = len(gt_maps)
    if n == 0:
        return {**_ci95([]), **_bootstrap_unit_meta(0, [])}
    clusters = _image_clusters(n, cluster_ids)
    yt_s = _ovr_patient_stack(gt_maps, fdi_teeth, diseases)
    yp_s = _ovr_patient_stack(pred_maps, fdi_teeth, diseases)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample_cluster_rows(clusters, rng)
        val = _macro4_f1_from_stacks(yt_s[idx], yp_s[idx])
        if val is not None:
            stats.append(val)
    return {**_ci95(stats), **_bootstrap_unit_meta(n, clusters)}


def bootstrap_paired_binary_macro_f1_delta(
    y_true_patients: Sequence[Sequence[int]],
    y_pred_a_patients: Sequence[Sequence[int]],
    y_pred_b_patients: Sequence[Sequence[int]],
    *,
    names: Sequence[str],
    n_boot: int = PAPER_BOOTSTRAP_N,
    seed: int = PAPER_BOOTSTRAP_SEED,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Paired patient-level bootstrap of Macro F1_A − Macro F1_B.

    Resample the same patients for both systems so the comparison respects
    within-patient image correlation and within-radiograph slot correlation.
    """
    n = len(y_true_patients)
    if n == 0 or n != len(y_pred_a_patients) or n != len(y_pred_b_patients):
        return {**_ci95([]), "delta_point": None, **_bootstrap_unit_meta(n, [])}
    clusters = _image_clusters(n, cluster_ids)
    yt_p = _stack_patient_slots(y_true_patients)
    ya_p = _stack_patient_slots(y_pred_a_patients)
    yb_p = _stack_patient_slots(y_pred_b_patients)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample_cluster_rows(clusters, rng)
        fa = _binary_macro_f1_fast(yt_p[idx].ravel(), ya_p[idx].ravel())
        fb = _binary_macro_f1_fast(yt_p[idx].ravel(), yb_p[idx].ravel())
        stats.append(float(fa) - float(fb))
    point_a = _binary_macro_f1_fast(yt_p.ravel(), ya_p.ravel())
    point_b = _binary_macro_f1_fast(yt_p.ravel(), yb_p.ravel())
    out = {**_ci95(stats), **_bootstrap_unit_meta(n, clusters)}
    out["delta_point"] = float(point_a) - float(point_b)
    return out


def bootstrap_paired_macro4_f1_delta(
    gt_maps: Sequence[dict],
    pred_a_maps: Sequence[dict],
    pred_b_maps: Sequence[dict],
    fdi_teeth: Sequence[int],
    diseases: Sequence[str],
    *,
    n_boot: int = PAPER_BOOTSTRAP_N,
    seed: int = PAPER_BOOTSTRAP_SEED,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Paired patient-level bootstrap of Macro-4 F1_A − Macro-4 F1_B."""
    n = len(gt_maps)
    if n == 0 or n != len(pred_a_maps) or n != len(pred_b_maps):
        return {**_ci95([]), "delta_point": None, **_bootstrap_unit_meta(n, [])}
    clusters = _image_clusters(n, cluster_ids)
    yt_s = _ovr_patient_stack(gt_maps, fdi_teeth, diseases)
    ya_s = _ovr_patient_stack(pred_a_maps, fdi_teeth, diseases)
    yb_s = _ovr_patient_stack(pred_b_maps, fdi_teeth, diseases)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = _resample_cluster_rows(clusters, rng)
        fa = _macro4_f1_from_stacks(yt_s[idx], ya_s[idx])
        fb = _macro4_f1_from_stacks(yt_s[idx], yb_s[idx])
        if fa is None or fb is None:
            continue
        stats.append(float(fa) - float(fb))
    point_a = _macro4_f1_from_stacks(yt_s, ya_s)
    point_b = _macro4_f1_from_stacks(yt_s, yb_s)
    out = {**_ci95(stats), **_bootstrap_unit_meta(n, clusters)}
    out["delta_point"] = None if point_a is None or point_b is None else float(point_a) - float(point_b)
    return out
