#!/usr/bin/env python3
"""Finding-level hallucination metrics for the dental VLM comparison.

This module is a *separate, additive* evaluation component. It does NOT touch
the existing pooled-slot metrics in ``eval_metrics.py``; it reuses the same
prediction/ground-truth representation already stored in the saved run JSONs
and re-scores it from a hallucination point of view.

Terminology (kept deliberately careful, per the project brief):
  * A "finding" is a positive claim that a *condition* is present at a
    *location* (an FDI tooth). It is the unit of hallucination analysis.
  * TP  = predicted finding that is supported by the ground truth (matching
          rule below).
  * FP  = predicted finding NOT supported by the ground truth. This is the
          "hallucination candidate" / unsupported positive finding.
  * FN  = ground-truth finding the model failed to predict (missed finding).
  * TN  = a (location, condition) slot that is negative in BOTH the ground
          truth and the prediction. Meaningful here because the label space
          is a fixed finite grid of tooth x condition slots.

Matching rule
-------------
We reuse the repository's already-normalised labels verbatim (the parser wrote
``predicted_findings`` / ``predicted_teeth`` with the project's own
label-normalisation; we add none of our own). A predicted finding matches a
ground-truth finding iff they share BOTH the exact location (FDI tooth) and the
exact condition string. This is the same tooth-level, label-normalised
convention the pooled evaluation uses; we simply score it per finding.

  * DENTEX: a finding is (tooth, disease) with disease in
    {Impacted, Caries, Deep Caries, Periapical Lesion}. Location-aware.
  * Tufts : a finding is (tooth, "Missing"). The task is a single condition
    ("missing tooth"); the location is intrinsic to the finding.

Clinical Hallucination Rate (CHR)
---------------------------------
    CHR = (unsupported predicted findings) / (total predicted findings)
        = FP / (TP + FP)
Under this binary matching CHR == 1 - Precision; we report both because CHR is
the interpretable "share of the model's claims that are unsupported" number.
If TP + FP == 0 (the model predicted nothing) CHR is undefined -> NaN.

No bounding boxes or segmentation masks exist anywhere in the saved runs, so
IoU is intentionally not computed (see run_analysis.py report).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Finding extraction
# ---------------------------------------------------------------------------
# A finding is represented as a (location, condition) tuple of two strings.
# Serialised form for CSV output is "location:condition" (e.g. "26:Caries").

Finding = tuple[str, str]

TUFTS_CONDITION = "Missing"  # single condition name for the Tufts missing-tooth task


def findings_from_dentex(mapping: dict | None) -> set[Finding]:
    """DENTEX: {tooth: [disease, ...]} -> {(tooth, disease), ...}.

    ``mapping`` is the parser's ``predicted_findings`` / ``ground_truth_findings``.
    Labels are used exactly as stored (already normalised upstream).
    """
    out: set[Finding] = set()
    if not mapping:
        return out
    for tooth, labels in mapping.items():
        # labels is normally a list; tolerate a bare string just in case.
        if isinstance(labels, str):
            labels = [labels]
        for lab in labels or []:
            out.add((str(tooth), str(lab)))
    return out


def findings_from_tufts(teeth: Iterable[int] | None) -> set[Finding]:
    """Tufts: [tooth, ...] (missing teeth) -> {(tooth, "Missing"), ...}."""
    out: set[Finding] = set()
    if not teeth:
        return out
    for t in teeth:
        out.add((str(t), TUFTS_CONDITION))
    return out


def serialise_findings(findings: Iterable[Finding]) -> str:
    """Deterministic, human-readable serialisation for CSV cells."""
    return "; ".join(f"{loc}:{cond}" for loc, cond in sorted(findings))


# ---------------------------------------------------------------------------
# Per-image confusion at the finding level
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """Finding-level confusion for one image under one model."""

    image_id: str
    gt: set[Finding]
    pred: set[Finding]
    n_slots: int  # tooth x condition grid size for this image (for TN)

    @property
    def tp_set(self) -> set[Finding]:
        return self.pred & self.gt

    @property
    def fp_set(self) -> set[Finding]:  # hallucination candidates
        return self.pred - self.gt

    @property
    def fn_set(self) -> set[Finding]:  # missed findings
        return self.gt - self.pred

    @property
    def tp(self) -> int:
        return len(self.tp_set)

    @property
    def fp(self) -> int:
        return len(self.fp_set)

    @property
    def fn(self) -> int:
        return len(self.fn_set)

    @property
    def tn(self) -> int:
        # Slots negative in both = grid - (positives in either).
        return self.n_slots - len(self.gt | self.pred)

    @property
    def n_pred(self) -> int:
        return len(self.pred)

    @property
    def image_chr(self) -> float:
        """Per-image hallucination rate = FP / predicted findings (NaN if none)."""
        return self.fp / self.n_pred if self.n_pred else math.nan


def safe_div(num: float, den: float) -> float:
    """Division that returns NaN instead of raising on a zero denominator."""
    return num / den if den else math.nan


# ---------------------------------------------------------------------------
# Aggregate (model-level) metrics from a list of ImageResult
# ---------------------------------------------------------------------------


def aggregate_metrics(images: Sequence[ImageResult]) -> dict[str, Any]:
    """Pool TP/FP/FN/TN across images and derive every model-level metric.

    All rate metrics degrade to NaN (never crash) when their denominator is 0.
    """
    n_images = len(images)
    tp = sum(im.tp for im in images)
    fp = sum(im.fp for im in images)
    fn = sum(im.fn for im in images)
    tn = sum(im.tn for im in images)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (
        math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0
    ) else math.nan
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    chr_ = safe_div(fp, tp + fp)  # Clinical Hallucination Rate

    n_pred = sum(im.n_pred for im in images)
    images_with_hallucination = sum(1 for im in images if im.fp > 0)

    return {
        "Images": n_images,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "CHR": chr_,
        "Hallucinations_per_image": safe_div(fp, n_images),
        "Hallucination_image_rate": safe_div(images_with_hallucination, n_images),
        "Mean_predictions_per_image": safe_div(n_pred, n_images),
        "Mean_correct_findings_per_image": safe_div(tp, n_images),
        "Mean_hallucinations_per_image": safe_div(fp, n_images),
    }


# ---------------------------------------------------------------------------
# Per-condition metrics
# ---------------------------------------------------------------------------


def condition_metrics(
    images: Sequence[ImageResult],
    conditions: Sequence[str],
    n_teeth: int,
) -> dict[str, dict[str, Any]]:
    """One-vs-rest finding metrics restricted to each condition.

    TN for a condition = (n_teeth * n_images) - positives-in-either for that
    condition, i.e. the tooth slots where that condition is absent in both.
    """
    n_images = len(images)
    total_fp = sum(im.fp for im in images)  # for share-of-hallucinations
    out: dict[str, dict[str, Any]] = {}
    for cond in conditions:
        tp = fp = fn = pos_either = 0
        for im in images:
            g = {f for f in im.gt if f[1] == cond}
            p = {f for f in im.pred if f[1] == cond}
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
            pos_either += len(g | p)
        tn = n_teeth * n_images - pos_either
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if not (
            math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0
        ) else math.nan
        out[cond] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPR": safe_div(fp, fp + tn),
            "CHR": safe_div(fp, tp + fp),
            "Hallucinations": fp,
            # Share of this model's total hallucinations attributable to cond:
            "Pct_of_model_hallucinations": safe_div(fp, total_fp) * 100
            if total_fp
            else math.nan,
        }
    return out


# ---------------------------------------------------------------------------
# Location-aware breakdown (DENTEX only; multi-condition + tooth-specific GT)
# ---------------------------------------------------------------------------


@dataclass
class LocationBreakdown:
    """Classification of every predicted finding by *why* it is right/wrong."""

    correct: int = 0            # correct condition + correct tooth  (== TP)
    wrong_tooth: int = 0        # correct condition, wrong tooth (location hallucination)
    wrong_condition: int = 0    # tooth has some GT finding, but a different condition
    unsupported: int = 0        # tooth has no GT finding at all + condition absent in image
    total_pred: int = 0
    wrong_tooth_examples: list[str] = field(default_factory=list)

    @property
    def wrong_tooth_hallucination_rate(self) -> float:
        """Correct-condition-wrong-tooth predictions / total predicted findings."""
        return safe_div(self.wrong_tooth, self.total_pred)

    def as_row(self) -> dict[str, Any]:
        return {
            "Correct_condition_correct_tooth": self.correct,
            "Correct_condition_wrong_tooth": self.wrong_tooth,
            "Wrong_condition_right_tooth": self.wrong_condition,
            "Unsupported_finding": self.unsupported,
            "Total_predicted_findings": self.total_pred,
            "Wrong_tooth_hallucination_rate": self.wrong_tooth_hallucination_rate,
        }


def location_breakdown(images: Sequence[ImageResult]) -> LocationBreakdown:
    """Classify predicted findings into the four location-aware buckets.

    Only meaningful when the ground truth is tooth-specific AND more than one
    condition exists (DENTEX). For Tufts the single condition IS the tooth, so
    "wrong tooth" collapses into an ordinary FP and this breakdown is skipped.
    """
    lb = LocationBreakdown()
    for im in images:
        gt_conditions = {c for _, c in im.gt}      # conditions present anywhere in image
        gt_teeth = {t for t, _ in im.gt}           # teeth carrying any GT finding
        for (tooth, cond) in im.pred:
            lb.total_pred += 1
            if (tooth, cond) in im.gt:
                lb.correct += 1
            elif cond in gt_conditions:
                lb.wrong_tooth += 1
                if len(lb.wrong_tooth_examples) < 20:
                    gt_locs = sorted(t for t, c in im.gt if c == cond)
                    lb.wrong_tooth_examples.append(
                        f"{im.image_id}: predicted {cond}@{tooth}; "
                        f"GT {cond}@{gt_locs}"
                    )
            elif tooth in gt_teeth:
                lb.wrong_condition += 1
            else:
                lb.unsupported += 1
    return lb


# ---------------------------------------------------------------------------
# Image-level bootstrap confidence intervals
# ---------------------------------------------------------------------------

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

# Metrics we bootstrap. Each maps to a function that takes pooled counts and a
# per-image hallucination-flag total and returns the scalar metric.


def _pooled_from_indices(images: Sequence[ImageResult], idx: np.ndarray) -> dict[str, float]:
    tp = fp = fn = tn = n_pred = img_hall = 0
    for i in idx:
        im = images[i]
        tp += im.tp
        fp += im.fp
        fn += im.fn
        tn += im.tn
        n_pred += im.n_pred
        img_hall += 1 if im.fp > 0 else 0
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (
        math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0
    ) else math.nan
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "CHR": safe_div(fp, tp + fp),
        "Hallucination_image_rate": safe_div(img_hall, len(idx)),
    }


def bootstrap_metrics(
    images: Sequence[ImageResult],
    *,
    metrics: Sequence[str] = ("Precision", "Recall", "F1", "CHR", "Hallucination_image_rate"),
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile 95% CIs by resampling IMAGES (the independent unit).

    Predictions from the same image are not independent, so resampling happens
    at image level, never at finding level. Fixed seed for reproducibility.
    Returns {metric: {point, low, high, mean, n_boot}}.
    """
    n = len(images)
    point = _pooled_from_indices(images, np.arange(n)) if n else {}
    if n == 0:
        return {m: {"point": math.nan, "low": math.nan, "high": math.nan,
                    "mean": math.nan, "n_boot": 0} for m in metrics}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = _pooled_from_indices(images, idx)
        for m in metrics:
            v = vals[m]
            if not math.isnan(v):
                draws[m].append(v)
    out: dict[str, dict[str, float]] = {}
    for m in metrics:
        arr = np.asarray(draws[m], dtype=float)
        if arr.size:
            out[m] = {
                "point": point[m],
                "low": float(np.quantile(arr, 0.025)),
                "high": float(np.quantile(arr, 0.975)),
                "mean": float(np.mean(arr)),
                "n_boot": int(arr.size),
            }
        else:
            out[m] = {"point": point.get(m, math.nan), "low": math.nan,
                      "high": math.nan, "mean": math.nan, "n_boot": 0}
    return out


# ---------------------------------------------------------------------------
# Paired model comparison (same images evaluated by both models)
# ---------------------------------------------------------------------------


def _chr_and_hpi(images: Sequence[ImageResult], idx: np.ndarray) -> tuple[float, float]:
    fp = sum(images[i].fp for i in idx)
    tp = sum(images[i].tp for i in idx)
    chr_ = safe_div(fp, tp + fp)
    hpi = safe_div(fp, len(idx))  # hallucinations per image
    return chr_, hpi


def paired_comparison(
    images_a: Sequence[ImageResult],
    images_b: Sequence[ImageResult],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired image-level bootstrap of (A - B) for CHR and hallucinations/image.

    ``images_a`` and ``images_b`` MUST be aligned by image (same index = same
    image, both models). The same resampled image indices are used for both
    models in each replicate so the comparison respects the pairing.
    """
    n = len(images_a)
    if n == 0 or n != len(images_b):
        return {}
    full = np.arange(n)
    chr_a, hpi_a = _chr_and_hpi(images_a, full)
    chr_b, hpi_b = _chr_and_hpi(images_b, full)
    rng = np.random.default_rng(seed)
    d_chr: list[float] = []
    d_hpi: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # one resample, applied to BOTH models
        ca, ha = _chr_and_hpi(images_a, idx)
        cb, hb = _chr_and_hpi(images_b, idx)
        if not (math.isnan(ca) or math.isnan(cb)):
            d_chr.append(ca - cb)
        d_hpi.append(ha - hb)

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return math.nan, math.nan
        a = np.asarray(vals)
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    chr_lo, chr_hi = ci(d_chr)
    hpi_lo, hpi_hi = ci(d_hpi)
    return {
        "CHR_A": chr_a,
        "CHR_B": chr_b,
        "Delta_CHR": chr_a - chr_b,
        "Delta_CHR_CI95_low": chr_lo,
        "Delta_CHR_CI95_high": chr_hi,
        "Hallucinations_per_image_A": hpi_a,
        "Hallucinations_per_image_B": hpi_b,
        "Delta_hallucinations_per_image": hpi_a - hpi_b,
        "Delta_hpi_CI95_low": hpi_lo,
        "Delta_hpi_CI95_high": hpi_hi,
        "n_images": n,
    }
