#!/usr/bin/env python3
"""Synthetic unit tests for the hallucination metric maths.

Run:  python -m hallucination.test_metrics   (from the repo root)
      or  python hallucination/test_metrics.py

Every expected TP/FP/FN below is worked out by hand so the implementation is
checked against known ground truth, not against itself.
"""
from __future__ import annotations

import math

from hallucination.metrics import (
    ImageResult,
    LocationBreakdown,
    aggregate_metrics,
    bootstrap_metrics,
    condition_metrics,
    location_breakdown,
    paired_comparison,
)


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


def _f(loc: str, cond: str):
    return (loc, cond)


# n_slots is large enough that TN never goes negative; its exact value only
# matters for the TN/Specificity checks, which use their own controlled cases.


def test_brief_example() -> None:
    """The exact worked example from the project brief.

    GT   = {caries, missing_tooth}
    Pred = {caries, periapical_lesion, fracture}
    (single 'tooth' so location does not interfere)
    Expected: TP=1, FP=2, FN=1, Precision=1/3, Recall=1/2, F1=0.4, CHR=2/3
    """
    gt = {_f("1", "caries"), _f("1", "missing_tooth")}
    pred = {_f("1", "caries"), _f("1", "periapical_lesion"), _f("1", "fracture")}
    im = ImageResult(image_id="ex", gt=gt, pred=pred, n_slots=100)
    m = aggregate_metrics([im])
    assert m["TP"] == 1, m
    assert m["FP"] == 2, m
    assert m["FN"] == 1, m
    assert _approx(m["Precision"], 1 / 3), m
    assert _approx(m["Recall"], 1 / 2), m
    assert _approx(m["F1"], 0.4), m
    assert _approx(m["CHR"], 2 / 3), m
    # CHR == 1 - Precision under binary matching.
    assert _approx(m["CHR"], 1 - m["Precision"]), m
    print("test_brief_example: OK")


def test_tn_and_specificity() -> None:
    """Controlled TN / specificity / FPR on a tiny 4-slot grid.

    Grid = 4 slots. GT positive = {A}. Pred positive = {A, B}.
      TP=1 (A), FP=1 (B), FN=0, TN = 4 - |{A,B}| = 2.
      Specificity = TN/(TN+FP) = 2/3.  FPR = FP/(FP+TN) = 1/3.
      Accuracy = (TP+TN)/4 = 3/4.
    """
    gt = {_f("t", "A")}
    pred = {_f("t", "A"), _f("t", "B")}
    im = ImageResult(image_id="g", gt=gt, pred=pred, n_slots=4)
    m = aggregate_metrics([im])
    assert m["TP"] == 1 and m["FP"] == 1 and m["FN"] == 0, m
    assert m["TN"] == 2, m
    assert _approx(m["Specificity"], 2 / 3), m
    assert _approx(m["FPR"], 1 / 3), m
    assert _approx(m["Accuracy"], 3 / 4), m
    print("test_tn_and_specificity: OK")


def test_empty_prediction_is_nan_chr() -> None:
    """Model predicts nothing -> CHR undefined (NaN), no crash. FN counts."""
    gt = {_f("1", "caries")}
    im = ImageResult(image_id="empty", gt=gt, pred=set(), n_slots=100)
    m = aggregate_metrics([im])
    assert m["TP"] == 0 and m["FP"] == 0 and m["FN"] == 1, m
    assert isinstance(m["CHR"], float) and math.isnan(m["CHR"]), m
    assert isinstance(m["Precision"], float) and math.isnan(m["Precision"]), m
    assert _approx(m["Recall"], 0.0), m  # 0/(0+1)
    assert m["Hallucination_image_rate"] == 0.0, m
    print("test_empty_prediction_is_nan_chr: OK")


def test_no_gt_but_predicts_disease() -> None:
    """No abnormality in GT but model predicts disease -> all FP, CHR=1."""
    pred = {_f("1", "caries"), _f("2", "caries")}
    im = ImageResult(image_id="hall", gt=set(), pred=pred, n_slots=100)
    m = aggregate_metrics([im])
    assert m["TP"] == 0 and m["FP"] == 2 and m["FN"] == 0, m
    assert _approx(m["CHR"], 1.0), m
    assert m["Hallucination_image_rate"] == 1.0, m
    print("test_no_gt_but_predicts_disease: OK")


def test_location_breakdown() -> None:
    """Wrong-tooth vs wrong-condition vs unsupported classification.

    GT   = {26:Caries, 11:Impacted}
    Pred = {16:Caries,             # Caries present in GT (at 26), wrong tooth -> wrong_tooth
            11:Impacted,           # exactly correct
            11:Periapical Lesion,  # tooth 11 has a GT finding, but Periapical Lesion is
                                   #   absent everywhere in GT -> wrong_condition
            37:Deep Caries}        # tooth 37 has no GT finding + condition absent -> unsupported
    Expected: correct=1, wrong_tooth=1, wrong_condition=1, unsupported=1, total=4
    Wrong-tooth hallucination rate = 1/4.

    Priority rule (per brief): if the predicted condition exists ANYWHERE in the
    GT it is scored as a location (wrong-tooth) error before considering the
    tooth, because the wrong-tooth rate counts correct-condition predictions
    placed on the wrong tooth.
    """
    gt = {_f("26", "Caries"), _f("11", "Impacted")}
    pred = {
        _f("16", "Caries"),
        _f("11", "Impacted"),
        _f("11", "Periapical Lesion"),
        _f("37", "Deep Caries"),
    }
    im = ImageResult(image_id="loc", gt=gt, pred=pred, n_slots=128)
    lb = location_breakdown([im])
    assert lb.correct == 1, lb
    assert lb.wrong_tooth == 1, lb
    assert lb.wrong_condition == 1, lb
    assert lb.unsupported == 1, lb
    assert lb.total_pred == 4, lb
    assert _approx(lb.wrong_tooth_hallucination_rate, 1 / 4), lb
    print("test_location_breakdown: OK")


def test_condition_metrics() -> None:
    """Per-condition split, multi-image, with a share-of-hallucinations check.

    Image 1: GT {26:Caries}          Pred {26:Caries, 15:Impacted}
    Image 2: GT {11:Impacted}        Pred {12:Impacted, 30:Caries}
      Caries:   TP=1(26), FP=1(30), FN=0            -> hallucinations=1
      Impacted: TP=0,     FP=2(15,12), FN=1(11)     -> hallucinations=2
    Total model hallucinations (FP) = 3.
      Caries share   = 1/3 * 100
      Impacted share = 2/3 * 100
    n_teeth=32 -> Caries TN = 32*2 - |pos_either|.  pos_either(Caries)= {26,30}=2 -> TN=62.
    """
    im1 = ImageResult("1", {_f("26", "Caries")},
                      {_f("26", "Caries"), _f("15", "Impacted")}, n_slots=128)
    im2 = ImageResult("2", {_f("11", "Impacted")},
                      {_f("12", "Impacted"), _f("30", "Caries")}, n_slots=128)
    cm = condition_metrics([im1, im2], ["Caries", "Impacted"], n_teeth=32)
    assert cm["Caries"]["TP"] == 1 and cm["Caries"]["FP"] == 1 and cm["Caries"]["FN"] == 0, cm
    assert cm["Impacted"]["TP"] == 0 and cm["Impacted"]["FP"] == 2 and cm["Impacted"]["FN"] == 1, cm
    assert cm["Caries"]["TN"] == 62, cm
    assert _approx(cm["Caries"]["Pct_of_model_hallucinations"], 1 / 3 * 100), cm
    assert _approx(cm["Impacted"]["Pct_of_model_hallucinations"], 2 / 3 * 100), cm
    print("test_condition_metrics: OK")


def test_bootstrap_reproducible_and_bracketed() -> None:
    """Bootstrap is deterministic under the fixed seed and brackets the point."""
    ims = [
        ImageResult(str(i),
                    {_f(str(i), "Caries")},
                    {_f(str(i), "Caries")} if i % 2 == 0 else {_f(str(i), "Impacted")},
                    n_slots=128)
        for i in range(40)
    ]
    b1 = bootstrap_metrics(ims, n_boot=200, seed=42)
    b2 = bootstrap_metrics(ims, n_boot=200, seed=42)
    assert b1["CHR"]["low"] == b2["CHR"]["low"], "bootstrap not reproducible"
    assert b1["CHR"]["low"] <= b1["CHR"]["point"] <= b1["CHR"]["high"] + 1e-9, b1["CHR"]
    print("test_bootstrap_reproducible_and_bracketed: OK")


def test_paired_comparison_sign() -> None:
    """Paired delta sign: a hallucinating model A vs a clean model B."""
    a = [ImageResult(str(i), {_f(str(i), "Caries")},
                     {_f(str(i), "Caries"), _f("99", "Impacted")}, 128) for i in range(30)]
    b = [ImageResult(str(i), {_f(str(i), "Caries")},
                     {_f(str(i), "Caries")}, 128) for i in range(30)]
    res = paired_comparison(a, b, n_boot=200, seed=42)
    assert res["Delta_CHR"] > 0, res           # A hallucinates more
    assert res["Delta_hallucinations_per_image"] > 0, res
    print("test_paired_comparison_sign: OK")


def main() -> None:
    test_brief_example()
    test_tn_and_specificity()
    test_empty_prediction_is_nan_chr()
    test_no_gt_but_predicts_disease()
    test_location_breakdown()
    test_condition_metrics()
    test_bootstrap_reproducible_and_bracketed()
    test_paired_comparison_sign()
    print("\nAll hallucination metric tests passed.")


if __name__ == "__main__":
    main()
