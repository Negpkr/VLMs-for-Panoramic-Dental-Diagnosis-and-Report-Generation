# Hallucination analysis (finding-level)

An **additive** evaluation component that measures whether each VLM hallucinates
dental findings. It only *reads* the existing run JSONs under `Results/` and
*writes* new CSVs/figures. It does **not** modify any prediction, ground-truth
label, or existing metric file, and it does not touch `eval_metrics.py`.

## What a "finding" is

A finding is a positive claim that a **condition** is present at a **location**
(an FDI tooth). It is the unit of hallucination analysis and is taken directly
from the parser's already-normalised output:

| Dataset | Finding | Source fields | Conditions |
|---------|---------|---------------|------------|
| DENTEX (QED) | `(tooth, disease)` | `predicted_findings`, `ground_truth_findings` (`{tooth: [disease]}`) | Impacted, Caries, Deep Caries, Periapical Lesion |
| Tufts (missing teeth) | `(tooth, "Missing")` | `predicted_teeth`, `ground_truth_teeth` (lists) | Missing (single condition) |

**Matching rule (reuses the project's tooth-level, label-normalised convention):**
a predicted finding is a **TP** iff an identical `(tooth, condition)` exists in
the ground truth; otherwise it is a **FP** (hallucination candidate). A GT
finding with no matching prediction is a **FN**. **TN** = tooth×condition slots
negative in both. No new label normalisation is applied — labels are used
exactly as the parser stored them.

**Clinical Hallucination Rate:** `CHR = FP / (TP + FP)`. Under binary matching
this equals `1 − Precision`; both are reported because CHR reads directly as
"share of the model's claims that are unsupported". `TP+FP == 0` → `NaN`
(never a crash).

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Pure metric functions (no I/O). Model-, condition-, image-level metrics; location-aware breakdown; image-level bootstrap CIs; paired model comparison. |
| `run_analysis.py` | Loads the run JSONs, writes all CSVs + a 10-case manual audit. |
| `figures.py` | Renders the figure set (PNG @300 DPI + PDF) from the CSVs. |
| `test_metrics.py` | Synthetic unit tests with hand-computed TP/FP/FN (incl. the brief's worked example). |

## Reproduce

```bash
python -m hallucination.test_metrics          # validate the maths
python -m hallucination.run_analysis          # writes results/hallucination/*.csv
python -m hallucination.figures               # writes figures/hallucination/<dataset>_<strategy>/*
```

All randomness is seeded: bootstrap uses **1000 resamples, seed 42**, resampling
at **image level** (findings within an image are not independent). The 10-case
audit draw uses seed 42 and is salted with guaranteed hard cases.

## Outputs

`results/hallucination/`
- `model_hallucination_summary.csv` — one row per (Dataset, Strategy, Model): TP/FP/FN/TN, Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, CHR, hallucinations/image, hallucination-image-rate, mean predicted/correct/hallucinated per image.
- `condition_hallucination_summary.csv` — per condition: TP/FP/FN/TN, P/R/F1, FPR, CHR, #hallucinations, % of that model's hallucinations.
- `image_level_hallucination_results.csv` — one row per image per model with GT/pred/correct/hallucinated/missed findings (tooth numbers preserved) and per-image counts + rate.
- `location_analysis.csv` — **DENTEX only.** Splits every predicted finding into correct / correct-condition-wrong-tooth / wrong-condition-right-tooth / unsupported, and reports the Wrong-Tooth Hallucination Rate.
- `bootstrap_confidence_intervals.csv` — image-level 95% CIs for Precision, Recall, F1, CHR, hallucination-image-rate.
- `model_pairwise_comparisons.csv` — paired image-level bootstrap of ΔCHR and Δhallucinations-per-image for every model pair on identical images.
- `verification_sample.txt` — the 10-case manual audit per run.
- `README_run_context.json` — provenance + the exact matching rule.

`figures/hallucination/<dataset>_<strategy>/` — Figures 1–8 per run; count/rate
condition heatmaps and top-conditions for DENTEX (multi-condition). Figure 9
(Precision vs CHR) is intentionally omitted because `CHR = 1 − Precision`
(see `FIGURE9_not_generated.txt`).

## Scope notes / limitations

- **IoU is not computed** — no bounding boxes or segmentation masks exist in the
  saved runs; the localisation signal available is the FDI tooth number, which
  the wrong-tooth analysis uses.
- **Wrong-tooth analysis is DENTEX-only.** For Tufts the tooth *is* the finding
  (single condition), so a wrong tooth is simply an ordinary FP.
- **FP ≠ "hallucination" unconditionally.** A large share of DENTEX FPs are
  correct-condition-wrong-tooth localisation errors (see `location_analysis.csv`),
  not fully unsupported conditions. CHR counts all unsupported *positive* findings;
  interpret alongside the location breakdown.
- **TN/Specificity/FPR live on a negative-heavy slot grid** (mostly-absent
  conditions), so accuracy and specificity are optimistic by class imbalance;
  Precision/Recall/F1/CHR are the honest hallucination signals.
