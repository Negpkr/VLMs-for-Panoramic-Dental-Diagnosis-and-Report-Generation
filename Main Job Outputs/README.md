# Main Job Outputs

These files are from the three 20 August 2026 jobs.


| Job                                      | ID    | Dataset    | Status                                |
| ---------------------------------------- | ----- | ---------- | ------------------------------------- |
| DENTEX qed 752 × 4 models × 3 strategies | 27255 | DENTEX     | Complete                              |
| RQ3 expert reports, all 121 cases        | 27258 | PR-Reports | Complete                              |
| Tufts 1000 × 4 models × 3 strategies     | 27299 | Tufts      | Complete (ended 17:17 AEST, 6 h 44 m) |


## Layout

- `dentex/` — pooled-slot summary, supplementary P/R, and confusion CSVs (abnormal 2×2, disease OvR, 5-class exclusive).
- `rq3/` — BLEU / ROUGE-L / BERTScore summary, full per-case JSON, empty Likert rater sheet.
- `tufts/` — pooled-slot summary + supplementary for zero-shot, few-shot, and CoT (1000 cases), plus missing-vs-present 2×2 counts (`tufts_1000_confusion_missing.csv`).
- `figures/` — Figures 1–6 (PDF + PNG + captions). Figures 3 and 4 were regenerated after Tufts 27299 finished so the Tufts panels use the 1000-case pooled matrix.

Raw per-case model JSON (12–24 MB each) stays in `Results/` locally and is not duplicated here.

Mean pred/GT columns in the DENTEX summaries are **unique abnormal teeth per case**, not diagnosis-label counts. No-finding-only baseline Macro F1 is 0.459 (not a floor); baseline accuracy is 0.849. Tufts mean pred/GT columns are unique missing teeth per case (GT = 6.741).