# Main Job Outputs

Replaces `temp_test_result/` (7 August 100-case / DENTEX val-50 supervisor pack).

These files are from the three 20 August 2026 jobs. `Paper/` is gitignored, so figures are copied here so they are on GitHub.

| Job | ID | Dataset | Status |
|---|---|---|---|
| DENTEX qed 752 × 4 models × 3 strategies | 27255 | DENTEX | Complete |
| RQ3 expert reports, all 121 cases | 27258 | PR-Reports | Complete |
| Tufts 1000 × 4 models × 3 strategies | 27299 | Tufts | Zero-shot and few-shot complete; **CoT still running** |

## Layout

- `dentex/` — pooled-slot summary, supplementary P/R, and confusion CSVs (abnormal 2×2, disease OvR, 5-class exclusive).
- `rq3/` — BLEU / ROUGE-L / BERTScore summary, full per-case JSON, empty Likert rater sheet.
- `tufts/` — pooled-slot summary + supplementary for zero-shot and few-shot (1000 cases). CoT will be added when 27299 finishes.
- `figures/` — Figures 1–6 (PDF + PNG + captions). Figure 3’s Tufts panel was drawn before job 27299 finished, so regenerate it after CoT completes.

Raw per-case model JSON (12–24 MB each) stays in `Results/` locally and is not duplicated here.

Mean pred/GT columns in the DENTEX summaries are **unique abnormal teeth per case**, not diagnosis-label counts. No-finding-only baseline Macro F1 is 0.459 (not a floor); baseline accuracy is 0.849.
