# Figure 5 selection protocol (locked before inspecting model outcomes)

Categories are searched in this order. For each category, take the **first** matching case in sorted `case_id` order from a predeclared model/strategy. Do not skip a match to find a “nicer” image. Do not substitute a motivating example (including `val_18`) if an earlier case already matches.

**Declared source (after the run):** HuatuoGPT-Vision, zero-shot, on the scored split. If that model failed, use the first complete model in registry order: LLaVA, LLaVA-Med, DentVLM. Tufts panel B uses the same model’s zero-shot Tufts full run.

| Panel | Category | Dataset | Match rule (first hit, sorted case_id) |
|---|---|---|---|
| A | Completely correct | DENTEX qed scored set | Parsed findings equal GT on every FDI slot (including empty = no finding) and GT has ≥1 abnormal tooth |
| B | Missing-tooth FN | Tufts full | ≥1 Universal slot with GT missing and prediction present |
| C | Abnormality FN | DENTEX | ≥1 FDI with GT diagnoses and empty prediction for that tooth |
| D | Correct multi-label | DENTEX | ≥1 tooth with ≥2 GT diagnoses and prediction set equals GT set for that tooth |
| E | Partial multi-label FN | DENTEX | ≥1 tooth with ≥2 GT diagnoses, prediction shares ≥1 label and misses ≥1 label |

If a category has no match, that panel is omitted (not replaced by a prettier near-miss).

The selection log is written to `Paper/figures_main/Figure5_selection_log.json` when Figure 5 is generated.
