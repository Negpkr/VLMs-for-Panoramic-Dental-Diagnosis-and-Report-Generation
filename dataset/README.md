# Datasets

This folder documents the three panoramic radiograph datasets used in the current evaluation (jobs 27255, 27258, 27299 on 20 August 2026).

The radiographs themselves are **not** in GitHub. They stay on this machine under the original download folders, which are gitignored:

| Dataset | Local path | On-disk size (approx.) |
|---|---|---|
| Tufts Dental Database | `../Tufts_Dental_Database/` | 936 MB |
| DENTEX | `../DENTEX/` | 26 GB |
| PR-Reports | `../PR-Reports/` | 77 MB |

Canonical counts are in `../runs/dataset_counts.json`. Tables and figures from these runs are in `../Main Job Outputs/`.

## How they are used

| Dataset | Research question | Task | Numbering | Scored set | Job |
|---|---|---|---|---|---|
| Tufts | RQ1 / RQ2 | Missing vs present tooth | Universal 1–32 | 1000 cases × 32 slots = 32,000 | 27299 |
| DENTEX QED | RQ1 / RQ2 | Abnormal vs no annotated finding; four diseases | FDI 11–48 | 752 images × 32 slots = 24,064 | 27255 |
| PR-Reports | RQ3 | Generated report vs expert report | FDI in the report text | All 121 cases, zero-shot | 27258 |

Models on all three: LLaVA-1.5-7B, LLaVA-Med, HuatuoGPT-Vision, DentVLM.  
Tufts and DENTEX use zero-shot, few-shot, and chain-of-thought. RQ3 is zero-shot only.

---

## 1. Tufts Dental Database

**Source:** Panetta et al., “Tufts Dental Database: A multimodal panoramic X-ray dataset for benchmarking diagnostic systems,” *IEEE J. Biomed. Health Inform.*, 2022. [doi:10.1109/JBHI.2021.3117575](https://doi.org/10.1109/JBHI.2021.3117575)

**Local layout**

- `Radiographs/` — 1000 panoramic images
- `Segmentation/teeth_bbox.json` — present-tooth boxes (ground truth for missing teeth)
- `Expert/expert.json` — expert narrative reports (**not** used as RQ3 gold)
- `Student/` — student annotations (not scored)

**What we score.** Every usable case with a bbox record and a readable radiograph: **1000 cases**. Ground truth missing teeth are the Universal numbers **1–32 that are absent from `teeth_bbox.json`**. Mean GT missing teeth per case is **6.741** (6741 missing slots, 25259 present).

**What we do not score.** The earlier 50-case Phase 1 pilot (`../archive/`) and the 100-case / July 2026 runs. Those are not the paper evaluation.

---

## 2. DENTEX

**Source:** Hamamci et al., DENTEX (MICCAI abnormal-tooth / enumeration / diagnosis benchmark).  
Challenge paper: [arXiv:2305.19112](https://arxiv.org/abs/2305.19112). Dataset paper: Hamamci et al., *Sci. Data*, 2023, [doi:10.1038/s41597-023-02453-z](https://doi.org/10.1038/s41597-023-02453-z).

**Local layout (QED = quadrant–enumeration–disease)**

- `training_data/quadrant-enumeration-disease/` — 705 images + `train_quadrant_enumeration_disease.json`
- `validation_data/quadrant_enumeration_disease/` — 50 images
- `validation_triple.json` — validation labels
- `disease/` — 250-image test split with LabelMe labels (has no Deep Caries class; **not** in the paper run)

**What we score (`qed`).** Train + val radiographs that exist on disk (**755**), minus three **text-only** few-shot exemplars that are held out of scoring for every strategy so the image set stays the same:

- `train_1` (impacted only)
- `train_14` (multi-label / mixed)
- `train_123` (no annotated finding)

That leaves **752 images**, **24,064 FDI slots**.

DENTEX labels diagnosis only on abnormal teeth. Unannotated slots are **No annotated finding**, not “normal.” One tooth may carry more than one disease; Table B scores Impacted, Caries, Deep Caries, and Periapical Lesion as independent one-vs-rest tasks.

| Quantity | Count |
|---|---|
| Unique abnormal teeth | 3627 (mean 4.82 / case) |
| Diagnosis-label positives | 3696 (mean 4.91 / case; a multi-label tooth is counted in every disease) |
| No-finding slots | 20437 |
| Disease support (OvR) | Impacted 642, Caries 2286, Deep Caries 602, Periapical 166 |
| No-finding-only baseline | Macro F1 **0.459**, accuracy 20437/24064 = **0.849** |

Held-out exemplar details: `../runs/dentex_heldout_exemplars.json`.

---

## 3. PR-Reports (RQ3 gold)

**Source:** Hugging Face [`saatwiksy/PR-Reports`](https://huggingface.co/datasets/saatwiksy/PR-Reports).  
IEEE DataPort: [doi:10.21227/dyae-tt87](https://doi.org/10.21227/dyae-tt87).

**Local layout**

- `images/` — 121 panoramics (`1.jpg` …)
- `train-00000-of-00001.parquet` — expert maxillofacial radiologist reports
- Case list used for the job: `../runs/rq3_pr_reports_all.json`

**What we score.** The **full 121** cases (no hold-out). Gold is the expert report in this dataset, **not** Tufts `expert.json`. Metrics: BLEU-4, ROUGE-L, BERTScore (`bert-base-uncased`).

Strata in the 121-case list: impacted 40, periapical 33, caries 21, other 18, mixed dentition 9.

---

## Not copied here

- Image files and zips stay in `Tufts_Dental_Database/`, `DENTEX/`, and `PR-Reports/` (gitignored).
- Phase 1 LLaVA vs BLIP-2 50-case materials are in `../archive/`, not in this evaluation.
- Result tables from the three jobs are in `../Main Job Outputs/`.
