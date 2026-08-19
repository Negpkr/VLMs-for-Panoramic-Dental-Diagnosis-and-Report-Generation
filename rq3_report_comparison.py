#!/usr/bin/env python3
"""RQ3: model-generated panoramic reports vs expert radiologist reports.

Dataset: PR-Reports (all OPG + maxillofacial radiologist pairs in the database).
Protocol: full database, four VLMs, zero-shot sectional prompt.
Metrics: BLEU-4, ROUGE-L F1, BERTScore F1 (if bert-score is installed).

Does not launch itself. Use:
    python rq3_report_comparison.py --prepare --dry-run
    bash runs/run_rq3.sh
    sbatch runs/sbatch_rq3.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_models  # noqa: E402

ROOT = Path("/home/s222393187/Dental")
PARQUET = ROOT / "PR-Reports" / "train-00000-of-00001.parquet"
IMAGE_DIR = ROOT / "PR-Reports" / "images"
CASE_LIST = ROOT / "runs" / "rq3_pr_reports_all.json"
RESULTS_DIR = ROOT / "Results" / "rq3_reports"
SEED = 20260820
BERTSCORE_MODEL = "bert-base-uncased"

REPORT_PROMPT = (
    "You are a maxillofacial radiologist. Look at this panoramic dental radiograph "
    "and write a structured report using FDI tooth numbers.\n\n"
    "Use EXACTLY these headings and fill each line. Write Nil if there is no finding.\n\n"
    "Teeth present:\n"
    "<upper arch FDI numbers, right to left>\n"
    "<lower arch FDI numbers, left to right>\n"
    "Caries present: <FDI numbers or Nil>\n"
    "Restorations present: <FDI numbers or Nil>\n"
    "Endodontically treated: <FDI numbers or Nil>\n"
    "Root stumps: <FDI numbers or Nil>\n"
    "Periapical changes: <finding and tooth, or Nil>\n"
    "Impacted tooth: <FDI numbers or Nil>\n"
    "Alveolar bone: <Normal or describe bone loss>\n"
    "Temporomandibular joint: <Intact/Normal or describe>\n"
    "Maxillary sinus: <Intact or describe>\n"
    "Artifacts seen: <describe or Nil>\n"
    "Central lesions: <describe or Nil>\n"
    "Radiographic impression: <one or two sentences covering the main diagnoses>\n"
)

FIELD_NAMES = (
    "Caries present",
    "Restorations present",
    "Endodontically treated",
    "Root stumps",
    "Periapical changes",
    "Impacted tooth",
    "Alveolar bone",
    "Temporomandibular joint",
    "Maxillary sinus",
    "Artifacts seen",
    "Central lesions",
    "Radiographic impression",
)


def _field(text: str, name: str) -> str:
    pat = rf"{re.escape(name)}\s*:\s*(.*?)(?:\n[A-Z][A-Za-z][A-Za-z /]*:|\Z)"
    m = re.search(pat, text, re.S)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _is_nil(value: str) -> bool:
    v = value.lower().strip(" .;")
    return v in {"", "nil", "none", "normal", "intact", "-", "na", "n/a"}


def assign_stratum(report: str) -> str:
    mixed = "mixed dentition" in report.lower()
    impacted = not _is_nil(_field(report, "Impacted tooth"))
    peri = not _is_nil(_field(report, "Periapical changes"))
    caries = not _is_nil(_field(report, "Caries present"))
    if mixed:
        return "mixed_dentition"
    if impacted:
        return "impacted"
    if peri:
        return "periapical"
    if caries:
        return "caries"
    return "other"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bleu4(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not hyp or not ref:
        return 0.0
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    return float(
        sentence_bleu(
            [ref],
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    n, m = len(ref), len(hyp)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[m]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def load_parquet_rows() -> list[dict[str, Any]]:
    table = pq.read_table(PARQUET)
    reports = table.column("report").to_pylist()
    images = table.column("image").to_pylist()
    rows = []
    for i, (rep, img) in enumerate(zip(reports, images)):
        raw_name = Path(str((img or {}).get("path") or f"{i}.jpg")).name
        stem = Path(raw_name).stem
        case_id = f"pr_{stem}"
        rows.append(
            {
                "index": i,
                "case_id": case_id,
                "file_name": raw_name,
                "expert_report": rep,
                "image_bytes": (img or {}).get("bytes"),
                "stratum": assign_stratum(rep),
            }
        )
    return rows


def prepare_dataset(log=print) -> dict[str, Any]:
    if not PARQUET.exists():
        raise SystemExit(f"Missing {PARQUET}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CASE_LIST.parent.mkdir(parents=True, exist_ok=True)
    rows = load_parquet_rows()
    eval_rows = sorted(rows, key=lambda r: r["index"])

    extracted = 0
    for row in rows:
        dest = IMAGE_DIR / row["file_name"]
        if not dest.exists():
            Image.open(BytesIO(row["image_bytes"])).convert("RGB").save(dest, quality=95)
            extracted += 1
        row["image_path"] = str(dest)

    payload = {
        "dataset": "PR-Reports",
        "source": "https://huggingface.co/datasets/saatwiksy/PR-Reports",
        "seed": SEED,
        "n_eval": len(eval_rows),
        "n_holdout": 0,
        "n_total": len(rows),
        "eval_stratum_counts": dict(Counter(r["stratum"] for r in eval_rows)),
        "eval_cases": [
            {
                "case_id": r["case_id"],
                "file_name": r["file_name"],
                "image_path": str(IMAGE_DIR / r["file_name"]),
                "stratum": r["stratum"],
                "expert_report": r["expert_report"],
            }
            for r in eval_rows
        ],
        "holdout_case_ids": [],
    }
    CASE_LIST.write_text(json.dumps(payload, indent=2))
    log(f"prepared all {len(eval_rows)} PR-Reports cases -> {CASE_LIST}")
    log(f"extracted {extracted} new images into {IMAGE_DIR}")
    log(f"eval strata: {payload['eval_stratum_counts']}")
    return payload


def load_case_list() -> dict[str, Any]:
    if not CASE_LIST.exists():
        return prepare_dataset()
    return json.loads(CASE_LIST.read_text())


def score_pair(reference: str, hypothesis: str) -> dict[str, float]:
    return {
        "bleu4": bleu4(reference, hypothesis),
        "rouge_l_f1": rouge_l_f1(reference, hypothesis),
    }


def bertscore_f1(references: list[str], hypotheses: list[str]) -> list[float] | None:
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None
    _, _, f1 = bert_score(
        hypotheses,
        references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        verbose=False,
        device="cpu",
    )
    return [float(x) for x in f1]


def summarize_model(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bleu = [c["metrics"]["bleu4"] for c in cases if "metrics" in c]
    rouge = [c["metrics"]["rouge_l_f1"] for c in cases if "metrics" in c]
    bert = [c["metrics"]["bertscore_f1"] for c in cases if c.get("metrics", {}).get("bertscore_f1") is not None]
    return {
        "n_scored": len(bleu),
        "n_errors": sum(1 for c in cases if "error" in c),
        "bleu4_mean": mean(bleu),
        "rouge_l_f1_mean": mean(rouge),
        "bertscore_f1_mean": mean(bert),
    }


def evaluate_model(
    model_key: str,
    cases: list[dict[str, Any]],
    *,
    prompt: str,
    max_new_tokens: int,
    log=print,
) -> dict[str, Any]:
    spec = vlm_models.MODEL_REGISTRY[model_key]
    log(f"=== {spec.display_name} ({spec.repo_id}) ===")
    bundle = None
    results: list[dict[str, Any]] = []
    try:
        bundle = vlm_models.load_model(model_key)
        log(f"loaded in {bundle.load_seconds:.1f}s on {bundle.device}")
        for i, case in enumerate(cases, 1):
            t0 = time.time()
            try:
                reply = vlm_models.generate(
                    bundle,
                    case["image_path"],
                    prompt,
                    seed_key=f"rq3:{model_key}:{case['case_id']}",
                    max_new_tokens=max_new_tokens,
                )
                pair = score_pair(case["expert_report"], reply)
                results.append(
                    {
                        "case_id": case["case_id"],
                        "file_name": case["file_name"],
                        "stratum": case["stratum"],
                        "response": reply,
                        "seconds": round(time.time() - t0, 2),
                        "metrics": pair,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "case_id": case["case_id"],
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
            if i % 10 == 0 or i == len(cases):
                log(f"  {i}/{len(cases)} cases done")
    finally:
        vlm_models.unload(bundle)

    refs = [c["expert_report"] for c in cases]
    hyps = []
    aligned_idx = []
    by_id = {c["case_id"]: c for c in cases}
    for j, row in enumerate(results):
        if "response" not in row:
            continue
        hyps.append(row["response"])
        aligned_idx.append(j)
    bert = bertscore_f1([by_id[results[j]["case_id"]]["expert_report"] for j in aligned_idx], hyps) if hyps else None
    if bert is not None:
        for j, val in zip(aligned_idx, bert):
            results[j]["metrics"]["bertscore_f1"] = val
    else:
        log("BERTScore skipped (pip install bert-score to enable)")

    summary = summarize_model(results)
    log(
        f"  mean BLEU-4={summary['bleu4_mean']:.3f}  "
        f"ROUGE-L={summary['rouge_l_f1_mean']:.3f}  "
        f"BERTScore={summary['bertscore_f1_mean'] if summary['bertscore_f1_mean'] is not None else 'n/a'}"
    )
    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "repo_id": spec.repo_id,
        "summary": summary,
        "cases": results,
    }


def write_summary_csv(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_scored", "BLEU-4", "ROUGE-L", "BERTScore-F1"])
        for key, entry in payload.get("models", {}).items():
            s = entry.get("summary") or {}
            w.writerow(
                [
                    entry.get("display_name", key),
                    s.get("n_scored", ""),
                    "" if s.get("bleu4_mean") is None else f"{s['bleu4_mean']:.4f}",
                    "" if s.get("rouge_l_f1_mean") is None else f"{s['rouge_l_f1_mean']:.4f}",
                    "" if s.get("bertscore_f1_mean") is None else f"{s['bertscore_f1_mean']:.4f}",
                ]
            )


def write_rater_sheet(cases: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "stratum",
                "image_path",
                "readability_1to5",
                "clinical_usefulness_1to5",
                "factual_accuracy_1to5",
                "notes",
            ]
        )
        for c in cases:
            w.writerow([c["case_id"], c["stratum"], c["image_path"], "", "", "", ""])


def _model_complete(entry: dict[str, Any]) -> bool:
    s = entry.get("summary") or {}
    return s.get("n_scored", 0) > 0 and "error" not in entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="extract images and lock the full-case list")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(vlm_models.MODEL_REGISTRY))
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    if args.prepare or not CASE_LIST.exists():
        listing = prepare_dataset()
    else:
        listing = load_case_list()

    cases = listing["eval_cases"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rater_sheet(cases, args.out_dir / "rater_sheet_all.csv")

    if args.dry_run:
        print(
            f"RQ3 dry-run dataset=PR-Reports n_eval={len(cases)} "
            f"n_holdout={listing['n_holdout']} seed={SEED} "
            f"models={args.models} prompt_chars={len(REPORT_PROMPT)} "
            f"max_new_tokens={args.max_new_tokens} "
            f"strata={listing['eval_stratum_counts']}"
        )
        print("first_case", cases[0]["case_id"], cases[0]["stratum"])
        print("case_list", CASE_LIST)
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"rq3_pr_all_{stamp}"
    log_path = args.out_dir / f"{stem}.log"
    resume_payload = None
    skip_models: list[str] = []
    if args.resume_from:
        resume_payload = json.loads(args.resume_from.read_text())
        stem = args.resume_from.stem
        log_path = args.resume_from.with_suffix(".log")
        skip_models = [k for k, e in resume_payload.get("models", {}).items() if _model_complete(e)]
        args.models = [k for k in args.models if k not in skip_models]

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    payload = resume_payload or {
        "timestamp": stamp,
        "dataset": "PR-Reports",
        "task": "RQ3_expert_report_comparison",
        "n_cases": len(cases),
        "case_ids": [c["case_id"] for c in cases],
        "seed": SEED,
        "prompt": REPORT_PROMPT,
        "metrics": ["bleu4", "rouge_l_f1", "bertscore_f1"],
        "bertscore_model": BERTSCORE_MODEL,
        "stratum_counts": listing["eval_stratum_counts"],
        "models": {},
    }
    if skip_models:
        log(f"resume; skipping completed models: {skip_models}")

    log(
        f"RQ3 PR-Reports n={len(cases)} models={args.models} "
        f"max_new_tokens={args.max_new_tokens} seed={SEED}"
    )
    out_json = args.out_dir / f"{stem}.json"
    if not args.models:
        log("nothing left to run")
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        log("RQ3_COMPARISON_DONE")
        return 0

    for key in args.models:
        try:
            payload["models"][key] = evaluate_model(
                key,
                cases,
                prompt=REPORT_PROMPT,
                max_new_tokens=args.max_new_tokens,
                log=log,
            )
        except Exception as e:
            log(f"MODEL {key} FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            payload["models"][key] = {"model_key": key, "error": f"{type(e).__name__}: {e}"}
        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        write_summary_csv(payload, args.out_dir / f"{stem}_summary.csv")
        log(f"checkpoint -> {out_json.name}")

    log("=== RQ3 means (all PR-Reports expert reports) ===")
    log(f"{'model':<22}{'BLEU4':>8}{'ROUGE-L':>8}{'BERT-F1':>8}")
    for key, entry in payload["models"].items():
        s = entry.get("summary") or {}
        b = s.get("bleu4_mean")
        r = s.get("rouge_l_f1_mean")
        t = s.get("bertscore_f1_mean")
        log(
            f"{entry.get('display_name', key):<22}"
            f"{(b or 0):>8.3f}"
            f"{(r or 0):>8.3f}"
            f"{(t if t is not None else float('nan')):>8.3f}"
        )
    log("RQ3_COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
