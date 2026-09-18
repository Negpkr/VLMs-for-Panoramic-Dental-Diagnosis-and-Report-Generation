#!/usr/bin/env python3
"""Augment the RQ3 report-comparison results with KL divergence, JS divergence,
ROUGE-1 F1 and ROUGE-2 F1, placed alongside the existing BLEU-4 / ROUGE-L /
BERTScore metrics.

Reads the finished RQ3 run (per-case model responses) plus the case list that
holds the expert reference reports, computes the four extra metrics per case,
writes them back into the results JSON, and refreshes the summary CSV.

Usage:
    python rq3_add_distribution_metrics.py \
        --results Results/rq3_reports/rq3_pr_all_20260820_040310.json \
        --cases   runs/rq3_pr_reports_all.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr

# Same tokenizer used by rq3_report_comparison.py so the new metrics line up
# with the BLEU-4 / ROUGE-L numbers already in the file.
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Additive (Lidstone) smoothing weight applied over the union vocabulary of each
# reference/hypothesis pair. Keeps KL finite when a token appears on only one
# side; small enough not to swamp the observed counts.
SMOOTH_ALPHA = 1e-3


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference: str, hypothesis: str, n: int) -> float:
    """ROUGE-N F1 on n-gram overlap (matches the ROUGE-L convention in the run)."""
    ref = _ngrams(tokenize(reference), n)
    hyp = _ngrams(tokenize(hypothesis), n)
    ref_total = sum(ref.values())
    hyp_total = sum(hyp.values())
    if ref_total == 0 or hyp_total == 0:
        return 0.0
    overlap = sum((ref & hyp).values())
    if overlap == 0:
        return 0.0
    prec = overlap / hyp_total
    rec = overlap / ref_total
    return 2 * prec * rec / (prec + rec)


def _smoothed_distributions(reference: str, hypothesis: str):
    """Unigram probability distributions for ref (P) and hyp (Q) over the union
    vocabulary, with additive smoothing so both are strictly positive."""
    ref_counts = Counter(tokenize(reference))
    hyp_counts = Counter(tokenize(hypothesis))
    vocab = sorted(set(ref_counts) | set(hyp_counts))
    if not vocab:
        return None, None
    p = np.array([ref_counts.get(w, 0) + SMOOTH_ALPHA for w in vocab], dtype=float)
    q = np.array([hyp_counts.get(w, 0) + SMOOTH_ALPHA for w in vocab], dtype=float)
    p /= p.sum()
    q /= q.sum()
    return p, q


def kl_divergence(reference: str, hypothesis: str) -> float | None:
    """KL(P_expert || Q_model) in bits. Lower = model token distribution is
    closer to the expert's. Undefined if there are no tokens."""
    p, q = _smoothed_distributions(reference, hypothesis)
    if p is None:
        return None
    return float(np.sum(rel_entr(p, q)) / np.log(2))


def js_divergence(reference: str, hypothesis: str) -> float | None:
    """Jensen-Shannon divergence in bits, symmetric and bounded [0, 1]."""
    p, q = _smoothed_distributions(reference, hypothesis)
    if p is None:
        return None
    dist = jensenshannon(p, q, base=2)  # JS distance = sqrt(JS divergence)
    return float(dist**2)


def load_references(cases_path: Path) -> dict[str, str]:
    data = json.loads(cases_path.read_text())
    return {c["case_id"]: c["expert_report"] for c in data["eval_cases"]}


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def augment(results_path: Path, cases_path: Path) -> dict[str, Any]:
    payload = json.loads(results_path.read_text())
    refs = load_references(cases_path)

    for entry in payload.get("models", {}).values():
        rouge1, rouge2, kls, jsds = [], [], [], []
        for case in entry.get("cases", []):
            if "response" not in case or "metrics" not in case:
                continue
            ref = refs.get(case["case_id"], "")
            hyp = case["response"]
            r1 = rouge_n_f1(ref, hyp, 1)
            r2 = rouge_n_f1(ref, hyp, 2)
            kl = kl_divergence(ref, hyp)
            jsd = js_divergence(ref, hyp)
            case["metrics"]["rouge_1_f1"] = r1
            case["metrics"]["rouge_2_f1"] = r2
            case["metrics"]["kl_divergence"] = kl
            case["metrics"]["js_divergence"] = jsd
            rouge1.append(r1)
            rouge2.append(r2)
            if kl is not None:
                kls.append(kl)
            if jsd is not None:
                jsds.append(jsd)
        summary = entry.setdefault("summary", {})
        summary["rouge_1_f1_mean"] = mean(rouge1)
        summary["rouge_2_f1_mean"] = mean(rouge2)
        summary["kl_divergence_mean"] = mean(kls)
        summary["js_divergence_mean"] = mean(jsds)

    for m in ("rouge_1_f1", "rouge_2_f1", "kl_divergence", "js_divergence"):
        if m not in payload.get("metrics", []):
            payload.setdefault("metrics", []).append(m)
    return payload


def write_summary_csv(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "n_scored",
                "BLEU-4",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "BERTScore-F1",
                "KL-div",
                "JS-div",
            ]
        )
        for key, entry in payload.get("models", {}).items():
            s = entry.get("summary") or {}

            def fmt(v, nd=4):
                return "" if v is None else f"{v:.{nd}f}"

            w.writerow(
                [
                    entry.get("display_name", key),
                    s.get("n_scored", ""),
                    fmt(s.get("bleu4_mean")),
                    fmt(s.get("rouge_1_f1_mean")),
                    fmt(s.get("rouge_2_f1_mean")),
                    fmt(s.get("rouge_l_f1_mean")),
                    fmt(s.get("bertscore_f1_mean")),
                    fmt(s.get("kl_divergence_mean"), 3),
                    fmt(s.get("js_divergence_mean")),
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    args = ap.parse_args()

    payload = augment(args.results, args.cases)
    args.results.write_text(json.dumps(payload, indent=2))
    csv_path = args.results.with_name(args.results.stem + "_summary.csv")
    write_summary_csv(payload, csv_path)

    print(f"updated {args.results}")
    print(f"updated {csv_path}\n")
    header = f"{'model':<22}{'BLEU4':>8}{'R-1':>8}{'R-2':>8}{'R-L':>8}{'BERT':>8}{'KL':>8}{'JSD':>8}"
    print(header)
    for key, entry in payload["models"].items():
        s = entry.get("summary") or {}

        def g(name, default=float("nan")):
            v = s.get(name)
            return default if v is None else v

        print(
            f"{entry.get('display_name', key):<22}"
            f"{g('bleu4_mean'):>8.3f}"
            f"{g('rouge_1_f1_mean'):>8.3f}"
            f"{g('rouge_2_f1_mean'):>8.3f}"
            f"{g('rouge_l_f1_mean'):>8.3f}"
            f"{g('bertscore_f1_mean'):>8.3f}"
            f"{g('kl_divergence_mean'):>8.3f}"
            f"{g('js_divergence_mean'):>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
