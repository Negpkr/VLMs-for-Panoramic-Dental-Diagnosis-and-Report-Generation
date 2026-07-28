#!/usr/bin/env python3
"""Four-way missing-teeth comparison: LLaVA-1.5 vs LLaVA-Med vs HuatuoGPT-Vision vs DentVLM.

Ground-truth extraction, prompt and answer parsing are kept identical to
llava_evaluation.ipynb so the numbers line up with the earlier LLaVA-only run.

Usage:
    python vlm_comparison.py --cases 50 --models llava llava_med huatuogpt_vision dentvlm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_models  # noqa: E402

DATA_ROOT = Path("/home/s222393187/Dental/Tufts_Dental_Database")
PANOS_DIR = DATA_ROOT / "Radiographs"
TEETH_BBOX_JSON = DATA_ROOT / "Segmentation" / "teeth_bbox.json"
RESULTS_DIR = Path("/home/s222393187/Dental/Results/model_comparison")

# Shared prompt for the cross-model comparison. It has to be short: LLaVA-Med
# answers with an immediate EOS on the long notebook prompt, while all four
# models produce parseable output with this one.
COMPACT_PROMPT = (
    "This is a panoramic dental X-ray. Teeth are numbered 1-32 "
    "(1-16 upper jaw, 17-32 lower jaw).\n"
    "Identify which teeth are missing.\n"
    "Answer in exactly this format:\n"
    "Missing teeth: <comma-separated numbers, or None>\n"
    "Total missing: <count>\n"
    "Confidence: High/Medium/Low"
)

# The original prompt from llava_evaluation.ipynb, kept for reproducing the
# LLaVA-only results.
MISSING_TEETH_PROMPT = (
    "You are a dental radiologist. Look at this panoramic X-ray and identify missing teeth.\n\n"
    "Teeth are numbered 1-32:\n"
    "- Teeth 1-16: Upper jaw (right to left)\n"
    "- Teeth 17-32: Lower jaw (left to right)\n\n"
    "Look for gaps or empty spaces where teeth should be.\n\n"
    "Answer in this exact format:\n"
    "Missing teeth: [list tooth numbers, e.g., 3, 14, 19]\n"
    "Total missing: [count]\n"
    "Confidence: High/Medium/Low\n\n"
    "Examples:\n"
    "- No missing: Missing teeth: None, Total missing: 0, Confidence: High\n"
    "- Some missing: Missing teeth: 3, 14, 19, Total missing: 3, Confidence: High\n"
    "- Many missing: Missing teeth: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, "
    "17, 18, 19, 20, Total missing: 20, Confidence: High\n\n"
    "Be thorough - look at the entire dental arch."
)


def recover_json_array(path: Path) -> list[dict]:
    """Read as many complete records as possible from a JSON array.

    The Tufts annotation files in this copy are truncated at exactly 256 KB, so
    ``json.load`` fails outright. Decoding record-by-record keeps the complete
    objects that precede the cut instead of discarding the whole file.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    start = raw.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    idx = start + 1
    items: list[dict] = []
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, idx = decoder.raw_decode(raw, idx)
        except ValueError:
            break  # hit the truncation point
        items.append(obj)
    return items


class GroundTruthExtractor:
    """Missing teeth per case, derived from teeth_bbox.json.

    Records are keyed by their ``External ID`` (e.g. ``"53.JPG"``). The original
    notebook keyed them by list position instead, which does not correspond to
    the radiograph filenames.
    """

    def __init__(self, bbox_json: Path = TEETH_BBOX_JSON):
        self.truncated = False
        try:
            records = json.loads(bbox_json.read_text())
        except json.JSONDecodeError:
            records = recover_json_array(bbox_json)
            self.truncated = True
        if isinstance(records, dict):
            records = list(records.values())

        self.by_case: dict[str, Any] = {}
        for record in records:
            external_id = record.get("External ID")
            if external_id:
                self.by_case[Path(external_id).stem] = record

    def extract_missing_teeth(self, case_id: str) -> list[int]:
        record = self.by_case.get(case_id)
        if not record:
            return []
        detected: set[int] = set()
        for obj in record.get("Label", {}).get("objects", []):
            for n in re.findall(r"\d+", str(obj.get("title", ""))):
                if 1 <= int(n) <= 32:
                    detected.add(int(n))
        return sorted(set(range(1, 33)) - detected)

    def available_cases(self) -> list[str]:
        """Cases that have real annotations, in numeric order."""
        return sorted(self.by_case, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))


class MissingTeethParser:
    """Same lenient parser used for the published LLaVA numbers.

    ``loose`` collects every 1-32 integer in the reply, ``strict`` reads only the
    "Missing teeth:" line. Loose is the default so results stay comparable.
    """

    tooth_pattern = re.compile(r"\b([1-9]|[12][0-9]|3[0-2])\b")
    missing_line = re.compile(r"Missing teeth:\s*([^\n]+)", re.IGNORECASE)
    confidence_pattern = re.compile(r"Confidence:\s*(High|Medium|Low)", re.IGNORECASE)

    def __init__(self, max_missing_teeth: int = 28, mode: str = "loose"):
        self.max_missing_teeth = max_missing_teeth
        self.mode = mode

    def extract_missing_teeth(self, text: str) -> list[int]:
        source = text
        if self.mode == "strict":
            match = self.missing_line.search(text)
            if not match:
                return []
            source = match.group(1)
            if re.search(r"\bnone\b", source, re.IGNORECASE):
                return []
        teeth = sorted({int(t) for t in self.tooth_pattern.findall(source) if 1 <= int(t) <= 32})
        return teeth[: self.max_missing_teeth]

    def confidence(self, text: str) -> str:
        match = self.confidence_pattern.search(text)
        return match.group(1).title() if match else "Medium"


def find_image(case_id: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = PANOS_DIR / f"{case_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_is_readable(path: Path) -> bool:
    """309 radiographs in this copy are zero-byte or otherwise undecodable."""
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def usable_cases(gt: "GroundTruthExtractor") -> list[str]:
    """Cases that have both recovered annotations and a decodable radiograph."""
    out = []
    for case_id in gt.available_cases():
        path = find_image(case_id)
        if path is not None and image_is_readable(path):
            out.append(case_id)
    return out


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Image-level and per-tooth binary metrics, matching the notebook."""
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n_cases": 0}

    y_true_img, y_pred_img = [], []
    y_true_tooth, y_pred_tooth = [], []

    for r in ok:
        pred = set(r["predicted_teeth"])
        gt = set(r["ground_truth_teeth"])
        y_true_img.append(1 if gt else 0)
        y_pred_img.append(1 if pred else 0)
        for tooth in range(1, 33):
            y_true_tooth.append(1 if tooth in gt else 0)
            y_pred_tooth.append(1 if tooth in pred else 0)

    def block(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    return {
        "n_cases": len(ok),
        "n_failed": len(results) - len(ok),
        "image_level": block(y_true_img, y_pred_img),
        "per_tooth": block(y_true_tooth, y_pred_tooth),
        "mean_predicted_count": sum(len(r["predicted_teeth"]) for r in ok) / len(ok),
        "mean_ground_truth_count": sum(len(r["ground_truth_teeth"]) for r in ok) / len(ok),
        "mean_seconds_per_case": sum(r["seconds"] for r in ok) / len(ok),
    }


def evaluate_model(
    model_key: str,
    case_ids: list[str],
    gt: GroundTruthExtractor,
    parser: MissingTeethParser,
    *,
    prompt: str = COMPACT_PROMPT,
    load_in_4bit: bool = False,
    log=print,
) -> dict[str, Any]:
    """Load one model, score every case, then release the GPU."""
    spec = vlm_models.MODEL_REGISTRY[model_key]
    log(f"=== {spec.display_name} ({spec.repo_id}) ===")

    bundle = None
    results: list[dict[str, Any]] = []
    try:
        bundle = vlm_models.load_model(model_key, load_in_4bit=load_in_4bit)
        log(f"loaded in {bundle.load_seconds:.1f}s on {bundle.device}")

        for i, case_id in enumerate(case_ids, 1):
            image_path = find_image(case_id)
            if image_path is None:
                results.append({"case_id": case_id, "error": "image not found"})
                continue
            t0 = time.time()
            try:
                reply = vlm_models.generate(
                    bundle,
                    image_path,
                    prompt,
                    seed_key=f"{model_key}:{case_id}",
                )
                predicted = parser.extract_missing_teeth(reply)
                results.append(
                    {
                        "case_id": case_id,
                        "response": reply,
                        "predicted_teeth": predicted,
                        "ground_truth_teeth": gt.extract_missing_teeth(case_id),
                        "confidence": parser.confidence(reply),
                        "seconds": time.time() - t0,
                    }
                )
            except Exception as e:  # keep going; one bad case shouldn't kill the run
                results.append({"case_id": case_id, "error": f"{type(e).__name__}: {e}"})
                log(f"  case {case_id} failed: {type(e).__name__}: {e}")

            if i % 10 == 0 or i == len(case_ids):
                log(f"  {i}/{len(case_ids)} cases done")
            vlm_models.free_vram()
    finally:
        vlm_models.unload(bundle)

    metrics = compute_metrics(results)
    log(f"  per-tooth F1={metrics.get('per_tooth', {}).get('f1_score', 0):.3f}")
    return {
        "model_key": model_key,
        "display_name": spec.display_name,
        "repo_id": spec.repo_id,
        "metrics": metrics,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases", type=int, default=0, help="number of cases to evaluate (0 = all usable)"
    )
    ap.add_argument("--models", nargs="+", default=list(vlm_models.MODEL_REGISTRY))
    ap.add_argument("--parse-mode", choices=["loose", "strict"], default="loose")
    ap.add_argument("--prompt", choices=["compact", "notebook"], default="compact")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    prompt = COMPACT_PROMPT if args.prompt == "compact" else MISSING_TEETH_PROMPT

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.out_dir / f"comparison_{stamp}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    gt = GroundTruthExtractor()
    parser = MissingTeethParser(mode=args.parse_mode)

    all_cases = usable_cases(gt)
    case_ids = all_cases if args.cases <= 0 else all_cases[: args.cases]
    if gt.truncated:
        log(
            f"WARNING teeth_bbox.json is truncated: only {len(gt.by_case)} of ~1000 "
            f"annotations recovered; {len(all_cases)} of those have a readable radiograph"
        )
    log(f"cases={len(case_ids)} models={args.models} parse_mode={args.parse_mode}")

    payload = {
        "timestamp": stamp,
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "parse_mode": args.parse_mode,
        "prompt_name": args.prompt,
        "prompt": prompt,
        "models": {},
    }

    for key in args.models:
        try:
            payload["models"][key] = evaluate_model(
                key,
                case_ids,
                gt,
                parser,
                prompt=prompt,
                load_in_4bit=args.load_in_4bit,
                log=log,
            )
        except Exception as e:
            log(f"MODEL {key} FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            payload["models"][key] = {"model_key": key, "error": f"{type(e).__name__}: {e}"}

        out_json = args.out_dir / f"comparison_{stamp}.json"
        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)
        log(f"checkpoint -> {out_json.name}")

    log("=== SUMMARY (per-tooth) ===")
    log(f"{'model':<24}{'F1':>8}{'Prec':>8}{'Recall':>8}{'Acc':>8}")
    for key, entry in payload["models"].items():
        pt = entry.get("metrics", {}).get("per_tooth")
        if not pt:
            log(f"{key:<24}{'ERROR':>8}")
            continue
        log(
            f"{entry['display_name']:<24}{pt['f1_score']:>8.3f}{pt['precision']:>8.3f}"
            f"{pt['recall']:>8.3f}{pt['accuracy']:>8.3f}"
        )
    log("COMPARISON_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
