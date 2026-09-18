#!/usr/bin/env python3
"""Preflight for the new-VLM jobs (no GPU/torch). Datasets, output dirs, registry, readiness."""
from __future__ import annotations
import importlib.util, re
from pathlib import Path
ROOT = Path("/home/s222393187/Dental")
def has(m): return importlib.util.find_spec(m) is not None
def cached(repo):
    d = Path.home()/".cache/huggingface/hub"/("models--"+repo.replace("/","--"))
    return bool(repo) and d.exists()
print("== datasets ==")
for label,p in [("Tufts radiographs",ROOT/"Tufts_Dental_Database/Radiographs"),
                ("Tufts bbox",ROOT/"Tufts_Dental_Database/Segmentation/teeth_bbox.json"),
                ("DENTEX train json",ROOT/"DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"),
                ("DENTEX val json",ROOT/"DENTEX/validation_triple.json"),
                ("PR-Reports parquet",ROOT/"PR-Reports/train-00000-of-00001.parquet")]:
    print(f"  [{'OK ' if p.exists() else 'MISSING'}] {label}")
reg=(ROOT/"vlm_models.py").read_text(); keys=re.findall(r'"([a-z0-9_]+)": ModelSpec', reg)
print("== default new roster (hf_it2t) ==")
        "qwen25_vl":"Qwen/Qwen2.5-VL-7B-Instruct","internvl25":"OpenGVLab/InternVL2_5-8B",
        "llava_onevision":"llava-hf/llava-onevision-qwen2-7b-ov-hf"}
deps_ok = has("transformers") and has("accelerate")
for k,repo in roster.items():
    wt=cached(repo)
    print(f"  {k:<16} registered={'Y' if k in keys else 'N'}  weights_cached={'Y' if wt else 'N'}  "
          f"deps(transformers,accelerate)={'Y' if deps_ok else 'N'}  -> {'READY' if (wt and deps_ok and k in keys) else 'needs download'}")
import os
env=os.environ.get("CUSTOM_VLMS_ENV","custom-vlms")
env_ok=(Path.home()/".conda/envs"/env).exists()
print(f"== legacy custom (shared env '{env}' via subprocess workers) ==")
for k in ["llava_rad","med_flamingo","radfm"]:
    per=os.environ.get(f"{k.upper()}_ENV",env); ok=(Path.home()/".conda/envs"/per).exists()
    print(f"  {k:<16} registered={'Y' if k in keys else 'N'}  env={per} exists={'Y' if ok else 'N'}  -> {'ready' if ok else 'run env_setup/setup_custom_vlms.sh'}")
print("\nNOTE: 'needs download' clears after download_new_models.py; then run one smoke test per model at start.")
