#!/usr/bin/env python3
"""Download weights for the new-VLM roster. Does NOT run automatically.
    HF_TOKEN=... python download_new_models.py             # dry list
    HF_TOKEN=... python download_new_models.py --download   # fetch (accept MedGemma license first)
"""
from __future__ import annotations
import argparse, os, sys, time
# key: (repo_id, group, note)
REPOS = {
  # --- default new roster: HF-native image-text-to-text, load via AutoModelForImageTextToText ---
  "medgemma":       ("google/medgemma-4b-it",            "hf_it2t", "GATED: accept Health-AI license + HF_TOKEN"),
  "qwen25_vl":      ("Qwen/Qwen2.5-VL-7B-Instruct",      "hf_it2t", "ungated"),
  "internvl25":     ("OpenGVLab/InternVL2_5-8B",         "hf_it2t", "trust_remote_code; recent transformers"),
  "llava_onevision":("llava-hf/llava-onevision-qwen2-7b-ov-hf", "hf_it2t", "ungated"),
  # --- optional originals: need bespoke code + separate conda envs (not in default runs) ---
  "llava_rad":      ("microsoft/llava-rad",  "custom", "LLaVA-Rad codebase + BiomedCLIP-CXR"),
  "radfm":          ("chaoyi-wu/RadFM",      "custom", "RadFM code + pytorch_model.zip (~14B)"),
}
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--group", default="hf_it2t", choices=["hf_it2t","custom","all"], help="which group to fetch")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"{'model':<16}{'group':<9}{'repo_id'}")
    for k,(repo,grp,note) in REPOS.items():
        print(f"{k:<16}{grp:<9}{repo}   # {note}")
    if not a.download:
        print("\nDry list only. Re-run with --download (accept MedGemma license first).")
        return 0
    from huggingface_hub import login, snapshot_download
    if tok: login(token=tok, add_to_git_credential=False)
    fail=[]
    for k,(repo,grp,note) in REPOS.items():
        if a.group != "all" and grp != a.group: continue
        print(f"START {k} <- {repo}"); t0=time.time()
        try:
            p=snapshot_download(repo_id=repo, token=tok); print(f"OK {k} {(time.time()-t0)/60:.1f} min -> {p}")
        except Exception as e:
            print(f"FAIL {k}: {type(e).__name__}: {e}"); fail.append(k)
    print("DONE" if not fail else f"FAILED: {fail}")
    return 1 if fail else 0
if __name__ == "__main__":
    sys.exit(main())
