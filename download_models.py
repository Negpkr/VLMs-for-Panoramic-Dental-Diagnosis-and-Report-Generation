#!/usr/bin/env python3
"""Download the medical/dental VLMs used in the four-way comparison.

Auth: set HF_TOKEN in the environment, or run `huggingface-cli login`.
Do not hard-code tokens in this file.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

from huggingface_hub import login, snapshot_download

REPOS = [
    ("llava", "llava-hf/llava-1.5-7b-hf"),
    ("llava_med", "chaoyinshe/llava-med-v1.5-mistral-7b-hf"),
    ("huatuogpt_vision", "FreedomIntelligence/HuatuoGPT-Vision-7B-hf"),
    ("dentvlm", "ZJU-AI4H/DentVLM"),
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        log("authenticated with HF_TOKEN from environment")
    else:
        log("no HF_TOKEN in env — using cached huggingface-cli login if present")

    failures = []
    for key, repo in REPOS:
        log(f"START {key} <- {repo}")
        t0 = time.time()
        try:
            path = snapshot_download(repo_id=repo, max_workers=4, token=token)
            log(f"OK {key} in {(time.time()-t0)/60:.1f} min -> {path}")
        except Exception as e:
            log(f"FAIL {key}: {type(e).__name__}: {e}")
            failures.append(key)

    if failures:
        log(f"DOWNLOADS_FAILED: {failures}")
        return 1
    log("ALL_DOWNLOADS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
