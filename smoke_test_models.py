#!/usr/bin/env python3
"""Load each registered VLM one at a time and run a single dental inference."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/s222393187/Dental")
import vlm_models  # noqa: E402
from vlm_comparison import COMPACT_PROMPT, find_image  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "1000"


def main() -> int:
    image_path = find_image(CASE)
    if image_path is None:
        print(f"no image for case {CASE}")
        return 1
    print(f"image: {image_path}\n")

    failures = []
    for key in vlm_models.MODEL_REGISTRY:
        spec = vlm_models.MODEL_REGISTRY[key]
        print("=" * 72)
        print(f"{key}  ({spec.display_name})")
        print("=" * 72)
        bundle = None
        try:
            bundle = vlm_models.load_model(key, local_files_only=True)
            peak_load = torch.cuda.max_memory_allocated() / 1e9
            print(f"  loaded in {bundle.load_seconds:.1f}s | device={bundle.device} | {peak_load:.1f} GB")

            t0 = time.time()
            reply = vlm_models.generate(
                bundle, image_path, COMPACT_PROMPT, seed_key=f"{key}:{CASE}"
            )
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  inference {dt:.1f}s | peak VRAM {peak:.1f} GB")
            print(f"  --- reply ({len(reply)} chars) ---")
            print("  " + reply[:600].replace("\n", "\n  "))
            print()
        except Exception as e:
            import traceback

            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(key)
        finally:
            vlm_models.unload(bundle)
            torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    print("FAILED:" if failures else "ALL_MODELS_OK", failures or "")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
