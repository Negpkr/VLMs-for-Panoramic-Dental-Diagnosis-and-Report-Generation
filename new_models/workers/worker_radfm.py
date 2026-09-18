#!/usr/bin/env python3
"""RadFM inference worker (env `radfm`). Prints ONLY the reply to stdout.
RadFM has a bespoke MultiModality model + image-token tokenizer. VERIFY against
https://github.com/chaoyi-wu/RadFM (Quick_demo/test.py) and set --radfm-src / weights."""
import argparse, sys, os
def log(*a): print(*a, file=sys.stderr, flush=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--radfm-src", default="new_models/repos/RadFM/Quick_demo")
    ap.add_argument("--ckpt", default="new_models/repos/RadFM/pytorch_model.bin")  # from pytorch_model.zip
    a = ap.parse_args()
    prompt = open(a.prompt_file, encoding="utf-8").read()
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        sys.path.append(a.radfm_src)
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM     # TODO confirm path
        from Dataset.multi_dataset_test_for_close import combine_and_preprocess  # TODO confirm helper
        from transformers import LlamaTokenizer
    except Exception as e:
        log(f"IMPORT_ERROR: {type(e).__name__}: {e}"); sys.exit(3)
    # NOTE: RadFM loads a LLaMA tokenizer with added image-placeholder tokens (<image>..),
    # builds vision_x from 3D-padded image tensors, then model.generate. Replicate their
    # Quick_demo/test.py exactly here once the repo + checkpoint are present.
    log("RadFM worker scaffold: complete the demo wiring against the cloned repo before use.")
    sys.exit(4)
if __name__ == "__main__": main()
