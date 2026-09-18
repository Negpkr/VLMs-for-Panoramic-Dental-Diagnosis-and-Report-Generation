#!/usr/bin/env python3
"""LLaVA-Rad inference worker. Runs in the `llava-rad` conda env; prints ONLY the
reply to stdout (all logging goes to stderr). Called by vlm_models via `conda run`.

VERIFY at integration against https://github.com/microsoft/LLaVA-Rad and
https://huggingface.co/microsoft/llava-rad (model_base, conv template, tokens).
"""
import argparse, sys
def log(*a): print(*a, file=sys.stderr, flush=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--model-path", default="microsoft/llava-rad")
    ap.add_argument("--model-base", default="lmsys/vicuna-7b-v1.5")  # TODO confirm
    ap.add_argument("--conv-mode", default="v1")                      # TODO confirm
    a = ap.parse_args()
    prompt = open(a.prompt_file, encoding="utf-8").read()
    try:
        import torch
        from PIL import Image
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
    except Exception as e:
        log(f"IMPORT_ERROR: {type(e).__name__}: {e}"); sys.exit(3)
    name = get_model_name_from_path(a.model_path)
    tok, model, image_processor, _ = load_pretrained_model(a.model_path, a.model_base, name)
    img = Image.open(a.image).convert("RGB")
    # LLaVA-Rad's process_images non-pad branch calls image_processor(...) which is not
    # callable for the BiomedCLIP tower; use the HF .preprocess API directly.
    pv = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
    # BiomedCLIP processor returns pixel_values as a list of per-image tensors.
    if isinstance(pv, (list, tuple)):
        pv = torch.stack([p if hasattr(p, "dim") else torch.as_tensor(p) for p in pv])
    if pv.dim() == 3:
        pv = pv.unsqueeze(0)
    img_t = pv.to(model.device, dtype=torch.float16)
    conv = conv_templates[a.conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    text = conv.get_prompt()
    input_ids = tokenizer_image_token(text, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids, images=img_t,
                             do_sample=False, max_new_tokens=a.max_new_tokens, use_cache=True)
    reply = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    print(reply)
if __name__ == "__main__": main()
