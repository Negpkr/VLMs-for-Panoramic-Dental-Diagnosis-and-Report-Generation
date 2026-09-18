#!/usr/bin/env python3
"""LLaVA-Rad inference worker (env `custom-vlms`).

Modes:
  * --serve : PERSISTENT. Load the model ONCE, print a JSON readiness line, then
    read one JSON request per line from stdin ({"image","prompt","max_new_tokens"})
    and print one JSON reply per line ({"reply": ...} / {"error": ...}).
  * one-shot (--image/--prompt-file): load, generate once, print reply.

Only protocol JSON (serve) or the reply (one-shot) goes to stdout; all model /
library noise is redirected to stderr. Base = lmsys/vicuna-7b-v1.5; image
preprocessing uses the BiomedCLIP tower's .preprocess (returns a list of
per-image tensors, stacked here); generate() does NOT accept image_sizes on this
LLaVA-Rad base, so it is omitted.
"""
import argparse, json, sys, contextlib
def log(*a): print(*a, file=sys.stderr, flush=True)

def load(model_path, model_base):
    with contextlib.redirect_stdout(sys.stderr):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        name = get_model_name_from_path(model_path)
        tok, model, image_processor, _ = load_pretrained_model(model_path, model_base, name)
    log(f"loaded llava_rad on {next(model.parameters()).device}")
    return tok, model, image_processor

def run_once(bundle, image_path, prompt, max_new_tokens, conv_mode):
    import torch
    from PIL import Image
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    tok, model, image_processor = bundle
    with contextlib.redirect_stdout(sys.stderr):
        img = Image.open(image_path).convert("RGB")
        pv = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        if isinstance(pv, (list, tuple)):
            pv = torch.stack([p if hasattr(p, "dim") else torch.as_tensor(p) for p in pv])
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        img_t = pv.to(model.device, dtype=torch.float16)
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        input_ids = tokenizer_image_token(text, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        with torch.inference_mode():
            out = model.generate(input_ids, images=img_t, do_sample=False,
                                 max_new_tokens=int(max_new_tokens), use_cache=True)
        # LLaVA-Rad can emit ids outside the tokenizer's SentencePiece range
        # (added/image tokens, or the negative image placeholder) which make
        # decode raise "piece id is out of range". Mask invalid ids to EOS first.
        seq = out[0]
        # Decode ONLY the generated continuation: out includes the input prompt,
        # so decoding the whole thing echoes the Vicuna system prompt. Slicing off
        # the input length also drops the negative image placeholder token.
        if seq.shape[0] > input_ids.shape[1]:
            seq = seq[input_ids.shape[1]:]
        vocab = tok.vocab_size
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        seq = seq.masked_fill((seq < 0) | (seq >= vocab), eos)
        reply = tok.decode(seq, skip_special_tokens=True).strip()
    return reply

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--prompt-file")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--model-path", default="microsoft/llava-rad")
    ap.add_argument("--model-base", default="lmsys/vicuna-7b-v1.5")
    ap.add_argument("--conv-mode", default="v1")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    try:
        bundle = load(a.model_path, a.model_base)
    except Exception as e:
        if a.serve:
            sys.stdout.write(json.dumps({"error": f"load failed: {type(e).__name__}: {e}"}) + "\n"); sys.stdout.flush()
        else:
            log(f"LOAD_ERROR: {type(e).__name__}: {e}")
        sys.exit(3)
    if a.serve:
        sys.stdout.write(json.dumps({"ready": True}) + "\n"); sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                reply = run_once(bundle, req["image"], req["prompt"],
                                 req.get("max_new_tokens", a.max_new_tokens), a.conv_mode)
                sys.stdout.write(json.dumps({"reply": reply}) + "\n")
            except Exception as e:
                sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
            sys.stdout.flush()
        return
    prompt = open(a.prompt_file, encoding="utf-8").read()
    print(run_once(bundle, a.image, prompt, a.max_new_tokens, a.conv_mode))

if __name__ == "__main__":
    main()
