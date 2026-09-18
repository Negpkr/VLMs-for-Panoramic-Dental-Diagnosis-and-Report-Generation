#!/usr/bin/env python3
"""MedGemma-4B-it inference worker (env `medgemma-env`: torch>=2.6 + transformers 4.53).

Runs out-of-process because Gemma-3 masking needs torch>=2.6, which the main
`dental-llava` env (torch 2.5.1) does not have. Gemma-3 also needs bfloat16
(fp16 produces NaN logits -> multinomial error).

Two modes:
  * --serve : PERSISTENT. Load the model once, print a JSON readiness line to
    stdout, then read one JSON request per line from stdin
    ({"image","prompt","max_new_tokens"}) and print one JSON reply per line
    ({"reply": ...} or {"error": ...}). Used by vlm_models' persistent bridge so
    the model is loaded ONCE for the whole run instead of per case.
  * one-shot (--image/--prompt-file): load, generate once, print reply. Handy for
    manual testing.

Only protocol JSON (serve) or the reply (one-shot) goes to stdout; every library
log / progress bar is forced to stderr.
"""
import argparse, hashlib, json, sys

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def _resize(img):
    from PIL import Image
    w, h = img.size
    if max(w, h) <= 672:
        return img
    if w >= h:
        return img.resize((672, int(h * 672 / w)), Image.Resampling.LANCZOS)
    return img.resize((int(w * 672 / h), 672), Image.Resampling.LANCZOS)

def load(repo_id):
    """Load model+processor once. All noisy output goes to stderr."""
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):   # keep stdout clean for the protocol
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            log("WARNING: CUDA not available; running on CPU")
        model.eval()
    log(f"loaded medgemma on {next(model.parameters()).device}")
    return model, processor

def run_once(model, processor, image_path, prompt, max_new_tokens):
    import contextlib
    import torch
    from PIL import Image
    with contextlib.redirect_stdout(sys.stderr):
        img = _resize(Image.open(image_path).convert("RGB"))
        # Gemma-3/MedGemma binds the image only when it is embedded IN the message
        # content and tokenized via the chat template (the separate
        # processor(text, images=[...]) path used for Qwen/LLaVA-OneVision leaves
        # the vision features unattended here -> identical output for every image).
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokenizer = getattr(processor, "tokenizer", None)
        pad_id = getattr(model.generation_config, "pad_token_id", None)
        if pad_id is None and tokenizer is not None:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        # Deterministic per-case seed (parent seeds in-process models the same way).
        seed = int(hashlib.md5((str(image_path) + prompt).encode()).hexdigest()[:8], 16) % (2**31)
        torch.manual_seed(seed)
        gen_kwargs = dict(max_new_tokens=int(max_new_tokens), do_sample=True,
                          temperature=0.4, top_p=0.9)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        reply_ids = out[0][input_len:]
        if tokenizer is not None:
            reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        else:
            reply = processor.decode(reply_ids, skip_special_tokens=True).strip()
    return reply

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--repo-id", default="google/medgemma-4b-it")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()

    if a.serve:
        try:
            model, processor = load(a.repo_id)
        except Exception as e:
            # Signal load failure on stdout so the parent's readline sees it.
            sys.stdout.write(json.dumps({"error": f"load failed: {type(e).__name__}: {e}"}) + "\n")
            sys.stdout.flush()
            sys.exit(3)
        sys.stdout.write(json.dumps({"ready": True}) + "\n")
        sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                reply = run_once(model, processor, req["image"], req["prompt"],
                                 req.get("max_new_tokens", a.max_new_tokens))
                sys.stdout.write(json.dumps({"reply": reply}) + "\n")
            except Exception as e:
                sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
            sys.stdout.flush()
        return

    # one-shot
    prompt = open(a.prompt_file, encoding="utf-8").read()
    model, processor = load(a.repo_id)
    print(run_once(model, processor, a.image, prompt, a.max_new_tokens))

if __name__ == "__main__":
    main()
