#!/usr/bin/env python3
"""Med-Flamingo inference worker (env `custom-vlms`).

Modes:
  * --serve : PERSISTENT. Load ONCE, print a JSON readiness line, then read one
    JSON request per line from stdin and print one JSON reply per line.
  * one-shot (--image/--prompt-file): load, generate once, print reply.

open_flamingo / DeepSpeed print init logs to stdout, so all load/generate noise
is redirected to stderr; stdout carries only protocol JSON (serve) or the reply
(one-shot). Checkpoint = med-flamingo/med-flamingo model.pt; base LM = LLaMA-7B.
"""
import argparse, json, sys, contextlib
def log(*a): print(*a, file=sys.stderr, flush=True)

def load(llama):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama, tokenizer_path=llama, cross_attn_every_n_layers=4)
        ckpt = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model = model.half().cuda().eval()
        tokenizer.padding_side = "left"
    log("loaded med_flamingo on cuda")
    return model, image_processor, tokenizer

def run_once(bundle, image_path, prompt, max_new_tokens):
    import torch
    from PIL import Image
    model, image_processor, tokenizer = bundle
    with contextlib.redirect_stdout(sys.stderr):
        img = Image.open(image_path).convert("RGB")
        vision_x = image_processor(img).unsqueeze(0).unsqueeze(0).unsqueeze(0).half().cuda()  # (b,T_img,F,C,H,W)
        text = f"<image>question: {prompt} answer:"
        lang_x = tokenizer([text], return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(vision_x=vision_x, lang_x=lang_x["input_ids"].cuda(),
                                 attention_mask=lang_x["attention_mask"].cuda(),
                                 max_new_tokens=int(max_new_tokens), num_beams=1)
        reply = tokenizer.decode(out[0], skip_special_tokens=True)
        reply = reply.split("answer:")[-1].strip()
    return reply

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--prompt-file")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--llama", default="huggyllama/llama-7b")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    try:
        bundle = load(a.llama)
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
                                 req.get("max_new_tokens", a.max_new_tokens))
                sys.stdout.write(json.dumps({"reply": reply}) + "\n")
            except Exception as e:
                sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
            sys.stdout.flush()
        return
    prompt = open(a.prompt_file, encoding="utf-8").read()
    print(run_once(bundle, a.image, prompt, a.max_new_tokens))

if __name__ == "__main__":
    main()
