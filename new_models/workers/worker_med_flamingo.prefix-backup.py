#!/usr/bin/env python3
"""Med-Flamingo inference worker (env `med-flamingo`). Prints ONLY the reply to stdout.
VERIFY against https://github.com/snap-stanford/med-flamingo and
https://huggingface.co/med-flamingo/med-flamingo (checkpoint name, prompt format)."""
import argparse, sys
def log(*a): print(*a, file=sys.stderr, flush=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--llama", default="huggyllama/llama-7b")  # base LM; TODO confirm access
    a = ap.parse_args()
    prompt = open(a.prompt_file, encoding="utf-8").read()
    try:
        import torch
        from PIL import Image
        from open_flamingo import create_model_and_transforms
        from huggingface_hub import hf_hub_download
    except Exception as e:
        log(f"IMPORT_ERROR: {type(e).__name__}: {e}"); sys.exit(3)
    import contextlib
    # open_flamingo / DeepSpeed print init logs to stdout; keep stdout clean so the
    # bridge captures ONLY the reply. Redirect all of that noise to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai",
            lang_encoder_path=a.llama, tokenizer_path=a.llama, cross_attn_every_n_layers=4)
        ckpt = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        model = model.half().cuda().eval()
        tokenizer.padding_side = "left"
        img = Image.open(a.image).convert("RGB")
        vision_x = image_processor(img).unsqueeze(0).unsqueeze(0).unsqueeze(0).half().cuda()  # (b,T_img,F,C,H,W)
        text = f"<image>question: {prompt} answer:"
        lang_x = tokenizer([text], return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(vision_x=vision_x, lang_x=lang_x["input_ids"].cuda(),
                                 attention_mask=lang_x["attention_mask"].cuda(),
                                 max_new_tokens=a.max_new_tokens, num_beams=1)
        reply = tokenizer.decode(out[0], skip_special_tokens=True)
        reply = reply.split("answer:")[-1].strip()
    print(reply)
if __name__ == "__main__": main()
