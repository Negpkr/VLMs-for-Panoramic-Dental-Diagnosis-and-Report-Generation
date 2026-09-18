import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
repo="google/medgemma-4b-it"
proc=AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(repo, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
R="Tufts_Dental_Database/Radiographs"
prompt=("This is a panoramic dental X-ray. List which teeth are missing. "
        "Answer 'Missing teeth: <numbers or None>'.")
describe="Describe this dental X-ray in one short sentence."
for img_name in ["1.JPG","1000.JPG"]:
    img=Image.open(f"{R}/{img_name}").convert("RGB")
    for tag,ptxt in [("DENTAL",prompt),("DESCRIBE",describe)]:
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":ptxt}]}]
        inp=proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        pv=inp.get("pixel_values")
        pvinfo=(tuple(pv.shape), round(float(pv.float().mean()),4)) if pv is not None else "NO pixel_values"
        inp={k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}
        if "pixel_values" in inp: inp["pixel_values"]=inp["pixel_values"].to(model.dtype)
        torch.manual_seed(0)
        out=model.generate(**inp, max_new_tokens=60, do_sample=False)
        rep=proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"[{img_name}/{tag}] pixel_values={pvinfo}")
        print(f"    reply: {rep!r}")
