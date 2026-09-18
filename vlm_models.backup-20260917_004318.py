#!/usr/bin/env python3
"""Unified loading and inference for the vision-language models compared on the
Tufts panoramic dental radiographs.

Each model is exposed through the same three calls so the evaluation pipeline
stays identical across architectures:

    spec = MODEL_REGISTRY["dentvlm"]
    bundle = load_model("dentvlm")
    text = generate(bundle, image, prompt)
    unload(bundle)
"""
from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Generation settings mirror the original LLaVA notebook so previously reported
# LLaVA numbers stay comparable with the new models.
GEN_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 250,
    "temperature": 0.4,
    "top_p": 0.92,
    "do_sample": True,
    "repetition_penalty": 1.15,
}


@dataclass
class ModelSpec:
    key: str
    display_name: str
    repo_id: str
    architecture: str  # "llava" or "qwen2_vl"
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "llava": ModelSpec(
        key="llava",
        display_name="LLaVA-1.5-7B",
        repo_id="llava-hf/llava-1.5-7b-hf",
        architecture="llava",
        notes="General-domain baseline used in the original evaluation.",
    ),
    "llava_med": ModelSpec(
        key="llava_med",
        display_name="LLaVA-Med-v1.5-7B",
        repo_id="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        architecture="llava",
        notes="HF-format conversion of microsoft/llava-med-v1.5-mistral-7b.",
    ),
    "huatuogpt_vision": ModelSpec(
        key="huatuogpt_vision",
        display_name="HuatuoGPT-Vision-7B",
        repo_id="FreedomIntelligence/HuatuoGPT-Vision-7B-hf",
        architecture="llava",
        notes="Medical VLM trained on PubMedVision; Qwen2 language tower.",
    ),
    "dentvlm": ModelSpec(
        key="dentvlm",
        display_name="DentVLM",
        repo_id="ZJU-AI4H/DentVLM",
        architecture="qwen2_vl",
        notes="Dental-specific VLM built on Qwen2-VL.",
    ),
}


@dataclass
class ModelBundle:
    spec: ModelSpec
    model: Any
    processor: Any
    device: str
    dtype: torch.dtype
    load_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resize_for_memory(image: Image.Image, max_size: int = 672) -> Image.Image:
    """Resize keeping aspect ratio if the largest side exceeds max_size."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w >= h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_h, new_w = max_size, int(w * max_size / h)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _quantization_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    key: str,
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    load_in_4bit: bool = False,
    local_files_only: bool = False,
) -> ModelBundle:
    """Load one model from the registry onto the visible GPU."""
    import time

    from transformers import AutoProcessor

    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[key]

    t0 = time.time()
    common = {
        "device_map": device_map,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    quant = _quantization_config(load_in_4bit)
    if quant is not None:
        common["quantization_config"] = quant
        common.pop("dtype")

    processor = AutoProcessor.from_pretrained(spec.repo_id, local_files_only=local_files_only)

    if spec.architecture == "llava":
        from transformers import AutoConfig, LlavaForConditionalGeneration

        # HuatuoGPT-Vision ships a processor_config without patch_size, which makes
        # the image-token expansion crash. Backfill it from the vision tower.
        if getattr(processor, "patch_size", None) is None:
            cfg = AutoConfig.from_pretrained(spec.repo_id, local_files_only=local_files_only)
            processor.patch_size = cfg.vision_config.patch_size

        model = LlavaForConditionalGeneration.from_pretrained(spec.repo_id, **common)
    elif spec.architecture == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(spec.repo_id, **common)
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")

    model.eval()
    device = str(next(model.parameters()).device)
    return ModelBundle(
        spec=spec,
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        load_seconds=time.time() - t0,
    )


def unload(bundle: ModelBundle | None) -> None:
    if bundle is None:
        return
    try:
        bundle.model.to("meta")
    except Exception:
        pass
    del bundle.model
    del bundle.processor
    free_vram()


def _build_inputs(bundle: ModelBundle, image: Image.Image, prompt: str):
    """Build processor inputs using each model's own chat template."""
    processor = bundle.processor

    if bundle.spec.architecture == "qwen2_vl":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[image], return_tensors="pt")

    # LLaVA-family models
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        if not isinstance(text, str):  # some templates return token ids
            raise TypeError("template returned non-string")
    except Exception:
        # HuatuoGPT-Vision has no processor-level template, only a tokenizer one,
        # so the image placeholder has to be inserted by hand.
        tokenizer = getattr(processor, "tokenizer", None)
        text = None
        if tokenizer is not None:
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"<image>\n{prompt}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = None
        if text is None:
            text = f"USER: <image>\n{prompt} ASSISTANT:"
    return processor(text=text, images=image, return_tensors="pt")


def _seed_for(case_key: str) -> int:
    return int(hashlib.md5(case_key.encode()).hexdigest()[:8], 16) % (2**31)


def generate(
    bundle: ModelBundle,
    image_path: Path | str,
    prompt: str,
    *,
    max_image_size: int = 672,
    seed_key: str | None = None,
    **gen_overrides: Any,
) -> str:
    """Run one image+prompt through the model and return only the reply text."""
    image = Image.open(image_path).convert("RGB")
    image = resize_for_memory(image, max_size=max_image_size)

    inputs = _build_inputs(bundle, image, prompt)
    inputs = {k: (v.to(bundle.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs = {**GEN_DEFAULTS, **gen_overrides}
    if not gen_kwargs.get("do_sample"):
        # Avoid transformers warnings about unused sampling args
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)

    tokenizer = getattr(bundle.processor, "tokenizer", None)
    pad_id = getattr(bundle.model.generation_config, "pad_token_id", None)
    if pad_id is None and tokenizer is not None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    # Deterministic per-case seed keeps repeated runs reproducible.
    torch.manual_seed(_seed_for(seed_key or str(image_path)))

    with torch.inference_mode():
        out = bundle.model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    reply_ids = out[0][input_len:]
    decoder = bundle.processor
    if tokenizer is not None:
        return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    return decoder.decode(reply_ids, skip_special_tokens=True).strip()


__all__ = [
    "GEN_DEFAULTS",
    "MODEL_REGISTRY",
    "ModelBundle",
    "ModelSpec",
    "free_vram",
    "generate",
    "load_model",
    "resize_for_memory",
    "unload",
]
