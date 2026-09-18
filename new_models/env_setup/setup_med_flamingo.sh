#!/usr/bin/env bash
set -euo pipefail
ENV=med-flamingo
conda create -y -n "$ENV" python=3.9
conda run -n "$ENV" pip install open-flamingo einops einops-exts torch torchvision huggingface_hub
echo "needs LLaMA-7B base access (huggyllama/llama-7b) + med-flamingo/model.pt checkpoint"
echo "med-flamingo env ready"
