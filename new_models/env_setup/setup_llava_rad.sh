#!/usr/bin/env bash
# Create the llava-rad conda env + code. Run once, on a VM, when told to start.
set -euo pipefail
ENV=llava-rad; ROOT=/home/s222393187/Dental/new_models/repos
conda create -y -n "$ENV" python=3.10
git clone https://github.com/microsoft/LLaVA-Rad "$ROOT/LLaVA-Rad" || true
conda run -n "$ENV" pip install -e "$ROOT/LLaVA-Rad"
conda run -n "$ENV" pip install open_clip_torch torch torchvision   # BiomedCLIP-CXR tower
echo "weights: HF_TOKEN=... python download_new_models.py --group custom  (fetches microsoft/llava-rad)"
echo "llava-rad env ready"
