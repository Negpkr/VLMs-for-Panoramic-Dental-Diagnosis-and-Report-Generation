#!/usr/bin/env bash
set -euo pipefail
ENV=radfm; ROOT=/home/s222393187/Dental/new_models/repos
conda create -y -n "$ENV" python=3.10
git clone https://github.com/chaoyi-wu/RadFM "$ROOT/RadFM" || true
conda run -n "$ENV" pip install torch torchvision transformers==4.28.1 einops numpy pillow
echo "weights: download chaoyi-wu/RadFM pytorch_model.zip, unzip -> $ROOT/RadFM/pytorch_model.bin"
echo "radfm env ready (then finish worker_radfm.py against Quick_demo/test.py)"
