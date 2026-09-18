#!/usr/bin/env bash
# ONE shared conda env for the 3 legacy VLMs (LLaVA-Rad, Med-Flamingo, RadFM).
# Run once, on a VM, at start. If pip cannot co-resolve all three (RadFM pins an
# old transformers), split just the holdout into its own env and point the bridge
# at it, e.g.:  RADFM_ENV=radfm bash new_models/env_setup/setup_radfm.sh
set -euo pipefail
ENV="${CUSTOM_VLMS_ENV:-custom-vlms}"; ROOT=/home/s222393187/Dental/new_models/repos
conda create -y -n "$ENV" python=3.10
git clone https://github.com/microsoft/LLaVA-Rad "$ROOT/LLaVA-Rad" || true
git clone https://github.com/chaoyi-wu/RadFM     "$ROOT/RadFM"     || true
# Best-effort shared install. transformers ~4.37 satisfies LLaVA-Rad + open_flamingo;
# RadFM officially wants 4.28.1 and may need the RADFM_ENV split above.
conda run -n "$ENV" pip install "transformers==4.37.2" torch torchvision \
    open-flamingo einops einops-exts open_clip_torch huggingface_hub pillow numpy
conda run -n "$ENV" pip install -e "$ROOT/LLaVA-Rad" || echo "WARN: LLaVA-Rad editable install needs review"
echo "Shared env '$ENV' created. Weights: python download_new_models.py --download --group custom"
echo "If RadFM import fails on transformers 4.37, run setup_radfm.sh and export RADFM_ENV=radfm."
