#!/usr/bin/env bash
# Dedicated conda env for MedGemma-4B-it (Gemma-3). Kept separate from the main
# dental-llava env because that env's transformers raises the
# or_mask_function / and_mask_function error on Gemma-3. transformers 4.53.x has
# stable Gemma-3 / MedGemma support and the older masking interface.
set -euo pipefail
ENV="${MEDGEMMA_ENV:-medgemma-env}"
conda create -y -n "$ENV" python=3.10
# torch>=2.6 is REQUIRED: transformers 4.53 Gemma-3 masking calls
# create_causal_mask with or_mask_function, which errors on torch<2.6.
# cu124 wheels run fine on the working Ada/L40S nodes.
conda run -n "$ENV" pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
conda run -n "$ENV" pip install \
    "transformers==4.53.2" "accelerate>=1.0" "huggingface_hub>=0.24" \
    pillow numpy sentencepiece protobuf
conda run -n "$ENV" python -c "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "Env '$ENV' ready. MedGemma weights are already cached at ~/.cache/huggingface/hub/models--google--medgemma-4b-it"
echo "Smoke: conda run -n $ENV python -m new_models.workers.worker_medgemma --image <img> --prompt-file <txt> --max-new-tokens 64"
