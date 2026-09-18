# New-VLM evaluation jobs (Tufts / DENTEX / RQ3) — 8 models

Runs the three existing tasks on 8 new VLMs. Paper results untouched; new jobs
write to `Results/{model_comparison,dentex_comparison,rq3_reports}_new`.
Scheduler: Slurm `sbatch`. STATUS: **prepared, NOT started.**

## Roster
### Group A — in-process, HF-native `hf_it2t` (AutoModelForImageTextToText)
| key | model | repo_id |
|-----|-------|---------|
| oralgpt_omni | OralGPT-Omni-7B | `OralGPT/OralGPT-Omni-7B-Instruct` (verify id) |
| medgemma | MedGemma-4B-it | `google/medgemma-4b-it` (GATED: license+HF_TOKEN) |
| qwen25_vl | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| internvl25 | InternVL2.5-8B | `OpenGVLab/InternVL2_5-8B` |
| llava_onevision | LLaVA-OneVision-7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |

### Group B — subprocess workers, ONE shared env `custom-vlms` (per-model split only if a pin conflicts)
| key | model | env | worker | needs |
|-----|-------|-----|--------|-------|
| llava_rad | LLaVA-Rad | `llava-rad` | `new_models/workers/worker_llava_rad.py` | LLaVA-Rad repo + microsoft/llava-rad + BiomedCLIP-CXR |
| med_flamingo | Med-Flamingo | `med-flamingo` | `worker_med_flamingo.py` | open_flamingo + LLaMA-7B base + med-flamingo/model.pt |
| radfm | RadFM | `radfm` | `worker_radfm.py` | RadFM repo + pytorch_model.zip (~14B); finish demo wiring |

`vlm_models.py` handles both: Group A loads in the `dental-llava` process;
Group B is dispatched by `generate()` via `conda run -n <env> python -m <worker>`
(image path + prompt in, reply text out). If a Group-B env is missing,
`load_model` raises a clear error — no silent failure. The three harnesses are
unchanged and can score all 8 models in one run.

## Start sequence (ONLY when told)
1. `export HF_TOKEN=...` and accept the MedGemma license on HF.
2. Group A weights: `python download_new_models.py --download --group hf_it2t`
3. Group B (ONE shared env): `bash new_models/env_setup/setup_custom_vlms.sh`
   then `python download_new_models.py --download --group custom`
   (finish `worker_radfm.py` against the cloned Quick_demo). If RadFM's old
   `transformers` pin conflicts, split only it: `bash new_models/env_setup/setup_radfm.sh`
   and `export RADFM_ENV=radfm` — the other two stay in `custom-vlms`.
4. `python new_models/preflight_check.py`   (inside dental-llava env)
5. Smoke test one image per model (Group B workers print IMPORT_ERROR/￼exit codes if unwired).
6. Submit: `sbatch runs/sbatch_tufts_new.sh` / `sbatch runs/sbatch_dentex_new.sh` / `sbatch runs/sbatch_rq3_new.sh`
   (split across the 4th GPU with `--export=ALL,MODELS="radfm"` etc.)

## Files
- `runs/run_{tufts,dentex,rq3}_new.sh`, `runs/sbatch_{...}_new.sh`, `runs/vm{1..4}_*_new.sh`
- `download_new_models.py` (--group hf_it2t|custom|all), `new_models/preflight_check.py`
- `new_models/workers/worker_*.py`, `new_models/env_setup/setup_*.sh`, `new_models/repos/` (clones land here)

## Verify-at-integration flags
- OralGPT-Omni exact HF id; MedGemma gating; InternVL trust_remote_code.
- Group-B worker inner code (model_base/conv for LLaVA-Rad; checkpoint name + prompt for Med-Flamingo; full RadFM demo) must be validated against upstream once weights/code are present.
