#!/bin/bash
#SBATCH -J render-all-xforms
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o render-all-xforms.%j.out
#SBATCH -e render-all-xforms.%j.out

# Render every non-unsafe transformation (hflip/vflip/grayscale/fill_color/occlude_bbox)
# for every LLM output in labels.json. Submit from the repo root:
#   sbatch experiment/render_all_transformations.sbatch
# Optional overrides (env vars): LABELS, OUT_DIR, SEED, SKIP_SAM=1
# fill_color/occlude_bbox need Grounding DINO + SAM, downloaded from Hugging Face on first use.
# If compute nodes lack internet, pre-download on the login node:
#   python -c "from huggingface_hub import snapshot_download as s; [s(r) for r in
#   ('IDEA-Research/grounding-dino-tiny','facebook/sam-vit-base')]"
# Resumable: rerun the same command and finished images are skipped.

REPO_ROOT="${REPO_ROOT:-/work/11808/shumera/ls6/projects/butterfly}"
LABELS="${LABELS:-$REPO_ROOT/experiment/runs/run_004/labels.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiment/all_transformations}"
SEED="${SEED:-42}"

source ~/.bashrc
source "$REPO_ROOT/butterfly/environment/activate.sh" || exit 1
export HF_HOME="${HF_HOME:-/work/11808/shumera/ls6/hf_cache}"
cd "$REPO_ROOT" || exit 1

nvidia-smi -L
python - <<'PY'
import transformers, torch
print('transformers', transformers.__version__, 'cuda', torch.cuda.is_available())
from transformers import AutoModelForZeroShotObjectDetection, SamModel  # fail early if unsupported
PY
[[ $? -eq 0 ]] || { echo 'GroundedSAM imports failed; use SKIP_SAM=1 or a newer transformers.' >&2; exit 1; }

extra=()
[[ -n "${SKIP_SAM:-}" ]] && extra+=(--skip-sam)
python experiment/render_all_transformations.py --labels "$LABELS" --out-dir "$OUT_DIR" \
    --seed "$SEED" "${extra[@]}"
