#!/bin/bash
#SBATCH -J render-examples
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o render-examples.%j.out
#SBATCH -e render-examples.%j.out

# CPU-only example run: render every transformation type (hflip, vflip, grayscale,
# fill_color, occlude_bbox) on at least N different images, plus each chosen
# image's other non-unsafe candidates. Submit from anywhere:
#   sbatch experiment/render_examples.sbatch
# Env overrides: N (default 3), FILL_STRENGTH (0.75), SEED (42), LABELS, OUT_DIR.
# Output: $OUT_DIR/images/<question_id>/{original.jpg,NN/000.png}, manifest.jsonl, summary.md
# Exits non-zero if some transformation type could not reach N successful images.

REPO_ROOT="${REPO_ROOT:-/work/11808/shumera/ls6/projects/butterfly}"
LABELS="${LABELS:-$REPO_ROOT/experiment/runs/run_004/labels.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiment/example_transformations}"
N="${N:-3}"
FILL_STRENGTH="${FILL_STRENGTH:-0.75}"
SEED="${SEED:-42}"

source ~/.bashrc
source "$REPO_ROOT/butterfly/environment/activate.sh" || exit 1
export HF_HOME="${HF_HOME:-/work/11808/shumera/ls6/hf_cache}"
export CUDA_VISIBLE_DEVICES=            # force CPU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT" || exit 1

python - <<'PY' || { echo 'GroundedSAM imports failed' >&2; exit 1; }
import torch, transformers
from transformers import AutoModelForZeroShotObjectDetection, SamModel
print('transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())
PY

python experiment/render_all_transformations.py --labels "$LABELS" --out-dir "$OUT_DIR" \
    --seed "$SEED" --fill-strength "$FILL_STRENGTH" --examples-per-transformation "$N"
