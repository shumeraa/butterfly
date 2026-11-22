#!/bin/bash
#SBATCH -J render-xforms-cpu
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o render-xforms-cpu.%j.out
#SBATCH -e render-xforms-cpu.%j.out

# CPU-only: render every non-unsafe transformation for every LLM output in labels.json.
#   sbatch experiment/render_all_transformations_cpu.sbatch
# Override with env vars: LABELS, OUT_DIR, SEED. For a shorter queue wait use the
# development partition: sbatch -p development -t 02:00:00 experiment/render_all_transformations_cpu.sbatch
# Resumable: rerun and already-rendered images are skipped; failed ones are retried.
# Needs the Grounding DINO / SAM weights in HF_HOME (already cached from earlier runs).

REPO_ROOT="${REPO_ROOT:-/work/11808/shumera/ls6/projects/butterfly}"
LABELS="${LABELS:-$REPO_ROOT/experiment/runs/run_004/labels.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiment/all_transformations}"
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

python experiment/render_all_transformations.py --labels "$LABELS" --out-dir "$OUT_DIR" --seed "$SEED"
