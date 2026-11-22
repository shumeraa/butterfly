#!/usr/bin/env bash
#SBATCH -J viz-meta-eval
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 02:00:00
#SBATCH -o /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-meta-eval.%j.out
#SBATCH -e /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-meta-eval.%j.out
#SBATCH --cpus-per-task=8

# Metamorphic evaluation of the frozen candidate programs: for each question, render the transforms
# chosen from the judge's labels, run all 5 programs on them, score against the expected answers
# (ViUniT's execution/scoring path in generate.py), and select the best program.
# Order:  merge_dataset.py -> sbatch run_judge.sh -> prepare_eval.py -> sbatch this script
# Chunks are time-aware and resumable like gen_viz_program_eval_full.sh: resubmit until all have DONE.
# After the last chunk the job writes comparison.jsonl and summary.md (export_metamorphic_eval.py).
# Overrides: EVAL_DIR (default merged/metamorphic_eval), MERGED.

ROOT=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval
MERGED="${MERGED:-$ROOT/merged}"
EVAL_DIR="${EVAL_DIR:-$MERGED/metamorphic_eval}"
PROJECT=/work/11808/shumera/ls6/projects/butterfly   # Slurm runs a spool copy of this script
LIMIT=${LIMIT_SECONDS:-7200}; MARGIN=240   # set LIMIT_SECONDS if you submit with a shorter -t
START=$SECONDS
mkdir -p "$ROOT/logs"

source ~/.bashrc
cd "$PROJECT" || exit 1
source "$PROJECT/butterfly/environment/activate.sh"
[[ -f "$EVAL_DIR/eval_ids.txt" ]] || { echo "Run prepare_eval.py first ($EVAL_DIR/eval_ids.txt missing)" >&2; exit 1; }

# vLLM/ViperGPT use every visible GPU; use one. xvlm is not installed, so it is disabled.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CONFIG_NAMES=config_codellama,viz_program_eval
export RAY_TEMP_DIR=$HOME/ray_${SLURM_JOB_ID:-local}
mkdir -p "$RAY_TEMP_DIR"
nvidia-smi -L
python - <<'PY' || { echo 'Grounded-SAM imports failed in this environment' >&2; exit 1; }
import transformers, torch
print('transformers', transformers.__version__, 'cuda', torch.cuda.is_available())
from transformers import AutoModelForZeroShotObjectDetection, SamModel
PY

chunks=($(ls -d "$EVAL_DIR"/chunk_* | sort -t_ -k2 -n))
slowest=0
for c in "${chunks[@]}"; do
    [ -f "$c/DONE" ] && { echo "$(basename $c) already done"; continue; }
    left=$((LIMIT - (SECONDS-START) - MARGIN))
    need=$(( slowest > 0 ? slowest*11/10 : 1500 ))
    if [ "$left" -lt "$need" ]; then echo "stopping: ${left}s left, need ~${need}s for $(basename $c)"; break; fi
    mkdir -p "$c/run"
    R=0; [ -f "$c/run/data.p" ] && R=1
    t0=$SECONDS
    echo "=== $(basename $c) start, ${left}s left (reload=$R) ==="
    timeout "$left" python -u generate.py --config_path "$c/config.yaml" \
        --data_path "$c/input.p" --reload $R \
        || { echo "$(basename $c) incomplete (rc=$?); resubmit to resume"; break; }
    touch "$c/DONE"
    dur=$((SECONDS-t0)); [ "$dur" -gt "$slowest" ] && slowest=$dur
    echo "=== $(basename $c) done in ${dur}s ==="
done
rm -rf "$RAY_TEMP_DIR"

n_done=$(ls -d "$EVAL_DIR"/chunk_*/DONE 2>/dev/null | wc -l)
echo "chunks done: $n_done/${#chunks[@]}"
if [ "$n_done" -eq "${#chunks[@]}" ]; then
    python butterfly/metamorphic_eval/export_metamorphic_eval.py "$MERGED" "$EVAL_DIR"
fi
