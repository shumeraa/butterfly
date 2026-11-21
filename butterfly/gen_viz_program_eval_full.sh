#!/usr/bin/env bash
#SBATCH -J viz-prog-eval
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 02:00:00
#SBATCH -o /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-prog-eval.%j.out
#SBATCH -e /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-prog-eval.%j.out
#SBATCH --cpus-per-task=8

# Builds the ViUniT GQA reference set (303 questions = 3 per question group, seed 42).
# The questions are frozen in 13 chunks of ~23 (shuffled, so any set of finished chunks is a
# representative subsample). Each chunk runs ViUniT + the all-candidates real-image pass + export.
# The job works through unfinished chunks until the 2h limit approaches: it only starts a chunk if
# the time left covers the slowest finished chunk, and it stops a running chunk 4 min before the
# limit. A stopped chunk resumes from its saved data.p, so just resubmit the same command:
#   python butterfly/prep_viz_program_eval_full.py 13      # once, on a login node
#   sbatch butterfly/gen_viz_program_eval_full.sh          # repeat until all chunks have DONE
# Output: <DATA>/full/chunk_<k>/{dataset.jsonl,data.p,real_pass/data.p,DONE}

DATA=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval
PROJECT=/work/11808/shumera/ls6/projects/butterfly   # Slurm runs a spool copy of this script
LIMIT=$((2*3600)); MARGIN=240
N_CHUNKS=${N_CHUNKS:-13}
START=$SECONDS

source ~/.bashrc
cd "$PROJECT"
source "$PROJECT/butterfly/environment/activate.sh"
ls "$PROJECT/viunit_configs/GQA/eval/viz_program_eval_full/chunk_0.yaml" >/dev/null \
  || { echo "Run butterfly/prep_viz_program_eval_full.py first" >&2; exit 1; }

slowest=0
for k in $(seq 0 $((N_CHUNKS-1))); do
    OUT=$DATA/full/chunk_$k
    [ -f "$OUT/DONE" ] && { echo "chunk $k already done"; continue; }
    left=$((LIMIT - (SECONDS-START) - MARGIN))
    # first chunk of the job always starts if >=30 min remain; later ones need the slowest observed time + 10%
    need=$(( slowest > 0 ? slowest*11/10 : 1800 ))
    if [ "$left" -lt "$need" ]; then echo "stopping: ${left}s left, need ~${need}s for chunk $k"; break; fi
    mkdir -p "$OUT"
    t0=$SECONDS
    echo "=== chunk $k start, ${left}s left ==="
    timeout "$left" bash butterfly/viz_program_eval_stages.sh \
        "viunit_configs/GQA/eval/viz_program_eval_full/chunk_$k.yaml" "$OUT" \
        || { echo "chunk $k incomplete (rc=$?); resubmit to resume it"; break; }
    dur=$((SECONDS-t0)); [ "$dur" -gt "$slowest" ] && slowest=$dur
    echo "=== chunk $k done in ${dur}s ==="
done
echo "chunks done: $(ls -d $DATA/full/chunk_*/DONE 2>/dev/null | wc -l)/$N_CHUNKS"
