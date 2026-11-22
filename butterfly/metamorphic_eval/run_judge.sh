#!/usr/bin/env bash
#SBATCH -J viz-judge
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 02:00:00
#SBATCH -o /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-judge.%j.out
#SBATCH -e /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/logs/viz-judge.%j.out
#SBATCH --cpus-per-task=8

# Runs the question-only LLM judge on the 303 merged questions (one call per question), then the
# code-side rules turn its flags into labels.json (transform verdicts + expected answers).
# First (login node):  python butterfly/metamorphic_eval/merge_dataset.py
# Submit:              sbatch butterfly/metamorphic_eval/run_judge.sh
# Resubmitting the same command resumes an unfinished run (same settings required).
# Overrides: QUESTIONS, RUN_DIR (e.g. a 10-question test file), MODEL, TENSOR_PARALLEL_SIZE.

ROOT=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval
export WORK=/work/11808/shumera/ls6
export REPO_ROOT=/work/11808/shumera/ls6/projects/butterfly   # Slurm runs a spool copy of this script
QUESTIONS="${QUESTIONS:-$ROOT/merged/questions.jsonl}"
RUN_DIR="${RUN_DIR:-$ROOT/merged/judge_run}"
mkdir -p "$ROOT/logs"
cd "$REPO_ROOT" || exit 1

[[ -f "$QUESTIONS" ]] || { echo "Missing $QUESTIONS; run merge_dataset.py first." >&2; exit 1; }
args=(--questions "$QUESTIONS" --run-dir "$RUN_DIR" --max-model-len 4096)
[[ -f "$RUN_DIR/run_metadata.json" ]] && args+=(--resume)

bash "$REPO_ROOT/butterfly/llm_judge/run_inference.sh" "${args[@]}"
