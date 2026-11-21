#!/bin/bash
# Run all stages for ONE run directory. Usage: viz_program_eval_stages.sh <config.yaml> <out_dir>
#  1. ViUniT pipeline (programs, unit tests, images, unit-test scoring, selection)  -> <out>/data.p
#  2. real-image pass: every candidate program on the real image                    -> <out>/real_pass/data.p
#  3. export                                                                        -> <out>/dataset.jsonl
# Each stage resumes from its saved data.p, so rerunning after a timeout is safe.
# Expects the environment from butterfly/environment/activate.sh to be active.
set -eo pipefail
CONFIG=$1; OUT=$2
cd /work/11808/shumera/ls6/projects/butterfly
# vLLM tensor-parallel size = visible GPUs and must divide 32 heads, so use one GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CONFIG_NAMES=config_codellama,viz_program_eval   # xvlm disabled (package not installed)
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-$HOME/ray_${SLURM_JOB_ID:-local}}
mkdir -p "$OUT/real_pass" "$RAY_TEMP_DIR"

R=0; [ -f "$OUT/data.p" ] && R=1
python -u generate.py --config_path "$CONFIG" --reload $R
cp -f "$CONFIG" "$OUT/config_used.yaml"
git rev-parse HEAD > "$OUT/git_commit.txt"

# stage 2: same config, no unit tests, new output dir, data copied from stage 1
sed -e 's#^do_unit_test:.*#do_unit_test: False#' -e "s#^output_dir:.*#output_dir: \"$OUT/real_pass\"#" "$CONFIG" > "$OUT/real_pass/config.yaml"
R=0; [ -f "$OUT/real_pass/data.p" ] && R=1
python -u generate.py --config_path "$OUT/real_pass/config.yaml" --data_path "$OUT/data.p" --reload $R

python butterfly/export_viz_program_eval.py "$OUT"
touch "$OUT/DONE"
rm -rf "$RAY_TEMP_DIR"
echo "=== $OUT finished ==="
