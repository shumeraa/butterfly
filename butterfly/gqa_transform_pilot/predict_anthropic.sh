#!/bin/bash
#SBATCH -J gqa-transform-anthropic
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o gqa-transform-anthropic.%j.out
#SBATCH -e gqa-transform-anthropic.%j.err
set -e
source ~/.bashrc
PROJECT=${PROJECT:-/work/11808/shumera/ls6/projects/butterfly}
cd "$PROJECT"
source "$PROJECT/butterfly/start_env.sh"
set -u
# Prediction runs through the API; image export uses Pillow/NumPy on CPU.
export CUDA_VISIBLE_DEVICES=""
exec python butterfly/gqa_transform_pilot/predict_anthropic.py "$@"
