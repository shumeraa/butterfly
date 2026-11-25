#!/usr/bin/env bash
#SBATCH -J occ-bench
#SBATCH -p gpu-a100-dev
#SBATCH -A CCR26069
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600/logs/occ-bench.%j.out
# usage: sbatch butterfly/metamorphic_eval/benchmark_occlusion.sh [pipeline_dir]
cd "${PROJECT:-/work/11808/shumera/ls6/projects/butterfly}" || exit 1
source ~/.bashrc; source butterfly/environment/activate.sh
python -u butterfly/metamorphic_eval/benchmark_occlusion.py \
    "${1:-/work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval_600/gqa-simple-v5}" cuda
