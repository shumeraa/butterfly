#!/bin/bash
#SBATCH -J gqa-transform-pilot
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:25:00
#SBATCH -o gqa-transform-pilot.%j.out
#SBATCH -e gqa-transform-pilot.%j.err
set -e
source ~/.bashrc
PROJECT=${PROJECT:-/work/11808/shumera/ls6/projects/butterfly}
cd "$PROJECT"
source "$PROJECT/butterfly/start_env.sh"
set -u
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python butterfly/gqa_transform_pilot/gqa_transform_pilot.py predict "$@"
