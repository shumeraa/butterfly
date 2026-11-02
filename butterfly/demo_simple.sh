#!/bin/bash
#SBATCH -J viunit-demo_simple
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -o viunit-demo-simple.%j.out
#SBATCH -e viunit-demo-simple.%j.err

source ~/.bashrc

PROJECT=/work/11808/shumera/ls6/projects/butterfly
cd "$PROJECT"

# Activate environment
source "$PROJECT/butterfly/start_env.sh"

echo "=== Environment ==="
which python
python --version
nvidia-smi

# Verify installation
python -c "import flash_attn; print('flash-attn installed')"

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

echo "=== Running ViUniT demo ==="
python /work/11808/shumera/ls6/projects/butterfly/butterfly/demo_simple.py \
  --num-tests 2 \
  --steps 25 \
  --output-dir plots/simple_unit_tests

echo "=== Finished ==="
