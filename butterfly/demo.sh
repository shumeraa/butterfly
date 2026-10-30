#!/bin/bash
#SBATCH -J viunit-demo
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o viunit-demo.%j.out
#SBATCH -e viunit-demo.%j.err

set -euo pipefail

PROJECT=/work/11808/shumera/ls6/projects/butterfly
cd "$PROJECT"

# Activate environment
source "$PROJECT/butterfly/start_env.sh"

echo "=== Environment ==="
which python
python --version
nvidia-smi

# Install flash-attn if missing
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "=== Installing flash-attn ==="

    MAX_JOBS=4 python -m pip install \
        flash-attn \
        --no-build-isolation
fi

# Verify installation
python -c "import flash_attn; print('flash-attn installed')"

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

echo "=== Running ViUniT demo ==="
python -u "$PROJECT/butterfly/demo.py"

echo "=== Finished ==="