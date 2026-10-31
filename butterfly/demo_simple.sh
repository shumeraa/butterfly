#!/bin/bash
#SBATCH -J viunit-demo_simple
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -o viunit-demo-simple.%j.out
#SBATCH -e viunit-demo-simple.%j.err

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
python -u "$PROJECT/butterfly/demo_simple.py" \
    --image /work/11808/shumera/ls6/butterfly_shared_resources/datasets/coco/val2017/000000482319.jpg \
    --question "How many people are in the image?"

echo "=== Finished ==="