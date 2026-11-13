#!/bin/bash
#SBATCH -J qwen38-27b-test
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 00:25:00
#SBATCH -o qwen38-27b-test.%j.out
#SBATCH -e qwen38-27b-test.%j.err
#SBATCH --cpus-per-task=8

module load tacc-apptainer/1.4.1

WORK=/work/11808/shumera/ls6
IMAGE=$WORK/containers/vllm-0.18.0.sif
SCRIPT=/work/11808/shumera/ls6/projects/butterfly/butterfly/llm_judge/eval_tests/qwen_smoke.py

# Cache model files and compiled artifacts on /work.
mkdir -p "$WORK/hf_cache" "$WORK/vllm_cache"

export APPTAINERENV_HF_HOME="$WORK/hf_cache"
export APPTAINERENV_VLLM_CACHE_ROOT="$WORK/vllm_cache"
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Host GPUs ==="
nvidia-smi

echo "=== Container environment and GPU test ==="

apptainer exec --nv --cleanenv \
    --bind "$WORK:$WORK" \
    --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
    "$IMAGE" \
    /usr/bin/python3 -c '
import torch, transformers, vllm

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2

print("=== Testing GPU computation ===", flush=True)

for i in range(2):
    x = torch.randn(100, 100, device=f"cuda:{i}")
    result = (x @ x).sum().item()
    print(f"GPU {i} computation successful: {result}", flush=True)

print("CUDA GPU tests passed!", flush=True)
'

if [ $? -ne 0 ]; then
    echo "WARNING: GPU test failed; attempting inference anyway."
fi

echo "=== Running inference ==="

apptainer exec --nv --cleanenv \
    --bind "$WORK:$WORK" \
    --bind "$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" \
    "$IMAGE" \
    /usr/bin/python3 "$SCRIPT"

