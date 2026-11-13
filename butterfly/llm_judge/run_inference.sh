#!/usr/bin/env bash
#SBATCH -J gqa-llm-judge
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A CCR26069
#SBATCH -t 00:25:00
#SBATCH -o gqa-llm-judge.%j.out
#SBATCH -e gqa-llm-judge.%j.err
#SBATCH --cpus-per-task=8

# Based on eval_tests/qwen_apptainer.sh. Submit from the repository root:
#   export WORK=/path/to/shared/work
#   sbatch butterfly/llm_judge/run_inference.sh --questions /work/11808/shumera/ls6/projects/butterfly/experiment/questions.jsonl --run-dir /work/11808/shumera/ls6/projects/butterfly/experiment/runs/run_001
# Set IMAGE for a different container and REPO_ROOT when submitting elsewhere.
# All arguments are forwarded; repeated Python options use the last value.

: "${WORK:?Set WORK to the execution-machine work directory containing containers/}"
IMAGE="${IMAGE:-$WORK/containers/vllm-0.18.0.sif}"
launch_dir="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -z "${REPO_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        # Slurm runs a spool copy, so BASH_SOURCE cannot locate the repository.
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
SCRIPT="$REPO_ROOT/butterfly/llm_judge/run_inference.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

[[ -f "$SCRIPT" ]] || { echo "Cannot find $SCRIPT; set REPO_ROOT to the repository root." >&2; exit 1; }
[[ -f "$IMAGE" ]] || { echo "Cannot find container $IMAGE; set IMAGE to its path." >&2; exit 1; }

if [[ -n "${APPTAINER_MODULE-tacc-apptainer/1.4.1}" ]] && type module >/dev/null 2>&1; then
    module load "${APPTAINER_MODULE-tacc-apptainer/1.4.1}"
fi
command -v apptainer >/dev/null || { echo 'apptainer is not available; load its module first.' >&2; exit 1; }

# Preserve the reference's shared caches and spawned vLLM workers.
cache_root="${CACHE_ROOT:-$WORK}"
mkdir -p "$cache_root/hf_cache" "$cache_root/vllm_cache" || exit $?
export APPTAINERENV_HF_HOME="$cache_root/hf_cache"
export APPTAINERENV_VLLM_CACHE_ROOT="$cache_root/vllm_cache"
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn

container=(apptainer exec --nv --cleanenv
    --bind "$WORK:$WORK"
    --bind "$REPO_ROOT:$REPO_ROOT"
    --bind "$launch_dir:$launch_dir"
    --bind "$cache_root:$cache_root"
    --pwd "$launch_dir"
    "$IMAGE")

echo '=== Host GPUs ==='
nvidia-smi

echo '=== Container environment and GPU test ==='
"${container[@]}" "$PYTHON_BIN" -c '
import torch, transformers, vllm

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is unavailable inside the container"
assert torch.cuda.device_count() >= 1, "No GPUs are visible"
for i in range(torch.cuda.device_count()):
    x = torch.randn(100, 100, device=f"cuda:{i}")
    result = (x @ x).sum().item()
    print(f"GPU {i} computation successful: {result}", flush=True)
' || exit $?

echo '=== Running inference ==='
exec "${container[@]}" "$PYTHON_BIN" "$SCRIPT" \
    --model "${MODEL:-Qwen/Qwen3.8-27B}" \
    --dtype "${DTYPE:-bfloat16}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    "$@"
