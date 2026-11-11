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

# Text inference smoke test, not a benchmark suite. Requires two A100
# 40GB/80GB GPUs, with vLLM BF16 inference and tensor parallelism of 2.
# No quantization or CPU offload.
#
# On the HPC, load your site's Python 3.12 / CUDA modules first, if needed.
# Prepare on a node with internet (allow ~60GB for weights plus dependencies):
#   bash test_qwen38_27b.sh --setup
#   bash test_qwen38_27b.sh --download
# Keep both files together. Submit from their directory:
#   sbatch test_qwen38_27b.sh
# To submit from elsewhere, export QWEN_SCRIPT_DIR=/path/to/both/files first.
# If required by your cluster, supply --account=YOUR_ACCOUNT to sbatch.
#
# Optional environment overrides (export before running/submitting):
#   QWEN_VENV=/path/to/venv    HF_HOME=/shared/scratch/huggingface
#   MODEL_ID=/path/to/local/model    MODEL_REVISION=commit_hash
#   RESULTS_DIR=/path/to/results     MAX_NEW_TOKENS=128
#   HF_HUB_OFFLINE=1                 # after --download, for offline workers
# Setup/download work from any directory; job results go in the submit dir.
# Model configuration: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
# vLLM parallelism: https://docs.vllm.ai/en/latest/serving/parallelism_scaling/

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
export MODEL_REVISION="${MODEL_REVISION:-main}"
QWEN_VENV="${QWEN_VENV:-${HOME}/.venvs/qwen38-test}"
PYTHON="${QWEN_VENV}/bin/python"
# Slurm spools the shell script, so find its Python companion in the submit
# directory. Set QWEN_SCRIPT_DIR when submitting from a different directory.
SCRIPT_DIR="${QWEN_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}"
PYTHON_SCRIPT="$SCRIPT_DIR/test_qwen38_27b.py"

case "${1:-}" in
    --setup)
        "${PYTHON_BOOTSTRAP:-python3}" -m venv "$QWEN_VENV"
        "$PYTHON" -m pip install --upgrade pip
        # Let vLLM install its matching PyTorch/CUDA dependencies.
        "$PYTHON" -m pip install --upgrade vllm huggingface_hub
        "$PYTHON" -m pip freeze > "$QWEN_VENV/qwen38-requirements.txt"
        echo "Environment ready: $QWEN_VENV"
        exit 0
        ;;
    --download|"") ;;
    *) echo "Usage: $0 [--setup|--download]" >&2; exit 2 ;;
esac

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing environment: $QWEN_VENV. Run: bash $0 --setup" >&2
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Missing Python script: $PYTHON_SCRIPT. Set QWEN_SCRIPT_DIR to its directory." >&2
    exit 1
fi

if [[ "${1:-}" == --download ]]; then
    "$PYTHON" "$PYTHON_SCRIPT" --download
    exit 0
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export RESULTS_DIR="${RESULTS_DIR:-$PWD/qwen38-results/${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
mkdir -p "$RESULTS_DIR"
echo "Started $(date -Is) on $(hostname); results: $RESULTS_DIR"
nvidia-smi

exec "$PYTHON" "$PYTHON_SCRIPT"
