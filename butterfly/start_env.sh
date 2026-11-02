#!/bin/bash

#module load cuda/12.2

# Initialize Conda for non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate viunit_env

# Compiler configuration
export CC=gcc
export CXX=g++

# CUDA configuration
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

# Model and dataset paths
export PRETRAINED_MODELS_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/models
export GQA_IMAGE_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa/images
export WINOGROUND_IMAGE_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/winoground/data/images
export COCO_VAL2017=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/coco/val2017
export PYTHONPATH="/work/11808/shumera/ls6/projects/butterfly:${PYTHONPATH:-}"

export VLLM_ATTENTION_BACKEND=XFORMERS
