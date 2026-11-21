#!/bin/bash
# Run on a GPU node:  bash butterfly/environment/test_env.sh
PROJECT=/work/11808/shumera/ls6/projects/butterfly
cd "$PROJECT"
source "$PROJECT/butterfly/environment/activate.sh"
python -c "import torch, vllm, accelerate, transformers as t; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',t.__version__)"
python -u "$PROJECT/butterfly/environment/test_env.py" "$@"
