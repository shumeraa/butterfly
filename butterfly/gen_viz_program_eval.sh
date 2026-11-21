#!/bin/bash
# Interactive/test run of the 10-question set (run on a GPU node, not via sbatch).
source ~/.bashrc
source /work/11808/shumera/ls6/projects/butterfly/butterfly/environment/activate.sh
bash /work/11808/shumera/ls6/projects/butterfly/butterfly/viz_program_eval_stages.sh \
  viunit_configs/GQA/eval/viz_program_eval_10.yaml \
  /work/11808/shumera/ls6/butterfly_shared_resources/datasets/viz_program_eval/run_10
