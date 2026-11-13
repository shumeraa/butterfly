
#!/bin/bash

# Model and dataset paths
export PRETRAINED_MODELS_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/models
export GQA_IMAGE_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa/images
export WINOGROUND_IMAGE_PATH=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/winoground/data/images
export COCO_VAL2017=/work/11808/shumera/ls6/butterfly_shared_resources/datasets/coco/val2017

export PYTHONPATH="/work/11808/shumera/ls6/projects/butterfly:${PYTHONPATH:-}"
