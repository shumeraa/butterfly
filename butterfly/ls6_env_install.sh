

git clone <REPOSITORY_URL>
cd butterfly

conda env create -f environment/environment.yml

conda activate viunit_env

# OLD DIRECTIONS BELOW

#conda create -n viunit_env python=3.10
#conda activate viunit_env

#conda install -c conda-forge gcc_linux-64=9.5.0
#conda install -c conda-forge compilers
#conda install -c conda-forge glib

#export CC=gcc
#export CXX=g++

#conda install nvidia/label/cuda-12.1.0::cuda-toolkit

#export PATH=$CONDA_PREFIX/bin:$PATH
#export CUDA_HOME=$CONDA_PREFIX
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
#alias ld="ld -L $LD_LIBRARY_PATH"

#conda install \
#  -c pytorch -c nvidia \
#  "pytorch::pytorch=2.5.1" \
#  "pytorch::torchvision=0.20.1" \
#  "pytorch::torchaudio=2.5.1" \
#  "pytorch-cuda=12.1"

#conda install pip
#unset PYTHONPATH

# This is where we have slightly different directions
#python -m pip install -r butterfly/req_base.txt

#python -m pip install vllm==0.6.4.post1

#python -m pip install astunparse

# Need a GPU
#python -m pip install flash-attn --no-build-isolation
