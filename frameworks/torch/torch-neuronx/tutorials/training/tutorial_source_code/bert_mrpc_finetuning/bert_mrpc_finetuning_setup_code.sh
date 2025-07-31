#!/bin/bash
set -eExuo

# Install packages and clone transformers
export HF_VER=4.53.2
export ACC_VER=1.9.0
export DATA_VER=4.0.0
export EVAL_VER=0.4.5
pip install -U transformers==$HF_VER accelerate==$ACC_VER datasets==$DATA_VER evaluate==$EVAL_VER scikit-learn
cd ~/
git clone https://github.com/huggingface/transformers --branch v$HF_VER
cd ~/transformers/examples/pytorch/text-classification
