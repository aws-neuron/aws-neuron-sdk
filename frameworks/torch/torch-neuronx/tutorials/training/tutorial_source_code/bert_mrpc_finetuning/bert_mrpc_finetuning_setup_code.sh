#!/bin/bash
set -eExuo

# Install packages and clone transformers
export HF_VER=4.52.0
export ACC_VER=1.7.0
pip install -U transformers==$HF_VER accelerate==$ACC_VER datasets evaluate scikit-learn
cd ~/
git clone https://github.com/huggingface/transformers --branch v$HF_VER
cd ~/transformers/examples/pytorch/text-classification
