#!/bin/bash
set -eExuo

# Install packages and clone transformers
export HF_VER=4.44.0
pip install -U transformers==$HF_VER datasets evaluate scikit-learn
cd ~/
git clone https://github.com/huggingface/transformers --branch v$HF_VER
cd ~/transformers/examples/pytorch/text-classification