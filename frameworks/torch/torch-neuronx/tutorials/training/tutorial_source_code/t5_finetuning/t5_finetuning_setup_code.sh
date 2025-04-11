#!/bin/bash
set -eExuo

# Install packages and clone transformers
export HF_VER=4.26.0
pip install -U transformers==$HF_VER datasets evaluate scikit-learn rouge_score pandas==1.4.0
cd ~/
git clone https://github.com/huggingface/transformers --branch v$HF_VER
cd ~/transformers/examples/pytorch/summarization