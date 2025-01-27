#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/tp_zero1_llama_hf_pretrain
ln -sf ~/neuronx-distributed/examples/training/llama/training_utils.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/modeling_llama_nxd.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/get_dataset.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/requirements.txt ./

python3 -m pip install -r requirements.txt
