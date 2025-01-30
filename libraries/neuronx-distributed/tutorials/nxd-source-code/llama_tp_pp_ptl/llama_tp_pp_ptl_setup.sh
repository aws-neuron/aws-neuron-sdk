#!/bin/bash
set -eExuo

cd ~/neuronx-distributed/examples/training/llama/lightning
ln -sf ~/neuronx-distributed/examples/training/llama/get_dataset.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/lr.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/modeling_llama_nxd.py ./
ln -sf ~/neuronx-distributed/examples/training/llama/requirements.txt ./
ln -sf ~/neuronx-distributed/examples/training/llama/requirements_ptl.txt ./
ln -sf ~/neuronx-distributed/examples/training/llama/training_utils.py ./

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_ptl.txt  # Currently we're supporting Lightning version 2.1.0