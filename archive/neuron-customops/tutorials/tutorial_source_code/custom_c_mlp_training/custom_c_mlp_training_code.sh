#!/bin/bash
set -eExuo

# Install requirements
pip install regex
pip install ninja

cd ~/aws-neuron-samples/torch-neuronx/training/customop_mlp

cd pytorch
python build.py

python train_cpu.py

cd ..
cd neuron
python build.py

python train.py