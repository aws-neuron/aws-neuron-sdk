#!/bin/bash
set -eExuo

# Install requirements
pip install regex
pip install ninja

cd ~/aws-neuron-samples/torch-neuronx/inference/customop_mlp

cd neuron
python build.py
python inference.py

cd ..
cd neuron-tcm
python build.py
python inference.py

cd ..
cd neuron-multicore
python build.py
python inference.py