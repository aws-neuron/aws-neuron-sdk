#!/bin/bash
set -eExuo

# Install requirements
cd ~/aws-neuron-samples/torch-neuronx/training/zero1_gpt2
python3 -m pip install -r requirements.txt

# Run precompile and training
neuron_parallel_compile bash run_clm.sh MIXED wikitext-103-raw-v1
bash run_clm.sh MIXED wikitext-103-raw-v1

