#!/bin/bash
set -eExuo

cd ~/aws-neuron-samples/torch-neuronx/training/mnist_mlp

# Single worker CPU training
python train_cpu.py

# Single worker MLP training
python train.py

# Multi-worker data-parallel MLP training
torchrun --nproc_per_node=2 train_torchrun.py

# Single-worker MLP evaluation
cd ~/aws-neuron-samples/torch-neuronx/training/mnist_mlp
python eval.py
python eval_using_trace.py