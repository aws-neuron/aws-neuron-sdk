#!/bin/bash
set -eExuo
#Run the setup script
cd aws-neuron-sdk/src/examples/pytorch
sudo apt install -y cargo 
cd libtorch_demo
chmod +x setup.sh && ./setup.sh

#Run sanity checks
./run_tests.sh bert_neuron_b6.pt

#Benchmark
./example-app bert_neuron_b6.pt