#!/bin/bash

# fail on error
set -e

# checkout tokenizers and apply neuron patch
if [ ! -e "tokenizers" ]; then
    git clone https://github.com/huggingface/tokenizers.git
    cp neuron.patch tokenizers/neuron.patch
    pushd tokenizers
    git checkout fc0a50a272c3fad4ae2f07b4a5bd84e106d1e266
    git am neuron.patch
    rm neuron.patch
    popd
fi

# build tests
pushd tokenizers_binding
chmod +x build_python.sh build.sh
./build_python.sh && ./build.sh
popd
cp -f tokenizers_binding/tokenizer.json .

# setup torch
if [ ! -e "libtorch" ]; then
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.7.1%2Bcpu.zip
    unzip libtorch-shared-with-deps-1.7.1+cpu.zip
    rm libtorch-shared-with-deps-1.7.1+cpu.zip
fi

# get libneuron_op.so and install into libtorch
if [ ! -e "venv" ]; then
    python3 -m venv venv
    . venv/bin/activate
    pip install -U pip
    pip install torch-neuron~=1.7 --extra-index-url=https://pip.repos.neuron.amazonaws.com
    deactivate
fi
cp -f $(find ./venv -name libneuron_op.so) libtorch/lib/

# compile example app
pushd example_app
chmod +x build.sh
./build.sh
popd

chmod +x run_tests.sh
echo "Successfully completed setup"
