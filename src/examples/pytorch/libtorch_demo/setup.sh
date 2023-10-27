#!/bin/bash

# fail on error
set -e

# For torch-neuron
INF1_VERSION="1.13.1"

# For torch-neuronx
VERSION="1.13.1"

# Python setup
PYTHON=python3
PYTHON_VERSION=$($PYTHON --version | cut -f2 -d' ' | cut -f1,2 -d'.')

if [ "$PYTHON_VERSION" == "3.8" ] || [ "$PYTHON_VERSION" == "3.9" ] || [ "$PYTHON_VERSION" == "3.10" ]
then
    echo "Python version is '$PYTHON_VERSION'"
else
    echo "ERROR: No suitable version of Python found for libtorch demo. Current Python version is '$PYTHON_VERSION'."
    echo "Install Python 3.8, 3.9, or 3.10 and set the PYTHON environment variable as needed."
    exit 1
fi

OLD_TOOL_CHAIN=$($PYTHON -c \
    "from bert_neuronx.detect_instance import get_instance_type; print('inf1' in get_instance_type())")

if [ "$OLD_TOOL_CHAIN" == "True" ]; then
    VERSION=${INF1_VERSION}
    echo "- Detected inf1 - using version ${VERSION}"
else
    echo "- Detected inf2 or trn1 - using version ${VERSION}"
fi

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
chmod +x build.sh
./build.sh
popd
cp -f tokenizers_binding/tokenizer.json .

# setup torch
if [ ! -e "libtorch" ]; then
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip
    unzip libtorch-shared-with-deps-${VERSION}+cpu.zip
    rm -f libtorch-shared-with-deps-${VERSION}+cpu.zip
fi

# get libneuron_op.so and install into libtorch
if [ ! -e "venv" ]; then

    $PYTHON -m venv venv
    . venv/bin/activate
    pip install -U pip

    if [ "$OLD_TOOL_CHAIN" == "True" ]
    then
        # Install compiler from the old tool chain
        pip install torch-neuron~=${VERSION} neuron-cc tensorflow==1.15.5 --extra-index-url=https://pip.repos.neuron.amazonaws.com
    else
        pip install torch-neuronx~=${VERSION} --extra-index-url=https://pip.repos.neuron.amazonaws.com
    fi
    
    pip install --upgrade "transformers==4.34.1"
    python bert_neuronx/compile.py
    deactivate

    if [ "$OLD_TOOL_CHAIN" == "True" ]
    then
        cp -f $(find ./venv -name libtorchneuron.so | grep torch_neuron) libtorch/lib/
        cp -f $(find ./venv -name libnrt.so) libtorch/lib/
        cp -f $(find ./venv -name libnrt.so.1) libtorch/lib/        
    else
        cp -f $(find ./venv -name libtorchneuron.so | grep torch_neuronx) libtorch/lib/
    fi
fi

# compile example app
pushd example_app
chmod +x build.sh
./build.sh
popd

chmod +x run_tests.sh
echo "Successfully completed setup"
