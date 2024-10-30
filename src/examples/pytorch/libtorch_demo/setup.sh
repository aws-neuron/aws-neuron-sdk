#!/bin/bash

# fail on error
set -e

TORCH_VERSION="2.1.0"

#Parse cli
while [ "$1" != "" ]; do
  case $1 in
    --torch-version ) shift
        TORCH_VERSION=$1
        ;;
  esac
  shift
done

echo "Using PyTorch version ${TORCH_VERSION}"

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
    TORCH_VERSION="1.13"
    echo "- Detected inf1 - using version ${TORCH_VERSION}"
else
    echo "- Detected inf2 or trn1 - using version ${TORCH_VERSION}"
fi

# checkout tokenizers and apply neuron patch
if [ ! -e "tokenizers" ]; then
    git clone https://github.com/huggingface/tokenizers.git
    cp neuron.patch tokenizers/neuron.patch
    pushd tokenizers
    git checkout d8c4388166cad8f0216dfc485efd6207a3275af2
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
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip
    unzip libtorch-shared-with-deps-${TORCH_VERSION}+cpu.zip
    rm -f libtorch-shared-with-deps-${TORCH_VERSION}+cpu.zip
fi

# get libneuron_op.so and install into libtorch
pip install --upgrade "transformers==4.40.0
python bert_neuronx/compile.py

if [ "$OLD_TOOL_CHAIN" == "True" ]
  then
    cp -f $(find ~/ -type d -name '*venv*' -exec find {} -type f -name 'libtorchneuron.so' \; | grep torch_neuron) libtorch/lib/
    cp -f $(find ~/ -type d -name '*venv*' -exec find {} -type f -name 'libnrt.so' \;) libtorch/lib/
    cp -f $(find ~/ -type d -name '*venv*' -exec find {} -type f -name 'libnrt.so.1' \;) libtorch/lib/
  else
    cp -f $(find ~/ -type d -name '*venv*' -exec find {} -type f -name 'libtorchneuron.so' \; | grep torch_neuronx) libtorch/lib/
fi

# compile example app
pushd example_app
chmod +x build.sh
./build.sh
popd

chmod +x run_tests.sh
echo "Successfully completed setup"