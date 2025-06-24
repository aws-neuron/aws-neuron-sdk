#!/bin/bash

# Installation script to build with torch dependency from /usr/local
set -x

# Get the PyTorch version from parent script or determine it
if [ -z "${TORCH_VERSION}" ]; then
    TORCH_VERSION=$(python -c "import torch; v=torch.__version__.split('+')[0]; print(f'{v}')")
fi

# Find paths for local packages
PATH_TOKENIZERS_LIB=../tokenizers_binding/lib
PATH_TORCH=../libtorch
PATH_TORCH_INC=${PATH_TORCH}/include
PATH_TORCH_LIB=${PATH_TORCH}/lib
PATH_NEURON_LIB=${PATH_TORCH}/lib

if [ ! -e "${PATH_TORCH_LIB}/libnrt.so.1" ] && [ -e "/opt/aws/neuron/lib/libnrt.so.1" ]
then
    PATH_NEURON_LIB=/opt/aws/neuron/lib/
fi

# Set CXX11_ABI flag based on PyTorch version
MAJOR_VERSION=$(echo "${TORCH_VERSION}" | cut -d. -f1)
MINOR_VERSION=$(echo "${TORCH_VERSION}" | cut -d. -f2)
    
if [ "$MAJOR_VERSION" -gt 2 ] || ([ "$MAJOR_VERSION" -eq 2 ] && [ "$MINOR_VERSION" -ge 7 ]); then
    CXX11_ABI_FLAG=1
else
    CXX11_ABI_FLAG=0
fi


g++ utils.cpp example_app.cpp \
    -o ../example-app \
    -O2 \
    -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI_FLAG} \
    -I${PATH_TORCH_INC} \
    -L${PATH_TOKENIZERS_LIB} \
    -L${PATH_NEURON_LIB} \
    -L${PATH_TORCH_LIB} \
    -Wl,-rpath,libtorch/lib \
    -Wl,-rpath,tokenizers_binding/lib \
    -Wl,-rpath,$PATH_NEURON_LIB \
    -Wl,-no-as-needed \
    -ltokenizers \
    -ltorchneuron \
    -ltorch_cpu \
    -lc10 \
    -lpthread \
    -lnrt \
    -std=c++17
