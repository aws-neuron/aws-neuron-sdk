#!/bin/bash

# Installation script to build with torch dependency from /usr/local
set -x

# Find paths for local packages
PATH_TOKENIZERS_LIB=../tokenizers_binding/lib
PATH_TORCH=../libtorch
PATH_TORCH_INC=${PATH_TORCH}/include
PATH_TORCH_LIB=${PATH_TORCH}/lib
PATH_NEURON_LIB=${PATH_TORCH}/lib

g++ utils.cpp example_app.cpp \
	-o ../example-app \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	-I${PATH_TORCH_INC} \
	-L${PATH_TOKENIZERS_LIB} \
	-L${PATH_NEURON_LIB} \
	-L${PATH_TORCH_LIB} \
	-Wl,-rpath,${PATH_TORCH_LIB} \
	-ltokenizers \
	-ltorchneuron \
	-ltorch_cpu \
	-lc10 \
	-lpthread \
	-lnrt
