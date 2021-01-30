Troubleshooting Guide for Torch-Neuron
======================================

General Torch-Neuron issues
---------------------------

If you see an error about "Unknown builtin op: neuron::forward_1" like below, please ensure that import line "import torch_neuron" (to register the Neuron custom operation) is in the inference script before using torch.jit.load.

::

   Unknown builtin op: neuron::forward_1.
   Could not find any similar ops to neuron::forward_1. This op may not exist or may not be currently supported in TorchScript.


TorchVision related issues
--------------------------

If you encounter an error like below, it is because latest torchvision
version >= 0.7 is not compatible with Torch-Neuron 1.5.1. Please
downgrade torchvision to version 0.6.1:

::

   E   AttributeError: module 'torch.jit' has no attribute '_script_if_tracing'                                                                                      
