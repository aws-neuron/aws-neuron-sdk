Troubleshooting Guide for Torch-Neuron
======================================

TorchVision related issues
--------------------------

If you encounter an error like below, it is because latest torchvision
version >= 0.7 is not compatible with Torch-Neuron 1.5.1. Please
downgrade torchvision to version 0.6.1:

::

   E   AttributeError: module 'torch.jit' has no attribute '_script_if_tracing'                                                                                      
