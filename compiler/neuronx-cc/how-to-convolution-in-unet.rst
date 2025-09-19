.. meta::
   :description: Learn how to modify UNet training models to use convolution kernels with AWS Neuron SDK
   :date_updated: 2025-09-09

.. _implement-convolution-kernels-unet:

================================================
How to Use Convolution Kernels in UNet Training Models
================================================

Task overview
-------------
This topic discusses how to modify UNet training models to use convolution kernels with the AWS Neuron SDK. This implementation helps avoid out-of-memory errors seen when performing training on the convolution-heavy UNet model.

Prerequisites
-------------
- AWS Neuron SDK 2.26 or later: Required for kernel implementation support
- trn1.32xlarge instance: Needed for model training  
- Existing UNet implementation: Base model to be modified
- PyTorch-Neuron environment: Required for neural network operations

Instructions
------------

**1: Import required dependencies**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Function
   import neuronxcc.nki as nki
   import neuronxcc.nki.language as nl
   from neuronxcc.nki._private_kernels.conv import conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh

**2: Create the convolution wrapper function**

.. code-block:: python

   @nki.jit
   def conv_wrap(img_ref, filter_ref, out_shape):
       out_arr = nl.ndarray(shape=out_shape, dtype=img_ref.dtype, buffer=nl.hbm)
       conv2d_dw_fb01_io01_01bf_rep_nhwc_Pcinh(img_ref, filter_ref, out_arr, **{
           'input': img_ref.shape,
           'filter': filter_ref.shape, 
           'output': out_shape,
           'in_perm': [0, 1, 2, 3],
           'kern_perm': [0, 1, 2, 3],
           'out_perm': [0, 1, 2, 3],
           'stride': (1, 1),
           'padding': ((1, 1), (1, 1))})
       return out_arr

**3: Implement the custom Conv2d module**

.. code-block:: python

   class BwdConv2dWithKernel(nn.Module):
       def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
           super().__init__()
           assert padding == 1
           assert bias == False
           self.in_channels = in_channels
           self.out_channels = out_channels
           self.kernel_size = kernel_size
           self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
           nn.init.kaiming_uniform_(self.weight, a=0.0, mode='fan_in', nonlinearity='leaky_relu')

**4: Replace standard convolutions in the UNet model**

.. code-block:: python

   class DoubleConvWithKernel(nn.Module):
       def __init__(self, in_channels, out_channels, mid_channels=None):
           super().__init__()
           if not mid_channels:
               mid_channels = out_channels
           self.double_conv = nn.Sequential(
               BwdConv2dWithKernel(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
               nn.BatchNorm2d(mid_channels),
               nn.ReLU(inplace=True),
               BwdConv2dWithKernel(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
           )

**5: Update the UNet model initialization**

.. code-block:: python

   def __init__(self, n_channels, n_classes, bilinear=False):
       super().__init__()
       self.n_channels = n_channels
       self.n_classes = n_classes
       self.bilinear = bilinear
       self.inc = (DoubleConvWithKernel(n_channels, 64))
       # ... rest of initialization

Confirm your work
-----------------

To confirm successful implementation, verify the following:

.. code-block:: bash

   Expected training output
   Training Device=xla:0 Epoch=1 Step=20 Loss=0.30803
   Training Device=xla:0 Epoch=2 Step=560 Loss=0.01826

Check for:

- No out-of-memory errors during execution
- Decreasing loss values across epochs

Common issues
-------------

.. rubric:: Memory Errors

- Solution: Verify all standard convolutions are replaced with BwdConv2dWithKernel implementations

.. rubric:: Compilation Errors

- Solution: Confirm Neuron SDK version is 2.26 or later

.. rubric:: Kernel Errors

- Solution: Use the kernel for supported configurations. The kernel will error out in unsupported scenarios.

Related information
-------------------

.. toctree::
   :maxdepth: 1

   * `UNet training sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/unet_image_segmentation>`_ - Sample UNet training implementation
