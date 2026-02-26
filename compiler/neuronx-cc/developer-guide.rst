.. meta::
   :description: Developer guides for the Neuron Compiler (neuronx-cc), including mixed precision training, performance tuning, and custom kernel implementation for AWS Trainium and Inferentia.
   :keywords: neuronx-cc, Neuron Compiler, mixed precision, BF16, FP16, TF32, auto-cast, convolution kernels, UNet, performance optimization, Trainium, Inferentia

Developer Guide
===================

Learn how to optimize your models with the Neuron Compiler (neuronx-cc). These guides cover mixed precision training, performance-accuracy tuning, and custom kernel implementations for AWS Trainium and Inferentia instances.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Mixed Precision and Performance-Accuracy Tuning
      :link: /about-neuron/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision
      :link-type: doc

      Learn how to use FP32, TF32, FP16, and BF16 data types with the Neuron Compiler's auto-cast options to balance performance and accuracy. Understand the tradeoffs between different data types and how to configure compiler settings for optimal model execution.

   .. grid-item-card:: How to Use Convolution Kernels in UNet Training Models
      :link: /compiler/neuronx-cc/how-to-convolution-in-unet
      :link-type: doc

      Modify UNet training models to use custom convolution kernels with NKI (Neuron Kernel Interface). This implementation helps avoid out-of-memory errors when training convolution-heavy models on Trainium instances.

.. toctree::
    :hidden:
    :maxdepth: 1
    
    /about-neuron/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision
    /compiler/neuronx-cc/how-to-convolution-in-unet