.. _neuron-pytorch:

PyTorch Support on Neuron
==========================

PyTorch running on Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and AWS Inferentia-based Amazon EC2 instances.

The PyTorch plugin for Neuron architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes. 
 
For help selecting a framework type for inference, see :ref:`torch-neuron_vs_torch-neuronx`. 

.. admonition:: Introducing TorchNeuron, a native backend for AWS Trainium

    At re:Invent '25, AWS Neuron announced their new PyTorch package, "TorchNeuron", which includes the ``torch-neuronx`` library and initial support for a native PyTorch backend (TorchDynamo) with eager execution, ``torch.compile``, and standard distributed APIs. 

    For more details on what is coming with TorchNeuron and PyTorch eager mode support, see :doc:`pytorch-native-overview`.

.. _pytorch-neuronx-training:

.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/torch/torch-setup
    /frameworks/torch/pytorch-native-overview


.. toctree::
    :maxdepth: 1
    :hidden:

    Inference (Inf2, Trn1, Trn2) </frameworks/torch/inference-torch-neuronx>
    Inference (Inf1) </frameworks/torch/inference-torch-neuron>


.. toctree::
    :maxdepth: 1
    :hidden:
    
    Training </frameworks/torch/training-torch-neuronx>


.. _pytorch-neuronx-main:

"""""""""""""""
PyTorch NeuronX
"""""""""""""""
 
.. card:: PyTorch NeuronX for training on Trn1 and Trn2
    :link: training-torch-neuronx
    :link-type: doc
    :class-body: sphinx-design-class-title-small

.. card:: PyTorch NeuronX for inference on Inf2, Trn1, and Trn2 
    :link: inference-torch-neuronx
    :link-type: doc
    :class-body: sphinx-design-class-title-small

.. _pytorch-neuron-main:

""""""""""""""
PyTorch Neuron
""""""""""""""

.. card:: PyTorch Neuron for inference on Inf1
    :link: inference-torch-neuron
    :link-type: doc
    :class-body: sphinx-design-class-title-small
