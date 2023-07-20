.. _pytorch-neuronx-main:
.. _neuron-pytorch:

PyTorch Neuron
==============

PyTorch Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and Inferentia-based Amazon EC2 instances.

PyTorch Neuron plugin architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes. 
 
.. _pytorch-neuronx-training:


.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/torch/torch-setup


.. toctree::
    :maxdepth: 1
    :hidden:

    Inference (Inf2 & Trn1) </frameworks/torch/inference-torch-neuronx>
    Inference (Inf1) </frameworks/torch/inference-torch-neuron>


.. toctree::
    :maxdepth: 1
    :hidden:
    
    Training </frameworks/torch/training-torch-neuronx>


.. note::

    For help selecting a framework type for Inference, see:

    :ref:`torch-neuron_vs_torch-neuronx`

.. card:: PyTorch Neuron(``torch-neuronx``) for Inference on ``Inf2`` & ``Trn1`` / ``Trn1n``
    :link: inference-torch-neuronx
    :link-type: ref
    :class-body: sphinx-design-class-title-small


.. card:: PyTorch Neuron(``torch-neuron``) for Inference on ``Inf1``
    :link: inference-torch-neuron
    :link-type: ref
    :class-body: sphinx-design-class-title-small

    
.. card:: PyTorch Neuron(``torch-neuronx``) for Training on ``Trn1`` / ``Trn1n``
    :link: training-torch-neuronx
    :link-type: ref
    :class-body: sphinx-design-class-title-small