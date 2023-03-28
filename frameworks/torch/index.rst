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
    
    Training </frameworks/torch/training-torch-neuronx>


.. toctree::
    :maxdepth: 1
    :hidden:
    
    /frameworks/torch/inference


.. dropdown::  Pytorch Neuron Setup
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include:: dropdown-neuron-setup.txt


.. tab-set::

    .. tab-item:: Training (``torch-neuronx``)

        .. include:: training-torch-neuronx.txt

    .. tab-item:: Inference (``torch-neuronx & torch-neuron``)

        .. note::

            For help selecting a framework type, see:

            :ref:`torch-neuron_vs_torch-neuronx`

        .. tab-set::

            .. tab-item:: Inference on Inf2 & Trn1/Trn1n (``torch-neuronx``)

                .. include:: inference-torch-neuronx.txt

            .. tab-item:: Inference on Inf1 (``torch-neuron``)

                .. include:: inference-torch-neuron.txt