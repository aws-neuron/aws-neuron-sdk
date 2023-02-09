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

    /frameworks/torch/training


.. toctree::
    :maxdepth: 1
    :hidden:

    /frameworks/torch/inference


.. tab-set::

    .. tab-item:: Training
        :name: torch-neuronx-training-main

        .. include:: tab-training.rst

    .. tab-item:: Inference

        .. note::

            For help selecting a framework type, see:

            :ref:`torch-neuron_vs_torch-neuronx`

        .. tab-set::

            .. tab-item:: Inference on Trn1 (``torch-neuronx``)
                :name: torch-neuronx-inference-main

                .. include:: tab-inference-torch-neuronx.txt

            .. tab-item:: Inference on Inf1  (``torch-neuron``)
                :name: torch-neuron-inference-main

                .. include:: tab-inference-torch-neuron.txt
