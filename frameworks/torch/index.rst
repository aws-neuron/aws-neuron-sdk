.. _pytorch-neuronx-main:
.. _neuron-pytorch:

PyTorch Neuron
==============

PyTorch Neuron unlocks high-performance and cost-effective deep learning acceleration on AWS Trainium-based and Inferentia-based Amazon EC2 instances.

PyTorch Neuron plugin architecture enables native PyTorch models to be accelerated on Neuron devices, so you can use your existing framework application and get started easily with minimal code changes. 
 
.. _pytorch-neuronx-training:


.. dropdown::  Pytorch Neuron Setup
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. tab-set::

        .. tab-item:: torch-neuronx (``Trn1 & Inf2``)
            :name: torch-neuronx-install-main

            * :ref:`Fresh Install <pytorch-neuronx-install>`
            * :ref:`Update to latest release <pytorch-neuronx-update>`
            * :ref:`Install previous releases <pytorch-neuronx-install-prev>`

            .. include:: /general/setup/install-templates/trn1-ga-warning.txt

        .. tab-item:: torch-neuron (``Inf1``)
            :name: torch-neuron-install-main

            * :ref:`Fresh Install <install-neuron-pytorch>`
            * :ref:`Update to latest release <update-neuron-pytorch>`
            * :ref:`Install previous releases <install-prev-neuron-pytorch>`
            * :ref:`pytorch-install-cxx11`

.. tab-set::

    .. tab-item:: Training (``torch-neuronx``)

        .. include:: tab-training-torch-neuronx.txt

    .. tab-item:: Inference (``torch-neuronx & torch-neuron``)

        .. tab-set::

            .. tab-item:: Inference on Inf2 & Trn1 (``torch-neuronx``)

                .. include:: tab-inference-torch-neuronx.txt

            .. tab-item:: Inference on Inf1 (``torch-neuron``)

                .. include:: tab-inference-torch-neuron.txt