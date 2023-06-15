.. _neuronx_distributed_setup:

Neuron Distributed Setup (``neuronx-distributed``)
==================================================

`Install PyTorch Neuron on
Trn1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html#pytorch-neuronx-install>`__
to create a pytorch environment. It is recommended to work out of python
virtual env so as to avoid package installation issues.

You can install the ``neuronx-distributed`` package using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Make sure the transformers version is set to ``4.26.0``




