.. _neuron-install-guide:

Neuron Install Guide
====================

.. _dlami:

Deep Learning AMI
~~~~~~~~~~~~~~~~~

`AWS Deep Learning AMI (DLAMI) <https://docs.aws.amazon.com/dlami/index.html>`_ is 
the recommended AMI to use with Neuron SDK, In addition to DLAMI Neuron SDK can be installed on Ubuntu or Amazon Linux using 
standard package managers (apt, yum, pip, and conda) to install and keep 
updates current, Neuron SDK is suppotred in `DLAMI with Conda <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_  and in 
`DLAMI Base <https://docs.aws.amazon.com/dlami/latest/devguide/base.html>`_ , for more information see
`The AWS Inferentia Chip With DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`_ .

.. note::

   Only Ubuntu 16,18 and Amazon Linux2 DLAMI are supported (Amazon Linux is not supported)

   Only DLAMI versions 26.0 and newer have Neuron support included.

   Neuron supports Python versions 3.5, 3.6, and 3.7.


Installing Neuron
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   install-pytorch
   install-tensorflow
   install-mxnet
   install-tensorboard
   dlcontainersß
