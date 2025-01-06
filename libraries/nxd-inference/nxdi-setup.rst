.. _nxdi-setup:

NxD Inference Setup Guide
=========================

The NeuronX Distributed (NxD) Inference framework is built on top of
:ref:`NxD Core <neuronx-distributed-index>`. Follow the steps in this
guide to set up your environment to run inference using the NxD Inference framework.

.. contents:: Table of contents
   :local:
   :depth: 2

Option 1: Launch an instance using a Neuron DLAMI
-------------------------------------------------
Neuron Deep Learning AMIs (DLAMIs) are Amazon Machine Images (AMIs) that come
with the Neuron SDK pre-installed. To quickly get started with NxD Inference,
you can launch an EC2 instance with the multi-framework DLAMI, which includes
NxD Inference and its dependencies. For more information, see the
:ref:`Neuron Multi-Framework DLAMI Guide <setup-ubuntu22-multi-framework-dlami>`
and :ref:`neuron-dlami-overview`.

After you launch an instance, you can run the following command to activate the
NxD Inference virtual environment.

::

   source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate



Option 2: Use a Neuron Deep Learning Container (DLC)
----------------------------------------------------
Neuron Deep Learning Containers (DLCs) are Docker images that come with the
Neuron SDK pre-installed. To run NxD Inference in a Docker container, use the
`Neuronx PyTorch Inference Containers <https://github.com/aws-neuron/deep-learning-containers#pytorch-inference-neuronx>`_.
For more information, see :ref:`neuron_containers`.


Option 3: Manually Install NxD Inference
----------------------------------------

Follow these instructions to manually install NxD Inference on an instance.

.. note:: 

   For information about which Python versions are compatible with the Neuron
   SDK, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.

Setup a Neuron Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you install NxD Inference, you must install the Neuron SDK and its
dependencies, including PyTorch Neuron (``torch-neuronx``). Follow instructions
for one of the following operating systems:

* :ref:`PyTorch NeuronX Setup on Ubuntu 22 <setup-torch-neuronx-ubuntu22>`
* :ref:`PyTorch NeuronX Setup on Amazon Linux 2023 <setup-torch-neuronx-al2023>`


Install NxD Inference
^^^^^^^^^^^^^^^^^^^^^

Run this command to install NxD Inference.

::

   source aws_neuron_venv_pytorch/bin/activate
   pip install -U pip
   pip install --upgrade neuronx-cc==2.* neuronx-distributed-inference --index-url https://pip.repos.neuron.amazonaws.com


Verify NxD Inference Installation
---------------------------------

To verify that NxD Inference installed successfully, check that you can
run the ``inference_demo`` console script.

::

   inference_demo --help
