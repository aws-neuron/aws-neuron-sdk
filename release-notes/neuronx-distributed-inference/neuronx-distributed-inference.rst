.. _neuronx-distributed-inference-rn:


NxD Inference Release Notes (``neuronx-distributed-inference``)
=============================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx Distributed Inference library.

Neuronx Distributed Inference [0.1.0] (Beta)
--------------------------------------------
Date: 12/03/2024

Features in this release
^^^^^^^^^^^^^^^^^^^^^^^^

NeuronX Distributed (NxD) Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.

This is the first release of NxD Inference (Beta) that includes:

* Support for Trn2 instances
* Compatibility with HuggingFace checkpoints and ``generate()`` API
* vLLM integration
* Model compilation and serialization
* Tensor parallelism
* Speculative decoding
* Quantization
* Dynamic sampling
* Llama3.1 405B Inference Example on Trn2
* Open Source Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

For more information about the features supported by NxDI, see :ref:`nxdi-feature-guide`.
