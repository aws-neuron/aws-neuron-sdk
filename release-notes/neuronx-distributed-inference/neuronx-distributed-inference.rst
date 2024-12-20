.. _neuronx-distributed-inference-rn:


NxD Inference Release Notes (``neuronx-distributed-inference``)
=============================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for Neuronx Distributed Inference library.


Neuronx Distributed Inference [0.1.0] (Beta) (Neuron 2.21 Release)
------------------------------------------------------------------
Date: 12/20/2024

Features in this Release
^^^^^^^^^^^^^^^^^^^^^^^^

NeuronX Distributed (NxD) Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.

This is the first release of NxD Inference (Beta) that includes:

* Support for Trn2, Inf2, and Trn1 instances
* Support for the following model architectures. For more information, including
  links to specific supported model checkpoints, see :ref:`nxdi-model-reference`.

  * Llama (Text), including Llama 2, Llama 3, Llama 3.1, Llama 3.2, and Llama 3.3
  * Llama (Multimodal), including Llama 3.2 multimodal
  * Mistral (using Llama architecture)
  * Mixtral
  * DBRX
  
* Support for onboarding additional models.
* Compatibility with HuggingFace checkpoints and ``generate()`` API
* vLLM integration
* Model compilation and serialization
* Tensor parallelism
* Speculative decoding

  * EAGLE speculative decoding
  * Medusa speculative decoding
  * Vanilla speculative decoding

* Quantization
* Dynamic sampling
* Llama3.1 405B Inference Example on Trn2
* Open Source Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

For more information about the features supported by NxDI, see :ref:`nxdi-feature-guide`.


Known Issues and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Longer Load Times for Large Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue: Users may experience extended load times when working with large models,
particularly during weight sharding and initial model load. This is especially
noticeable with models like Llama 3.1 405B.

Root Cause: These delays are primarily due to storage performance limitations.

Recommended Workaround: To mitigate this issue, we recommend that you store
model checkpoints in high-performance storage options:

* `Instance store volumes <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ssd-instance-store.html>`_:
  On supported instances, instance store volumes offer fast, temporary block-level storage.
* `Optimized EBS volumes <https://docs.aws.amazon.com/ebs/latest/userguide/ebs-performance.html>`_:
  For persistent storage with enhanced performance.

By using these storage optimizations, you can reduce model load times and improve
your overall workflow efficiency.

Note: Load times may still vary depending on model size and specific hardware configurations.


Other Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Llama 3.2 11B (Multimodal) is not yet supported with PyTorch 2.5.
* The following model architectures are tested only on Trn1 and Inf2:

  * Llama (Multimodal)

* The following model architectures are tested only on Trn1:
  
  * Mixtral
  * DBRX

* The following kernels are tested only on Trn2:
  
  * MLP
  * QKV
  
* If you run inference with an prompt that is larger than the model's ``max_context_length``,
  the model will generate incorrect output. In a future release, NxD Inference will
  throw an error in this scenario.
* Continuous batching (including through vLLM) supports batch size up to 4.
  Static batching supports larger batch sizes.
* To use greedy on-device sampling, you must set ``do_sample`` to ``True``.
* To use FP8 quantization or KV cache quantization, you must set the
  ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.


Neuronx Distributed Inference [0.1.0] (Beta) (Trn2)
---------------------------------------------------
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

  * EAGLE speculative decoding
  * Medusa speculative decoding
  * Vanilla speculative decoding

* Quantization
* Dynamic sampling
* Llama3.1 405B Inference Example on Trn2
* Open Source Github repository: `aws-neuron/neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_

For more information about the features supported by NxDI, see :ref:`nxdi-feature-guide`.
