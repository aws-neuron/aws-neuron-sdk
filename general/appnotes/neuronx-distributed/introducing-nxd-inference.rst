.. _introduce-nxd-inference:

Introducing NeuronX Distributed (NxD) Inference
==============================================

.. contents:: Table of contents
   :local:
   :depth: 2



What are we introducing?
------------------------


Starting with the Neuron SDK 2.21 release, we are introducing NxD Inference, an open-source PyTorch-based inference library that simplifies deep learning model deployment on AWS Inferentia and Trainium instances. NxD Inference is designed for optimized inference, enabling quick onboarding of PyTorch models with minimal changes. It features a modular architecture that facilitates easy integration of HuggingFace PyTorch models and is compatible with serving engines like vLLM.

Please see :ref:`nxdi-index` for NxD Inference overview and documentation.


How can I install NxD Inference library?
-----------------------------------------
Please refer to :ref:`nxdi-setup` for installation instructions.


I am currently using the Transformers NeuronX library for inference. How does the NxD Inference library affect me?
----------------------------------------------------------------------------------------------------------

If you are using Transformers NeuronX (TNx) in production, you can continue doing so. However, if you are planning to onboard new models to Neuron for inference, NxD Inference offers several advantages to consider.

NxD Inference is designed to enable easy on-boarding of PyTorch models and comes with new features and enhanced support:

* **Hardware Support**: While TNx is not supported on Trn2, NxD Inference supports all platforms (Trn1, Inf2, and Trn2)
* **Simplified interface**: To simplify model development with NxD Inference, you write modeling code using PyTorch with standard Python, rather than using PyHLO as in TNx.
* **Easy Migration**: NxD Inference was designed to provide seamless migration from TNx, especially if you are using it with vLLM. You can migrate your existing TNx inference scripts using the :ref:`migration guide <nxdi_migrate_from_tnx>`
* **Enhanced Capabilities**: NxD Inference offers more comprehensive support for MoE models and multimodal models (Llama 3.2) compared to TNx
* **Future Development**: New inference features and support for advanced model architectures (like multi-modality/video models) will be focused on NxD Inference



I am currently using vLLM with Transformers NeuronX library for inference. Does NxD Inference library support vLLM ?
---------------------------------------------------------------------------------------------------------------------

Yes, NxD Inference library supports vLLM inference engine.  Neuron vLLM integration in 2.21 release will start supporting both NxD Inference and Transformers NeuronX libraries.  To use vLLM with NxD Inference library, you can refer to the :ref:`nxdi-vllm-user-guide`.



What features and models are available in Transformers NeuronX (TNx) but not yet in NeuronX Distributed Inference?
------------------------------------------------------------------------------------------------------------------

While NxD Inference supports most features and models available in TNx, there are some differences in current support that users should be aware of.

**Features that are not yet supported in NxD Inference**: The following TNx features aren't supported yet in the NxD Inference library.

* Neuron persistent cache
* Multi-Node Inference support


**Models not part of NxD Inference Model Hub**: The following models are included in Transformers NeuronX but not currently in NxD Inference library:

* Bloom
* GPT2
* GPT-J
* GPT-NEOX

If you need to use these models with NxD Inference, we encourage you to follow the :ref:`onboarding models developer guide <nxdi-onboarding-models>`. The onboarding process in NxD Inference is more straightforward compared to TNx due to its PyTorch-based architecture.


I currently use Hugging Face TGI serving engine for deploying and serving Large Language Models (LLMs) on Neuron. How does NxD Inference library affect me?
-----------------------------------------------------------------------------------------------------------------------------------------------------------

If you are currently using Hugging Face TGI serving engine to deploy models on Neuron, the introduction of NxD Inference library will not have any impact and you can continue to use your existing inference workloads. Hugging Face TGI integrates with Neuron SDK Inference libraries in a way that abstracts the underlying library for the users.



I am new to Neuron and have inference workloads, what library should I use?
----------------------------------------------------------------------------

We recommend you use NxD Inference for your model inference workloads. To learn how to get started using NxD Inference, see the :ref:`nxdi-index` documentation








Additional Resources
--------------------

* :ref:`nxdi-index`
* :ref:`nxdi-overview`
* :ref:`neuronx-distributed-inference-rn`
