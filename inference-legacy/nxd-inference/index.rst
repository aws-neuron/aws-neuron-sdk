.. meta::
   :description: NxD Inference (NeuronX Distributed Inference) is an ML inference library included with the Neuron SDK that simplifies deploying deep learning models on AWS Inferentia and Trainium instances.
   :keywords: NxD Inference, NeuronX Distributed Inference, AWS Neuron SDK, Deep Learning Inference, LLM Deployment, Model Optimization, Tensor Parallelism, Sequence Parallelism, vLLM Integration, Speculative Decoding, Continuous Batching 
   :date-modified: 12/02/2025

.. _nxdi-index:

NxD Inference
=============

.. note:: Maintenance status

   NxD Inference and the inference-context NxD Core documentation in this section are in
   **maintenance status** — they continue to receive critical fixes and remain the supported
   path for **Trn1/Inf2** inference and for custom **NxD Inference API** serving (for example
   Flux/diffusion). This content is **not** end-of-support. For new inference work, the
   go-forward path is the :ref:`vLLM Neuron plugin <vllm-neuron-home>`; the task-oriented
   :ref:`Inference on Neuron <inference-home>` page shows where each path fits.

This section contains the technical documentation specific to the NxD Inference library included with the Neuron SDK.

.. toctree::
    :maxdepth: 1
    :hidden:

    Overview </inference-legacy/nxd-inference/overview-index>
    Setup </inference-legacy/nxd-inference/nxdi-setup>
    vLLM </inference-legacy/nxd-inference/vllm/index>
    Tutorials  </inference-legacy/nxd-inference/tutorials/index>
    Developer Guides  </inference-legacy/nxd-inference/developer_guides/index>
    API Reference Guide </inference-legacy/nxd-inference/api-guides/index>
    App Notes  </inference-legacy/nxd-inference/app-notes/index>
    Release Notes </release-notes/components/nxd-inference>
    Misc  </inference-legacy/nxd-inference/misc/index>

What is NxD Inference?
-----------------------

NxD Inference (NeuronX Distributed Inference) is an ML inference library included with the Neuron SDK that simplifies deploying deep learning models on AWS Inferentia and Trainium instances. It offers advanced features like continuous batching and speculative decoding for high-performance inference, and supports popular models like Llama-3.1, DBRX, and Mixtral.

With NxD Inference, developers can:

* Deploy production-ready LLMs with minimal configuration
* Leverage optimizations like KV Cache, Flash Attention, and Quantization
* Distribute large models across multiple NeuronCores using Tensor and Sequence Parallelism
* Integrate with vLLM for seamless production deployment
* Customize and extend models with a modular design approach

With NxD Inference, developers can:

Use vLLM for Inference
------------------------

Neuron recommends that use vLLM when building your inference models. Read more about Neuron's integration with vLLM here: :doc:`vLLM on Neuron </inference-legacy/nxd-inference/vllm/index>`

Quickstarts
------------

.. grid:: 1 1 2 2
    :gutter: 3
    
    .. grid-item-card:: Quickstart: Serve models online with vLLM on Neuron
        :link: /inference-legacy/nxd-inference/vllm/quickstart-vllm-online-serving
        :link-type: doc
        :class-card: sd-rounded-3
        
        Get started serving online models with vLLM. Time to complete: ~20 minutes.

    .. grid-item-card:: Quickstart: Run offline inference with vLLM on Neuron
        :link: /inference-legacy/nxd-inference/vllm/quickstart-vllm-offline-serving
        :link-type: doc
        :class-card: sd-rounded-3
        
        Get started running offline inference with vLLM. Time to complete: ~20 minutes.

NxD Inference documentation
----------------------------

.. grid:: 1 1 2 2
    :gutter: 3
    
    .. grid-item-card:: Overview
        :link: /inference-legacy/nxd-inference/overview-index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Learn about NxD Inference architecture, key features, and how it can help you deploy models efficiently on AWS Neuron hardware.

    .. grid-item-card:: Setup
        :link: /inference-legacy/nxd-inference/nxdi-setup
        :link-type: doc
        :class-card: sd-rounded-3
        
        Step-by-step instructions for setting up NxD Inference using DLAMI, Docker containers, or manual installation.

    .. grid-item-card:: Get Started with Models
        :link: /inference-legacy/nxd-inference/models/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Deploy production-ready models like Llama 3, DBRX, and Mixtral with optimized configurations for AWS Neuron hardware.

    .. grid-item-card:: Tutorials
        :link: /inference-legacy/nxd-inference/tutorials/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Hands-on tutorials for deploying various models, including Llama 3 variants, multimodal models, and using advanced features like speculative decoding.

    .. grid-item-card:: Developer Guides
        :link: /inference-legacy/nxd-inference/developer_guides/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        In-depth guides for model onboarding, feature integration, vLLM usage, benchmarking, and customizing inference workflows.

    .. grid-item-card:: API Reference
        :link: /inference-legacy/nxd-inference/api-guides/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Comprehensive API documentation for integrating NxD Inference into your applications and customizing inference behavior.

    .. grid-item-card:: Application Notes
        :link: /inference-legacy/nxd-inference/app-notes/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Detailed application notes on parallelism strategies and other advanced topics for optimizing inference performance.

    .. grid-item-card:: Misc Resources
        :link: /inference-legacy/nxd-inference/misc/index
        :link-type: doc
        :class-card: sd-rounded-3
        
        Release notes, troubleshooting guides, and other helpful resources for working with NxD Inference.

    .. grid-item-card:: NxD Inference Release Notes
        :link: /release-notes/components/nxd-inference
        :link-type: doc
        :class-card: sd-rounded-3
        
        Release notes, troubleshooting guides, and other helpful resources for working with NxD Inference.
