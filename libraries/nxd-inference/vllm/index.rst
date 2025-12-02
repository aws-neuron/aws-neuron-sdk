.. meta:: 
    :description: Run high-performance LLM inference with vLLM on AWS Neuron accelerators. Deploy models like Llama, Qwen, and more on Trainium and Inferentia instances.
    :date-modified: 11/25/2025


vLLM on Neuron
===============

vLLM on Neuron enables high-performance LLM inference on AWS Trainium and Inferentia instances, providing a streamlined deployment experience with minimal code changes. The integration leverages AWS Neuron's optimized AI inference capabilities and vLLM's advanced features like continuous batching to deliver efficient model serving for both latency-sensitive applications and high-throughput batch processing workloads.

Overview
---------

vLLM is a popular library for LLM inference and serving that integrates with AWS Neuron through the NxD Inference (neuronx-distributed-inference) library. This integration uses vLLM's Plugin System to extend the model execution components responsible for loading and invoking models within vLLM's LLMEngine, while maintaining vLLM's input processing, scheduling, and output processing behaviors.

**Key Features:**

- **Continuous batching** for efficient processing of multiple requests
- **Prefix caching** to improve time-to-first-token by reusing KV cache of common prompts
- **Speculative decoding** support (Eagle V1)
- **Quantization** with INT8/FP8 support for optimized performance
- **Dynamic sampling** and tool calling capabilities
- **Multimodal support** for models like Llama 4 Scout and Maverick

**Supported Models:**

- Llama 2/3.1/3.3
- Llama 4 Scout, Maverick (with multimodal capabilities)
- Qwen 2.5
- Qwen 3
- Custom models onboarded to NxD Inference

**Deployment Options:**

- Quick deployment using pre-configured Deep Learning Containers
- Manual installation from source with the vLLM-Neuron plugin
- Offline batch inference for processing multiple prompts
- Online model serving with an OpenAI-compatible API server

Get Started with Inference and vLLM on Neuron
----------------------------------------------

Learn how to run high-performance inference workloads using vLLM on AWS Neuron accelerators. These quickstart guides walk you through setting up both offline batch processing and online API serving, helping you deploy large language models efficiently on Trainium and Inferentia instances.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Deploy a Deep Learning Container with vLLM
      :link: /containers/get-started/quickstart-configure-deploy-dlc
      :link-type: doc
      :class-card: sd-border-1

      Quickly deploy a vLLM server on Trainium and Inferentia instances using a DLC image preconfigured with AWS Neuron SDK artifacts.

   .. grid-item-card:: Offline Model Serving
      :link: quickstart-vllm-offline-serving
      :link-type: doc
      :class-card: sd-border-1

      Run batch inference jobs with vLLM on Neuron. Install the plugin, process multiple prompts, and cache compiled artifacts for faster reruns.

   .. grid-item-card:: Online Model Serving
      :link: quickstart-vllm-online-serving
      :link-type: doc
      :class-card: sd-border-1

      Launch an OpenAI-compatible API server with vLLM on Neuron. Set up interactive endpoints, validate with curl, and integrate with the OpenAI SDK.

Guides for vLLM on Neuron
--------------------------

.. grid:: 1 
   :gutter: 3

   .. grid-item-card:: vLLM on Neuron User Guide (V1)
      :link: /libraries/nxd-inference/developer_guides/vllm-user-guide-v1
      :link-type: doc
      :class-card: sd-border-1

      Learn the details of developing inference models on Neuron with vLLM V1.

vLLM on Neuron Tutorials
--------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Deploy Llama4 with vLLM
      :link: /libraries/nxd-inference/tutorials/llama4-tutorial
      :link-type: doc
      :class-card: sd-border-1

      Learn how to deploy Llama4 multimodal models on Trainium2 instances using vLLM for both offline and online inference.

.. toctree::
    :hidden:
    :maxdepth: 1

    Quickstart: Offline Model Serving </libraries/nxd-inference/vllm/quickstart-vllm-offline-serving>
    Quickstart: Online Model Serving </libraries/nxd-inference/vllm/quickstart-vllm-online-serving>
    vLLM on Neuron User Guide </libraries/nxd-inference/developer_guides/vllm-user-guide-v1>
    Deploy Llama4 with vLLM </libraries/nxd-inference/tutorials/llama4-tutorial>
