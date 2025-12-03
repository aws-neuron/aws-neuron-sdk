.. meta::
    :description: Comprehensive tutorials for NeuronX Distributed (NxD) Inference on AWS Neuron hardware, covering various LLM deployments and optimizations.
    :date-modified: 12/02/2025

.. _nxdi-tutorials-index:

NxD Inference Tutorials
========================

Welcome to the NeuronX Distributed (NxD) Inference tutorials collection. These step-by-step guides help you deploy and optimize large language models (LLMs) on AWS Neuron hardware. Learn how to run various models like Llama3, GPT, and more with different optimization techniques including speculative decoding, tensor parallelism, and disaggregated inference.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Tutorials

    Disaggregated Inference (1P1D) </libraries/nxd-inference/tutorials/disaggregated-inference-tutorial-1p1d>
    Disaggregated Inference </libraries/nxd-inference/tutorials/disaggregated-inference-tutorial>
    Flux Inference </libraries/nxd-inference/tutorials/flux-inference-tutorial>
    Generating Results with Performance CLI </libraries/nxd-inference/tutorials/generating-results-with-performance-cli>
    GPT-OSS 120B </libraries/nxd-inference/tutorials/trn3-gpt-oss-120b-tutorial>
    Llama3.1 405B on Trn2 </libraries/nxd-inference/tutorials/trn2-llama3.1-405b-tutorial>
    Llama3.1 405B with Speculative Decoding </libraries/nxd-inference/tutorials/trn2-llama3.1-405b-speculative-tutorial>
    Llama3.1 70B Instruct Accuracy Evaluation </libraries/nxd-inference/tutorials/trn1-llama3.1-70b-instruct-accuracy-eval-tutorial>
    Llama3.1 8B with Multi-LoRA </libraries/nxd-inference/tutorials/trn2-llama3.1-8b-multi-lora-tutorial>
    Llama3.2 Multimodal </libraries/nxd-inference/tutorials/llama3.2-multimodal-tutorial>
    Llama3.3 70B with APC </libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial>
    Llama3.3 70B with Data Parallelism </libraries/nxd-inference/tutorials/trn2-llama3.3-70b-dp-tutorial>
    Llama3.3 70B with Speculative Decoding </libraries/nxd-inference/tutorials/trn2-llama3.3-70b-tutorial>
    Llama4 </libraries/nxd-inference/tutorials/llama4-tutorial>
    Llama4 Legacy </libraries/nxd-inference/tutorials/llama4-tutorial-v0>
    Pixtral </libraries/nxd-inference/tutorials/pixtral-tutorial>
    Speculative Decoding </libraries/nxd-inference/tutorials/sd-inference-tutorial>

    
.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Llama3.1 405B on Trn2
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.1-405b-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Learn how to deploy Llama3.1 405B on a single Trn2 instance using NxD Inference with vLLM and explore performance optimization techniques.

    .. grid-item-card:: Llama3.2 Multimodal
        :link: /libraries/nxd-inference/tutorials/llama3.2-multimodal-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Deploy and run Llama3.2 Multimodal models on AWS Neuron hardware to process both text and image inputs for multimodal inference.

    .. grid-item-card:: Llama3.3 70B on Trn2
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.3-70b-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Deploy Llama3.3 70B on Trn2 instances and learn how to optimize performance with tensor parallelism and other NxD Inference features.

    .. grid-item-card:: Llama3.3 70B with Data Parallelism
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.3-70b-dp-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Explore data parallelism techniques for Llama3.3 70B on Trn2 to increase throughput for high-volume inference workloads.

    .. grid-item-card:: Llama3.1 8B with Multi-LoRA
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.1-8b-multi-lora-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Learn how to use multiple LoRA adapters with Llama3.1 8B on Trn2 for efficient fine-tuning and domain-specific inference.

    .. grid-item-card:: Llama3.1 405B with Speculative Decoding
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.1-405b-speculative-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Optimize Llama3.1 405B inference on Trn2 using vanilla fused speculative decoding techniques for improved performance.

    .. grid-item-card:: Llama3.1 70B Instruct Accuracy Evaluation
        :link: /libraries/nxd-inference/tutorials/trn1-llama3.1-70b-instruct-accuracy-eval-tutorial
        :class-card: sd-rounded-3

        Evaluate the accuracy of Llama3.1 70B Instruct model on Trn1 hardware and learn how to measure model performance.

    .. grid-item-card:: Disaggregated Inference
        :link: /libraries/nxd-inference/tutorials/disaggregated-inference-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Implement disaggregated inference to distribute model components across multiple instances for large-scale LLM deployment.

    .. grid-item-card:: Disaggregated Inference (1P1D)
        :link: /libraries/nxd-inference/tutorials/disaggregated-inference-tutorial-1p1d
        :link-type: doc
        :class-card: sd-rounded-3

        Learn about the 1P1D (1 Prefill, 1 Decode) pattern for disaggregated inference to optimize latency and throughput.

    .. grid-item-card:: GPT-OSS on Trainium3
        :link: /libraries/nxd-inference/tutorials/trn3-gpt-oss-120b-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Deploy open-source GPT models on Trainium3 hardware using NxD Inference and explore Trn3-specific optimizations.

    .. grid-item-card:: Llama3.3 70B with APC
        :link: /libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Deploy Llama3.3 70B on Trn2 with Attention Pattern Caching (APC) to improve inference performance for repetitive patterns.

    .. grid-item-card:: Llama4 Tutorial
        :link: /libraries/nxd-inference/tutorials/llama4-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Deploy and optimize Llama4 models on AWS Neuron hardware using NxD Inference with various performance tuning options.

    .. grid-item-card:: Generating Results with Performance CLI
        :link: /libraries/nxd-inference/tutorials/generating-results-with-performance-cli
        :link-type: doc
        :class-card: sd-rounded-3

        Use the Performance CLI tool to benchmark and generate performance results for NxD Inference deployments.

    .. grid-item-card:: Flux Inference
        :link: /libraries/nxd-inference/tutorials/flux-inference-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Learn how to use Flux for efficient inference with NxD, enabling dynamic batch processing and optimized resource utilization.

    .. grid-item-card:: Pixtral Tutorial
        :link: /libraries/nxd-inference/tutorials/pixtral-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Learn how to deploy `mistralai/Pixtral-Large-Instruct-2411 <https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411>`__ on a single `trn2.48xlarge` instance.

    .. grid-item-card:: Speculative Decoding (Qwen3-32B) on Trainium2
        :link: /libraries/nxd-inference/tutorials/sd-inference-tutorial
        :link-type: doc
        :class-card: sd-rounded-3

        Implement speculative decoding techniques with Qwen3-32B on Trn2 instances to accelerate LLM inference with NxD Inference.
