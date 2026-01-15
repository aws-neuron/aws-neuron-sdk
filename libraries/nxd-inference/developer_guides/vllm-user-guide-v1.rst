.. _nxdi-vllm-user-guide-v1:

vLLM User Guide for NxD Inference
============================================

`vLLM <https://docs.vllm.ai/en/latest/>`_ is a popular library for LLM inference and serving utilizing advanced inference features such as continuous batching.
This guide describes how to utilize AWS Inferentia and AWS Trainium AI accelerators in vLLM by using NxD Inference (``neuronx-distributed-inference``).

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

NxD Inference integrates with vLLM by using `vLLM's Plugin System <https://docs.vllm.ai/en/latest/design/plugin_system.html>`_ to extend the model execution components responsible for loading and invoking models within vLLM's LLMEngine (see `vLLM architecture <https://docs.vllm.ai/en/latest/design/arch_overview.html#llm-engine>`_ 
for more details). This means input processing, scheduling and output 
processing follow the default vLLM behavior.

Versioning
^^^^^^^^^^

Plugin Version: ``0.2.2+lts``

Neuron SDK Version: ``2.27.0``

vLLM Version: ``0.11.0``

PyTorch Version: ``2.8.0``


Supported Models
----------------

The following models are supported on vLLM with NxD Inference:

- Llama 2/3.1/3.3
- Llama 4 Scout, Maverick
- Qwen 2.5
- Qwen 3
- Pixtral (limited, see Known Issues)

If you are adding your own model to NxD Inference, see :ref:`Integrating Onboarded Model with vLLM<nxdi-onboarding-models-vllm>`.

  
Setup
-----

Prerequisite: Launch an instance and install drivers and tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before installing vLLM with the instructions below, you must launch an Inferentia or Trainium instance and install the necessary
Neuron SDK dependency libraries. We recommend using a Neuron Deep Learning Container (DLC) for the best compatibility. 
Refer to :ref:`these setup instructions<nxdi-setup>` for information on using Neuron DLCs.


**Prerequisites:**

- Latest AWS Neuron SDK (`Neuron SDK 2.27.0 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.27.0.html>`_)
- Python 3.10+ (compatible with vLLM requirements)
- Supported AWS instances: Inf2, Trn1/Trn1n, Trn2

Installing the AWS Neuron fork of vLLM 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Neuron maintains a vLLM-Neuron Plugin that supports the latest features for NxD Inference. Follow the instructions below to obtain and configure it.

Quickstart using Docker
"""""""""""""""""""""""""""

You can use a preconfigured Deep Learning Container (DLC) with the AWS vLLM-Neuron plugin pre-installed.
Refer to the `vllm-inference-neuronx container <https://github.com/aws-neuron/deep-learning-containers?tab=readme-ov-file#vllm-inference-neuronx>`_
on `https://github.com/aws-neuron/deep-learning-containers <https://github.com/aws-neuron/deep-learning-containers>`_ to get started.

For a complete step-by-step tutorial on deploying the vLLM Neuron DLC, see :ref:`quickstart_vllm_dlc_deploy`.

Manually install from source
"""""""""""""""""""""""""""""""

Install the plugin from GitHub sources using the following commands. The plugin will automatically install the correct version of vLLM along with other required dependencies.
This version of the plugin is intended to work with the Neuron SDK 2.27.0, PyTorch 2.8, and vLLM 0.11.0. This is not needed if using a DLC container with the vllm-neuron plugin already installed.

.. code-block:: bash

    git clone --branch "0.2.2+lts" https://github.com/vllm-project/vllm-neuron.git
    cd vllm-neuron
    pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .


Usage
-----

Quickstart
^^^^^^^^^^^^

Here is a very basic example to get started:

.. code-block:: python

   from vllm import LLM, SamplingParams

   # Initialize the model
   llm = LLM(
       model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
       max_num_seqs=4,
       max_model_len=128,
       tensor_parallel_size=2,
       block_size=32
   )

   # Generate text
   prompts = [
       "Hello, my name is",
       "The president of the United States is",
       "The capital of France is",
   ]
   sampling_params = SamplingParams(temperature=0.0)
   outputs = llm.generate(prompts, sampling_params)

   for output in outputs:
       print(f"Prompt: {output.prompt}")
       print(f"Generated: {output.outputs[0].text}")

Feature Support
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Feature
     - Status
     - Notes
   * - Continuous batching
     - 🟢
     -
   * - Prefix Caching
     - 🟢
     - 
   * - Multi-LORA
     - 🟢
     - 
   * - Speculative Decoding
     - 🟢
     - Only Eagle V1 is supported
   * - Quantization
     - 🟢
     - INT8/FP8 quantization support
   * - Dynamic sampling	
     - 🟢
     -
   * - Tool calling
     - 🟢
     -
   * - CPU Sampling
     - 🟢
     -
   * - Chunked Prefill
     - 🚧
     - 
   * - Multimodal
     - 🚧
     - Llama4 and Pixtral are supported

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🚧 WIP: Under active development.

Feature Configuration
^^^^^^^^^^^^^^^^^^^^^^

NxD Inference models provide many configuration options. When using NxD Inference through vLLM,
you configure the model with a default configuration that sets the required fields from vLLM settings.

.. code:: ipython3

    neuron_config = dict(
        tp_degree=parallel_config.tensor_parallel_size,
        ctx_batch_size=1,
        batch_size=scheduler_config.max_num_seqs,
        max_context_length=scheduler_config.max_model_len,
        seq_len=scheduler_config.max_model_len,
        enable_bucketing=True,
        is_continuous_batching=True,
        quantized=False,
        torch_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        padding_side="right"
    )


Use the ``additional_config`` field to provide an ``override_neuron_config`` dictionary that specifies your desired NxD Inference configuration settings. You provide the settings you want to override as a dictionary (or JSON object when starting vLLM from the CLI) containing basic types. For example, to enable prefix caching:

.. code:: ipython3
    
    additional_config=dict(
        override_neuron_config=dict(
            is_prefix_caching=True,
            is_block_kv_layout=True,
            pa_num_blocks=4096,
            pa_block_size=32,
        )
    )

or when launching vLLM from the CLI

.. code:: bash

    --additional-config '{
        "override-neuron-config": {
            "is_prefix_caching": true,
            "is_block_kv_layout": true,
            "pa_num_blocks": 4096,
            "pa_block_size": 32
        }
    }'


For more information on NxD Inference features, see :ref:`NxD Inference Features Configuration Guide<nxdi-feature-guide>`
and :ref:`NxD Inference API Reference<nxd-inference-api-guide>`.

Scheduling and K/V Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^

NxD Inference uses a contiguous memory layout for the K/V cache instead of PagedAttention support.
It integrates into vLLM's block manager by setting the block size to the maximum length supported by the model
and allocating one block per maximum number of sequences configured. However, the vLLM scheduler currently does
not introspect the blocks associated to each sequence when (re-)scheduling running sequences. The scheduler requires an additional
free block regardless of space available in the current block resulting in preemption. This would lead to a large increase 
in latency for the preempted sequence because it would be rescheduled in the context encoding phase. Since NxD Inference's implementation ensures each block
is big enough to fit the maximum model length, preemption is never needed in our current integration. 
As a result, AWS Neuron disabled the preemption checks done by the scheduler in our fork. This significantly improves
E2E performance of the Neuron integration.

Decoding
^^^^^^^^^^

On-device sampling is enabled by default, which performs sampling logic on the Neuron devices 
rather than passing the generated logits back to CPU and sample through vLLM. This allows you to
use Neuron hardware to accelerate sampling and reduce the amount of data transferred between devices 
leading to improved latency.

However, on-device sampling comes with some limitations. Currently, we only support the following
sampling parameters: ``temperature``, ``top_k`` and ``top_p`` parameters. 
Other `sampling parameters <https://docs.vllm.ai/en/latest/dev/sampling_params.html>`_ are currently
not supported through on-device sampling.

When on-device sampling is enabled, we handle the following special cases:

* When ``top_k`` is set to -1, we limit ``top_k`` to 256 instead.
* When ``temperature`` is set to 0, we use greedy decoding to remain compatible with existing conventions. This is the same as setting ``top_k`` to 1.

By default, on-device sampling utilizes a greedy decoding strategy to select tokens with the highest probabilities. 
You can enable a different on-device sampling strategy by passing a ``on_device_sampling_config``
using the override neuron config feature (see :ref:`Model Configuration<nxdi-vllm-model-configuration>`). It is strongly recommended to make use
of the ``global_top_k`` configuration limiting the maximum value of ``top_k`` a user can request for improved performance.

Quantization
^^^^^^^^^^^^^^

NxD Inference supports quantization but has not yet been integrated with vLLM's configuration for quantization.
If you want to use quantization, **do not** set vLLM's ``--quantization`` setting to ``neuron_quant``. 
Keep it unset and use the Neuron configuration of the model to configure quantization of the NxD Inference model directly.
For more information on how to configure and use quantization with NxD Inference incl. requirements on checkpoints,
refer to :ref:`Quantization<nxdi-quantization>` in the NxD Inference Feature Guide.

.. _nxdi-vllm-v1-serialization:

Loading pre-compiled models / Serialization Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tracing and compiling the model can take a non-trivial amount of time depending on model size e.g. 
a small-ish model of 15GB might take around 15min to compile. Exact times depend on multiple factors.
Doing this on each server start would lead to unacceptable application startup times. 
Therefore, we support storing and loading the traced and compiled models.

Both are controlled through the ``NEURON_COMPILED_ARTIFACTS`` variable. When pointed to a path that contains a pre-compiled model,
we load the pre-compiled model directly, and any differing model configurations passed in to the vllm API will not trigger re-compilation. 
If loading from the ``NEURON_COMPILED_ARTIFACTS`` path fails, then we will recompile the model with the provided configurations and store 
the results in the provided location. If ``NEURON_COMPILED_ARTIFACTS`` is not set, we will compile the model and store it under a ``neuron-compiled-artifacts``
subdirectory in the directory of your model checkpoint.

Prefix Caching
^^^^^^^^^^^^^^^^

Starting in Neuron SDK 2.24, prefix caching is supported on the AWS Neuron fork of vLLM. Prefix caching allows developers to improve TTFT by 
re-using the KV Cache of the common shared prompts across inference requests. See :ref:`Prefix Caching <nxdi_prefix_caching>` for more information on how to 
enable prefix caching with vLLM. 


Examples
--------

For more in depth NxD Inference tutorials that include vLLM deployment steps, refer to :ref:`Tutorials <nxdi-tutorials-index>`.

The following examples use `TinyLlama/TinyLlama-1.1B-Chat-v1.0 <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>`_

If you have access to the model checkpoint locally, replace ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` with the path to your local copy. 

If you use a different instance type, you need to adjust the ``tensor_parallel_size`` according to the number of Neuron Cores 
available on your instance type. (For more information see: :doc:`Tensor-parallelism support </libraries/nxd-inference/app-notes/parallelism>`.)

Offline Inference Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For offline inference, refer to the code example in the Quickstart section above.

Online Inference Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can start an OpenAI API compatible server with the same settings as the offline example by running
the following command:

.. code:: bash

    python3 -m vllm.entrypoints.openai.api_server \
        --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
        --tensor-parallel-size 2 \
        --max-model-len 128 \
        --max-num-seqs 4 \
        --block-size 32 \
        --port 8000

In addition to the sampling parameters supported by OpenAI, we also support ``top_k``.
You can change the sampling parameters and enable or disable streaming.

.. code:: python

    from openai import OpenAI

    # Client Setup
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model_name = models.data[0].id

    # Sampling Parameters
    max_tokens = 64
    temperature = 1.0
    top_p = 1.0
    top_k = 50
    stream = False

    # Chat Completion Request
    prompt = "Hello, my name is Llama "
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stream=stream,
        extra_body={'top_k': top_k}
    )

    # Parse the response
    generated_text = ""
    if stream:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content
    else:
        generated_text = response.choices[0].message.content
        
    print(generated_text)


Known Issues
---------------

1. Chunked prefill is disabled by default on Neuron for optimal performance. To enable chunked prefill, set the environment variable ``DISABLE_NEURON_CUSTOM_SCHEDULER="1"``.
   
   * Users are required to provide a ``num_gpu_blocks_override`` arg, which should be at least ``ceil(max_model_len // block_size) * max_num_seqs`` when invoking vLLM to avoid a potential OOB error.

2. When using HuggingFace model IDs with both `shard on load <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/weights-sharding-guide.html#shard-on-load>`_ and models that have ``tie_word_embeddings`` set to ``true`` in their config (such as `Qwen3-8B <https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json#L24>`_), you may encounter the error ``NotImplementedError: Cannot copy out of meta tensor; no data!``. To resolve this, download the model checkpoint locally from Hugging Face and serve it from the local path instead of using the HuggingFace model ID.
3. For vLLM version 0.11.0 there is a bug where chat templates are not cached. This affects request preprocessing time. This is fixed in future vLLM versions.
4. Async tokenization in vLLM V1 can increase request preprocessing time for small inputs and batch sizes. The Neuron team is investigating potential solutions.
5. Pixtral has out of bounds issues for batch sizes greater than 4. The max sequence length is 10240.

Support
----------

- **Documentation**: `AWS Neuron Documentation <https://awsdocs-neuron.readthedocs-hosted.com/>`_
- **Issues**: `GitHub Issues <https://github.com/vllm-project/vllm-neuron/issues>`_
- **Community**: `AWS Neuron Forum <https://repost.aws/tags/TAjy-krivRTDqDPWNNBmV9lA>`_