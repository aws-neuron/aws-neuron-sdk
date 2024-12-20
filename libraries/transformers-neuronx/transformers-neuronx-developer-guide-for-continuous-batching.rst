.. _transformers_neuronx_developer_guide_for_cb:

Transformers NeuronX (``transformers-neuronx``) Developer Guide for Continuous Batching
=======================================================================================

Transformers NeuronX is integrated with vLLM to enable continuous batching for high-throughput 
LLM serving and inference. This guide aims to help users get started with continuous batching for
Transformers NeuronX and vLLM by providing:

- :ref:`Transformers NeuronX <cb-tnx-overview>` An overview of Transformers NeuronX.
- :ref:`cb-overview` The continuous batching procedure implemented by Transformers NeuronX and vLLM.
- :ref:`cb-install` Installation and usage instructions for Transformers NeuronX and vLLM.
- :ref:`cb-release-221-features` A showcase of new features in Transformers NeuronX and vLLM.
- :ref:`cb-faq`

.. _cb-tnx-overview:

Transformers NeuronX (``transformers-neuronx``)
-----------------------------------------------

Transformers NeuronX for Trn1 and Inf2 is a software package that enables
PyTorch users to perform large language model (LLM) :ref:`performant inference <neuron_llm_inference>` on
second-generation Neuron hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).
The :ref:`Neuron performance page <inf2-performance>` lists expected inference performance for commonly used Large Language Models.

.. _cb-overview:

Continuous Batching with Transformers NeuronX and vLLM
------------------------------------------------------

Transformers NeuronX implements the following operational flow with vLLM for continuous batching support:

1. Context encode multiple prompts using virtual dynamic batching.
2. Decode all sequences simultaneously until a sequence generates an EOS token.
3. Evict the finished sequence and insert a new prompt encoding.
4. Resume the decoding process, repeating steps 2 and 3 until all sequences are decoded.

.. _cb-supported-model-architectures:

Supported Model Architectures
-----------------------------

Transformers NeuronX supports continuous batching for models compatible with the following Hugging Face classes:

- ``LlamaForCausalLM``
- ``MistralForCausalLM``

.. _cb-install:

Install vLLM and Get Started with Offline Inference
---------------------------------------------------

Neuron maintains a fork of vLLM (v0.6.2) that contains the necessary changes to support inference with Transformers NeuronX.
Neuron is working with the vLLM community to upstream these changes to make them available in a future version.

Install vLLM
^^^^^^^^^^^^

First install ``neuronx-cc`` and the ``transformers-neuronx`` packages. Then install the vLLM fork from source:

.. code-block:: bash

    git clone -b v0.6.x-neuron https://github.com/aws-neuron/upstreaming-to-vllm.git
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" && pip install -e .

.. note::

    Please note the vLLM ``pip`` package from PyPI is not compatible with Neuron. To work with Neuron, install vLLM using the source as outlined above.

.. note::

    The current supported version of Pytorch for Neuron installs ``triton`` version ``2.1.0``. This is incompatible with ``vllm >= 0.5.3``. You may see an error ``cannot import name 'default_dump_dir...``. To work around this, run ``pip install --upgrade triton==3.0.0`` after installing the vLLM wheel.

If Neuron packages are detected correctly in the installation process, ``vllm-0.6.dev0+neuron215`` will be installed (The ``neuron`` version depends on the installed
``neuronx-cc`` version).

Run Offline Batched Inference with Transformers NeuronX and vLLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following example we demonstrate how to perform continuous batching with a Llama model.

.. note::

    Since Llama models are gated, please accept the Llama Community License Agreement and request access to the model.
    Then use a Hugging Face user access token to download the model.

.. code-block:: python

    from vllm import LLM, SamplingParams
    
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_num_seqs=8,
        # The max_model_len and block_size arguments are required to be same as max sequence length,
        # when targeting neuron device. Currently, this is a known limitation in continuous batching
        # support in transformers-neuronx.
        max_model_len=128,
        block_size=128,
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection, or explicitly assigned.
        device="neuron",
        tensor_parallel_size=2)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

Run the API Server
^^^^^^^^^^^^^^^^^^
To run the OpenAI-compatible API server in vLLM, run either command below:

.. code-block:: bash

    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 32 --max-num-seqs 16 --max-model-len 2048 --block-size 8 --override-neuron-config "shard_over_sequence:True"

.. code-block:: bash

    python3 -m vllm.entrypoints.openai.api_server meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 32 --max-num-seqs 16 --max-model-len 2048 --block-size 8 --override-neuron-config "shard_over_sequence:True"

.. _cb-release-221-features:

New Features in Neuron Release 2.21
-----------------------------------

Neuron's vLLM integration with Transformers NeuronX is tested using a public fork of vLLM v0.6.2.
New features and enhancements introduced in this fork will be described below.
Neuron's intent is to upstream these features to vLLM as soon as possible after release.
Prior to upstreaming, these features can be accessed in the AWS Neuron GitHub
repository https://github.com/aws-neuron/upstreaming-to-vllm/tree/v0.6.x-neuron.

**Neuron Release 2.21 Features for the v0.6.2 vLLM Neuron Fork**

- :ref:`Sequence bucketing <cb-sequence-bucketing>` configuration for context encoding and token generation.
- :ref:`Granular NeuronConfig control <cb-neuron-config-override>` in vLLM entrypoints.
- Inference support for :ref:`speculative decoding <cb-speculative-decoding>`.
- Inference support for :ref:`EAGLE speculative decoding <cb-eagle-speculative-decoding>`.

**Neuron Release 2.20 Features**

- Multi-node inference support for larger models. Example scripts are included in `vLLM <https://github.com/vllm-project/vllm/commit/e5a3c0904799ec8e04e25ac25e66024004a61533>`_ .
- Direct loading of Hugging Face-compatible checkpoints without creation of a ``-split`` directory.

.. _cb-sequence-bucketing:

Sequence Bucketing
^^^^^^^^^^^^^^^^^^
To configure buckets, set the following environment variables. Refer to the `developer guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html#bucketing>`_
for details on how to configure the values. These environment variables need to be set before starting the vLLM server or instantiating the ``LLM`` object.

- ``NEURON_CONTEXT_LENGTH_BUCKETS``:  Bucket sizes for context encoding.
- ``NEURON_TOKEN_GEN_BUCKETS``: Bucket sizes for token generation.

For example: ``export NEURON_CONTEXT_LENGTH_BUCKETS="128,512,1024"``


.. _cb-neuron-config-override:

NeuronConfig Override
^^^^^^^^^^^^^^^^^^^^^
The default ``NeuronConfig`` in vLLM uses the latest optimizations from the Neuron SDK. However, you can override the default values or add a new configuration from the `developer guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html#>`_ by setting the ``override_neuron_config`` parameter while creating the ``LLM`` object.

.. code-block:: python

    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_num_seqs=8,
        max_model_len=128,
        block_size=128
        device="neuron",
        tensor_parallel_size=32,
        #Override or update the NeuronConfig
        override_neuron_config={"shard_over_sequence":True})

While standing up the API server, set the ``override-neuron-config`` argument. For example:

.. code-block:: bash

    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 32 --max-num-seqs 16 --max-model-len 2048 --block-size 8 --override-neuron-config "shard_over_sequence:True"


.. _cb-quantization:

Quantization
^^^^^^^^^^^^
To use `int8 weight storage <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html#int8-weight-storage-support>`_ ,
set the environment variable ``NEURON_QUANT_DTYPE`` to ``s8``.


.. _cb-speculative-decoding:

Speculative Decoding
^^^^^^^^^^^^^^^^^^^^
Speculative decoding is a token generation optimization technique that
uses a small draft model to generate ``K`` tokens autoregressively and a
larger target model to determine which draft tokens to accept, all in a combined forward pass.
For more information on speculative decoding, please see `[Leviathan, 2023] <https://arxiv.org/abs/2211.17192>`_ and `[Chen et al., 2023] <https://arxiv.org/pdf/2302.01318>`_.

Speculative decoding is now available for inference with Transformers NeuronX and vLLM:

.. code-block:: python

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        speculative_model="meta-llama/Llama-3.2-1B-Instruct",
        # The max_model_len, speculative_max_model_len, and block_size arguments are required to be same as max sequence length,
        # when targeting neuron device. Currently, this is a known limitation in continuous batching
        # support in transformers-neuronx.
        max_model_len=128,
        block_size=128,
        speculative_max_model_len=128,
        dtype="bfloat16",
        max_num_seqs=4,
        num_speculative_tokens=4,
        # The device can be automatically detected when AWS Neuron SDK is installed.
        # The device argument can be either unspecified for automated detection, or explicitly assigned.
        device="neuron",
        tensor_parallel_size=32,
        use_v2_block_manager=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

.. note::

    Please ensure that the selected target and draft model are from the same model family. For example, if the target model is an instruction-tuned Llama model,
    the draft model must also be a lower-capacity instruction-tuned Llama model.

.. _cb-eagle-speculative-decoding:

EAGLE Speculative Decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^
Extrapolation Algorithm for Greater Language-model Efficiency (EAGLE) extends the speculative decoding
technique described above by:

- Utilizing a specially trained EAGLE draft model that predicts feature outputs through an Autoregression Head and next token outputs through an LM Head.
- Reducing sampling uncertainty by using the next autoregressively sampled token and a current feature map as draft model inputs.

For more information on EAGLE, please see `[Li et al., 2024] <https://arxiv.org/pdf/2401.15077>`_

EAGLE speculative decoding can be applied without changes to the speculative decoding code sample above. Transformers NeuronX and vLLM will recognize
a draft model as an EAGLE draft when ``is_eagle: True`` is set in the model's Hugging Face ``config.json`` file.


.. _cb-faq:

Frequently Asked Questions
--------------------------

**Is PagedAttention supported in the vLLM integration?**

No, PagedAttention is not currently supported. It will be supported in a future Neuron release.
