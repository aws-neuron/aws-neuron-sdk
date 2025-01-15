.. _nxdi-vllm-user-guide:

vLLM User Guide for NxD Inference
=================================

`vLLM <https://docs.vllm.ai/en/latest/>`_ is a popular library for LLM inference and serving utilizing advanced inference features such as continuous batching.
This guide describes how to utilize AWS Inferentia and AWS Trainium AI accelerators in vLLM by using NxD Inference (``neuronx-distributed-inference``).

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

NxD Inference integrates into vLLM by extending the model execution components responsible
for loading and invoking models used in vLLM’s LLMEngine (see https://docs.vllm.ai/en/latest/design/arch_overview.html#llm-engine 
for more details on vLLM architecture). This means input processing, scheduling and output 
processing follow the default vLLM behavior. 

You enable the Neuron integration in vLLM by setting the device type used by vLLM to ``neuron``.

Currently, we support continuous batching and streaming generation in the NxD Inference vLLM integration.
We are working with the vLLM community to enable support for other vLLM features like PagedAttention
and Chunked Prefill on Neuron instances through NxD Inference in upcoming releases.


Supported Models
----------------

We currently support the following models in vLLM by default through NxD Inference:

* DBRX
* LLama 2, 3, 3.1, 3.2 (1B and 3B), 3.3
* Mistral-V2
* Mixtral

If you are adding your own model to NxD Inference, please see :ref:`Integrating Onboarded Model with vLLM<nxdi-onboarding-models-vllm>`
for instructions on how to setup vLLM integration for it.

Setup
-----

.. note::

    Currently, we maintain a fork of vLLM (v0.6.2) that contains the necessary changes to support NxD Inference.
    Therefore, it is important to follow the installation instructions below rather than installing vLLM
    from PyPI or the official vLLM repository. We are working with the vLLM community to upstream these
    changes to make them available in a future vLLM version.

Before installing vLLM with the instructions below, you need to install ``neuronx-distributed-inference``.

.. code::

    git clone -b v0.6.x-neuron https://github.com/aws-neuron/upstreaming-to-vllm.git
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" pip install -e .

Usage
-----

Neuron Framework Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    The Neuron integration for vLLM supports both Transformers NeuronX and NxD Inference libraries. If you have both libraries installed,
    by default Transformers NeuronX is used. To override this, please select the NxD Inference as desired Neuron framework by setting
    the ``VLLM_NEURON_FRAMEWORK`` environment variable to ``neuronx-distributed-inference`` before running vLLM.

If you are migrating from Transformers NeuronX to NxD Inference, you can refer to this :ref:`Migration Guide<nxdi_migrate_from_tnx>` for
additional support.

Quickstart
^^^^^^^^^^

Here is a quick and minimal example to get running.

.. code::

    import os
    os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

    from vllm import LLM, SamplingParams
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=8,
        max_model_len=128,
        device="neuron",
        tensor_parallel_size=2)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # note that top_k must be set to lower than the global_top_k defined in
    # the neuronx_distributed_inference.models.config.OnDeviceSamplingConfig
    sampling_params = SamplingParams(top_k=10, temperature=0.8, top_p=0.95)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


.. nxdi-vllm-model-configuration::

Model Configuration
^^^^^^^^^^^^^^^^^^^

NxD Inference models provide many configuration options. When using NxD Inference through vLLM,
we configure the model with a default configuration that sets the required fields from vLLM settings.
It is recommended that you do not override these configuration settings unless you need it.

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


If you want to add or change any settings, you can use vLLM's ``override_neuron_config`` setting. 
You provide the settings you want to override as dictionary (or JSON object when starting vLLM from the CLI)
containing basic types e.g. to disable auto bucketing (for illustration), use 

.. code:: ipython3
    
    override_neuron_config={
        "enable_bucketing":False,
    }

or when launching vLLM from the CLI

.. code::

    --override-neuron-config "{\"enable_bucketing\":false}"


For more information on NxD Inference features, see :ref:`NxD Inference Features Configuration Guide<nxdi-feature-guide>`
and :ref:`NxD Inference API Reference<nxd-inference-api-guide>`.

Scheduling and K/V Cache
^^^^^^^^^^^^^^^^^^^^^^^^

We currently use a contiguous memory layout for the K/V cache instead of PagedAttention support in NxD Inference.
We integrated into vLLMs block manager by setting the block size to the maximum length supported by the model
and allocating one block per maximum number of sequences configured. However, the vLLM scheduler currently does
not introspect the blocks associated to each sequence when (re-)scheduling running sequences. It requires an additional
free block regardless of space available in the current block resulting in preemption. This would lead to a large increase 
in latency for the preempted sequence because it would be rescheduled in the context encoding phase. Since we ensure each block
is big enough to fit the maximum model length, preemption is never needed in our current integration. 
Therefore, we disabled the preemption checks done by the scheduler in our fork. This significantly improves
E2E performance of the Neuron integration.

Decoding
^^^^^^^^

:ref:`On-device sampling<nxdi-on-device-sampling>` performs sampling logic on the Neuron devices 
rather than passing the generated logits back to CPU and sample through vLLM. This allows us to
use Neuron hardware to accelerate sampling and reduce the amount of data transferred between devices 
leading to improved latency.

However, on-device sampling comes with some limitations. Currently, we only support the following
sampling parameters: ``temperature``, ``top_k`` and ``top_p`` parameters. 
Other sampling parameters (https://docs.vllm.ai/en/latest/dev/sampling_params.html) are currently
not supported through on-device sampling.

When on-device sampling is enabled, we handle the following special cases:

* When ``top_k`` is set to -1, we limit ``top_k`` to 256 instead.
* When ``temperature`` is set to 0, we use greedy decoding to remain compatible with existing conventions. This is the same as setting ``top_k`` to 1.

By default, on-device sampling is disabled. You can enable on-device sampling by passing a ``on_device_sampling_config``
using the override neuron config feature (see :ref:`Model Configuration<nxdi-vllm-model-configuration>`). It is strongly recommended to make use
of the ``global_top_k`` configuration limiting the maximum value of ``top_k`` a user can request for improved performance.

Quantization
^^^^^^^^^^^^

NxD Inference supports quantization but has not yet been integrated with vLLMs configuration for quantization.
If you want to use quantization, **do not** set vLLM’s  ``--quantization`` setting to ``neuron_quant``. 
Keep it unset and use the Neuron configuration of the model to configure quantization of the NxD Inference model directly.
For more information on how to configure and use quantization with NxD Inference incl. requirements on checkpoints,
refer to :ref:`Quantization<nxdi-quantization>` in the NxD Inference Feature Guide.

Loading pre-compiled models / Serialization Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tracing and compiling the model can take a non-trivial amount of time depending on model size e.g. 
a small-ish model of 15GB might take around 15min to compile. Exact times depend on multiple factors.
Doing this on each server start would lead to unacceptable application startup times. 
Therefore, we support storing and loading the traced and compiled models.

Both are controlled through the ``NEURON_COMPILED_ARTIFACTS`` variable. When pointed to a path that contains a model,
we load it directly. If loading fails or if ``NEURON_COMPILED_ARTIFACTS`` is not set, then we recompile and store
the results in the provided location.

Examples
--------

The following examples use `meta-llama/Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_ on a ``Trn1.32xlarge`` instance. 

If you have access to the model checkpoint locally, replace ``meta-llama/Llama-3.1-8B-Instruct`` with the path to your local copy. 
Otherwise, you need to request access through HuggingFace and login via `huggingface-cli login <https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login>`_ using 
a `HuggingFace user access token <https://huggingface.co/docs/hub/en/security-tokens>`_ before running the examples. 

If you use a different instance type, you need to adjust the ``tp_degree`` according to the number of Neuron Cores 
available on your instance type (for more information see: :ref:`Tensor-parallelism support<nxdi-tensor-parallelism>`).

Offline Inference Example
^^^^^^^^^^^^^^^^^^^^^^^^^

Here is an example for running offline inference. :ref:`Bucketing<nxdi-bucketing>` is only disabled to demonstrate 
how to override Neuron configuration values. Keeping it enabled generally delivers better
performance.

.. code:: ipython3

    import os
    os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1)

    # Create an LLM.
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_seqs=4,
        max_model_len=128,
        override_neuron_config={
            "enable_bucketing":False,
        },
        device="neuron",
        tensor_parallel_size=32)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

Online Inference Example
^^^^^^^^^^^^^^^^^^^^^^^^

You can start an OpenAI API compatible server with the same settings as the offline example by running
the following command:

.. code::

    VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' python -m vllm.entrypoints.openai.api_server \
        --model="meta-llama/Llama-3.1-8B-Instruct" \
        --max-num-seqs=4 \
        --max-model-len=128 \
        --tensor-parallel-size=8 \
        --port=8080 \
        --device "neuron" \
        --override-neuron-config "{\"enable_bucketing\":false}"

In addition to the sampling parameters supported by OpenAI, we also support ``top_k``.
You can change the sampling parameters and enable or disable streaming.

.. code::

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
    max_tokens = 1024
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
