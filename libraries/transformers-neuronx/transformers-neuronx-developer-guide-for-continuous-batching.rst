.. _transformers_neuronx_developer_guide_for_cb:

Transformers NeuronX (``transformers-neuronx``) Developer Guide For Continuous Batching
=======================================================================================

The continuous batching feature has been enabled with Transformers NeuronX.
Transformers NeuronX for Trn1 and Inf2 is a software package that enables
PyTorch users to perform large language model (LLM) :ref:`performant inference <neuron_llm_inference>` on
second-generation Neuron hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).
The :ref:`Neuron performance page <inf2-performance>` lists expected inference performance for commonly used Large Language Models.


Overview of continuous batching API and vLLM support
----------------------------------------------------

Transformers NeuronX supports continuous batching for ``LLaMA`` model class.

The basic flow for continuous batching support includes the following operations performed automatically by Transformers NeuronX:

1. Fill multiple prompts with context encoding by using virtual dynamic batching.
2. Decode each sequence until one of the sequences generates an EOS token.
3. Evict the finished sequence and insert a new prompt encoding.
4. Resume the decoding proces, repeating steps 2-3 until all of the sequences are decoded.

Installing vLLM and running a simple offline script
---------------------------------------------------

Once neuronx-cc and transformers-neuronx packages are installed, we will be able to install vLLM from source as follows:

.. code-block:: bash

   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout -b v0.3.3 v0.3.3
   touch ./vllm/model_executor/models/neuron/__init__.py
   pip install -U -r requirements-neuron.txt
   pip install .

.. note:

    Please note the vLLM pip package from PyPI is not compatible with Neuron. To work with Neuron, install vLLM using the source as outlined above.


If Neuron packages are detected correctly in the installation process, ``vllm-0.3.3+neuron213`` will be installed.


In the following example we demonstrate how to perform continuous batching with the ``LLaMA`` model.

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
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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


Known issues and FAQs
---------------------

**How to fix ModuleNotFoundError: No module named ‘vllm.model_executor.models.neuron’ ?**

Github issue: https://github.com/vllm-project/vllm/issues/3284

``pip install`` process may not copy neuron/llama.py into the site-packages directory.
This is due to the missing __init__.py in the neuron directory. The error looks like:

   ModuleNotFoundError: No module named ‘vllm.model_executor.models.neuron’

Besides, we need to add ``__init__.py`` file in the ``neuron`` directory **BEFORE** pip install, so that the directory would be copied in the pip install process. This is done using the ``touch`` Linux utility as shown in the installation steps above.

**Are other models than Llama supported?**

Currently, only LLaMA model support is upstreamed to vLLM. Support for other models like Mistral will be added in a future Neuron release.

**Is PagedAttention supported with vLLM integration?**

No, PagedAttention is not currently supported. It will be supported in a future Neuron release.
