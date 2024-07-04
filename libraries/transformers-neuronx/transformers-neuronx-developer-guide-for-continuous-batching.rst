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

Overview of models supported
----------------------------

We support continuous batching for models compatible with the the following HuggingFace classes:

- LlamaForCausalLM
- MistralForCausalLM


Installing vLLM and running a simple offline script
---------------------------------------------------

Once neuronx-cc and transformers-neuronx packages are installed, we will be able to install vLLM from source.

.. code-block:: bash

   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout v0.5.0
   pip install .
   pip install ray

.. note::

    Please note the vLLM pip package from PyPI is not compatible with Neuron. To work with Neuron, install vLLM using the source as outlined above.

.. note::

    There is currently a known issue with offline inference examples that can be solved by applying the `vllm_v0.5.0_neuron.patch`_ patch
    to vLLM source after checking out v0.5.0 tag by running ``git apply vllm_v0.5.0_neuron.patch``. We are in the process of upstreaming the 
    fix for a future vLLM release.


If Neuron packages are detected correctly in the installation process, ``vllm-0.5.0+neuron214`` (The neuron version depends on the installed 
neuronx-cc version) will be installed.


In the following example we demonstrate how to perform continuous batching with a ``LLaMA`` model.

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

**How to fix 'AssertionError: Cache operations are not supported for Neuron backend.' ?**

Make sure the following patch is applied to vLLM v0.5.0 before installing. You can do so with
the following steps: Navigating into the vLLM source directory. Create a ``vllm_v0.5.0_neuron.patch`` file 
inside the vLLM source directory with the content shown below. Then run ``git apply vllm_v0.5.0_neuron.patch`` 
to apply the patch and install vllm via `pip install .`

.. _vllm_v0.5.0_neuron.patch:

.. code-block::

    diff --git a/vllm/executor/neuron_executor.py b/vllm/executor/neuron_executor.py
    index e7f0e887..87564b76 100644
    --- a/vllm/executor/neuron_executor.py
    +++ b/vllm/executor/neuron_executor.py
    @@ -48,9 +48,9 @@ class NeuronExecutor(ExecutorBase):
        def execute_model(
                self,
                execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
    -        assert (execute_model_req.blocks_to_swap_in == {}
    -                and execute_model_req.blocks_to_swap_out == {}
    -                and execute_model_req.blocks_to_copy == {}), (
    +        assert (not execute_model_req.blocks_to_swap_in
    +                and not execute_model_req.blocks_to_swap_out
    +                and not execute_model_req.blocks_to_copy), (
                        "Cache operations are not supported for Neuron backend.")
            assert execute_model_req.num_lookahead_slots == 0, (
                "lookahead not supported for Neuron backend.")

**Is PagedAttention supported with vLLM integration?**

No, PagedAttention is not currently supported. It will be supported in a future Neuron release.
