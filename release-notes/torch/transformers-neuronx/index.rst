.. _OPT: https://huggingface.co/docs/transformers/model_doc/opt
.. _GPT2: https://huggingface.co/docs/transformers/model_doc/gpt2
.. _GPT-J: https://huggingface.co/docs/transformers/model_doc/gptj
.. _Tensor-parallelism-support: https://github.com/aws-neuron/transformers-neuronx/blob/main/README.md#tensor-parallelism-support
.. _features-support: https://github.com/aws-neuron/transformers-neuronx/blob/main/README.md#Currently-supported-models-and-features

.. |generate| replace:: :py:meth:`~transformers.generation_utils.GenerationMixin.generate`
.. |beam_search| replace:: :meth:`~transformers.generation_utils.GenerationMixin.beam_search`
.. |sample| replace:: :meth:`~transformers.generation_utils.GenerationMixin.sample`
.. |greedy_search| replace:: :meth:`~transformers.generation_utils.GenerationMixin.greedy_search`

.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
.. |Inf2| replace:: :ref:`Inf2 <aws-inf2-arch>`

.. _transformers-neuronx-rn:

Transformers Neuron (``transformers-neuronx``) release notes
============================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

Transformers Neuron for |Trn1|/|Inf2| is a software package that enables
PyTorch users to perform large language model (LLM) inference on
second-generation Neuron hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).

Model classes status
------------------------------

-  `BLOOM <https://huggingface.co/docs/transformers/model_doc/bloom>`__: [Beta]
-  `GPT2 <https://huggingface.co/docs/transformers/model_doc/gpt2>`__: [Beta]
-  `GPT-J <https://huggingface.co/docs/transformers/model_doc/gptj>`__: [Beta]
-  `GPT-Neox <https://huggingface.co/docs/transformers/model_doc/gpt_neox>`__: [Beta]
-  `Llama <https://huggingface.co/docs/transformers/main/model_doc/llama>`__: [Beta]
-  `Llama 2 <https://huggingface.co/docs/transformers/main/model_doc/llama2>`__: [Beta]
-  `Mistral <https://huggingface.co/docs/transformers/main/model_doc/mistral>`__: [Beta]


Model features
--------------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left

   * - Model
     - Flexible Tensor Parallelism
     - Prompt Estimate Support
     - Serialization Support

   * - BLOOM
     - Yes
     - Yes
     - Yes

   * - GPT2
     - Yes
     - Partial
     - Yes

   * - GPT-J
     - No
     - No
     - No

   * - GPT-NeoX
     - No
     - No
     - No

   * - Llama
     - Yes
     - Yes
     - Yes

   * - Llama 2
     - Yes
     - Yes
     - Yes

   * - Llama 3.1
     - Yes
     - Yes
     - Yes     

   * - Mistral
     - Yes
     - Yes
     - Yes

Release [0.13.380.0]
----------------------
Date: 01/14/2025

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- The transformers depedency has been pinned to ``transformers<4.48``


Release [0.13.322.0]
----------------------
Date: 12/20/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Flash decoding support for speculative decoding
- Enabled on-device generation support in speculative decoding flows	
- Added support for EAGLE speculative decoding support with greedy and lossless sampling
- Support for CPU compilation and sharded model saving


Performance Improvements
~~~~~~~~~~~~~~~~~~~~~~~~
- Performance optimized MLP and QKV kernels added for llama models with support for sequence parallel norm
- Added support to control concurrent compilation workers
- Added option to skip AllGather using duplicate Q weights during shard over sequence


Resolved Issues
~~~~~~~~~~~~~~~

- Fixed padding issues when requested batch size is smaller than neff compiled size	
- Fixed sequence parallel norm issue when executor is used with speculative decoding flows

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
- Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.


Release [0.12.313]
----------------------
Date: 09/16/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Support for model serialization (save and load) of all models except the ``GPTJForSampling`` and ``GPTNeoXForSampling``` model classes, which reduces future model load time by saving a transformed and sharded set of weights as a new safetensors checkpoint.
- Support for on device sampling (Top P) with Continuous batching
- Support for Scaled RoPE for LLAMA 3.1 models
- Support for multi-node inference for LLAMA 3.1 405B model for specific sequence lengths
- Support for FlashDecoding (using ``shard_over_sequence``) for supporting long context lengths upto 128k   `Tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-128k-sampling.ipynb>`__


Resolved Issues
~~~~~~~~~~~~~~~

- Fixes to handle ``seq_ids`` consistently across vLLM versions
- Fixes for KV head full replication logic errors

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
- Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.


Release [0.11.351.0]
----------------------
Date: 07/03/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Support for compiler optimized flash attention kernel to support context lengths of 16k/32k for Llama models
- Streamer support enabled for BLOOM, GPTJ, GPT2, GPT-NeoX and LLAMA models
- Support for on device generation for TopK in Mixtral models
- Continuous batching support for Mistral v0.2
- Minor API improvements with type annotations for NeuronConfig, end-of-support warnings for old arguments, and exposing top-level configurations

- Performance improvements such as an optimized logit ordering for continuous batching in Llama models, optimized QKV padding for certain GQA models, faster implementation of cumsum operation to improve TopP performance
  
Resolved Issues
~~~~~~~~~~~~~~~

- Removed ``start_ids=None`` from ``generate()``
- Mistral decoding issue that occurs during multiple sampling runs
- Mistralv0.1 sliding window error
- Off-by-one error in window context encoding
- Better error messaging

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``on_device_generation=GenerationConfig(do_sample=True)`` has some known failures for Llama models. Customers are advised not to use ``on_device_generation`` in such cases.
- GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
- Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use ``constants.LAYOUT_SBH`` instead.

Release [0.10.0.332]
----------------------
Date: 04/10/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Beta] Added support for continuous batching and a reference integration with vLLM (Llama models only)

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
- There is a known compiler issue for inference of some configurations of Llama-2 13B that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if Llama-2 13B accuracy issues occur.
- There is a known compiler issue for inference of some configurations of GPT-2 that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if GPT-2 accuracy issues occur.
- GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.
- Using ``cache_layout=constants.LAYOUT_BSH`` in NeuronConfig has known limitations with compilation. Customers are advised to use constants.LAYOUT_SBH instead.


Release [0.10.0.21]
----------------------
Date: 04/01/2024

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for on device log-softmax and on device sampling for TopK
- Added support for on device embedding for all models.
- Added support for Speculative Decoding
- [Beta] Added support for Mixtral-8x7b MoE
- [Beta] Added support for mistralai/Mistral-7B-Instruct-v0.2 with no sliding window
- Added faster checkpoint loading support for both sharded and whole checkpoints
- Added the ability to download checkpoints directly from huggingface hub repositories
- Added NeuronAutoModelForCausalLM class which automatically loads architecture-specific classes
- Added a warmup to all kernels to avoid unexpected initialization latency spikes
  
Resolved Issues
~~~~~~~~~~~~~~~

- Users no longer need a copy of the original checkpoint and can use safetensor checkpoints for optimal speed.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
- There is a known compiler issue for inference of some configurations of Llama-2 13B that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if Llama-2 13B accuracy issues occur.
- There is a known compiler issue for inference of some configurations of GPT-2 that can cause accuracy degredation. Customers are advised to use the ``--enable-saturate-infinity --enable-mixed-precision-accumulation`` compiler flags if GPT-2 accuracy issues occur.
- GPT-NeoX is sensitive to ``fp16`` and customers are advised to use only ``amp="f32"`` for GPT-NeoX.

Release [0.9.474]
----------------------
Date: 12/21/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Llama] [Beta] Added support for Llama-2 70B.
- [Mistral] [Beta] Added support for Mistral 7B.
- [Beta] Added support for PyTorch 2.1.
- [Beta] Added support for Grouped Query Attention (GQA).
- [Beta] Added support for ``safetensors`` serialization.
- [Llama] [Beta] Added support for early stopping in the ``sample_llama`` function.
- [GPT2] [Beta] Added sparse attention support.
- [Stable] Added support for ``BatchNorm``.
- Use the ``--auto-cast=none`` compiler flag by default for all models. This flag improves accuracy for ``float32`` operations.

Resolved Issues
~~~~~~~~~~~~~~~

- Resolved an issue in ``top_p`` in the ``sample_llama`` function so that it now selects the same number of tokens that the Hugging Face ``top_p`` implementation selects.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- There is a known compiler issue for inference of some configurations of Llama-2 70B that can cause accuracy degredation. Customers are advised to use the ``--enable-mixed-precision-accumulation`` compiler flag if Llama-2 70B accuracy issues occur.
- There are known compiler issues impacting inference accuracy of certain model configurations of ``Llama-2-13b`` when ``amp = fp16`` is used. If this issue is observed, ``amp=fp32`` should be used as a work around.  This issue will be addressed in future Neuron releases.

Release [0.8.268]
----------------------
Date: 10/26/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Llama] [Beta] Added support for ``int8`` quantization for Llama.
- [BLOOM] [Beta] Added multi bucket context encoding support for BLOOM.
- [Beta] Added model Serialization for all supported models (except GPT-J and GPT-NeoX).
- [Beta] Added the ability to return output logit scores during sampling.
- [Stable] Added support for ``SOLU`` activation and ``GroupNorm``.

Resolved Issues
~~~~~~~~~~~~~~~

- [GPT2] Fixed an issue in ``GPT2ForSamplingWithContextBroadcasting`` where the input prompt would get truncated if it was longer than the ``context_length_estimate``.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Release [0.7.84]
----------------------
Date: 09/15/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use the ``--model-type=transformer`` compiler flag by default for all models. This flag improves performance and compilation time for all models. This flag replaces the ``--model-type=transformer-inference`` flag, which is now depracated.

Resolved Issues
~~~~~~~~~~~~~~~

- Fixed an issue where the ``HuggingFaceGenerationModelAdapter`` class falls back to serial context encoding for models that have parallel context encoding (``GPT2ForSamplingWithContextBroadcasting``, ``LlamaForSampling``, etc.)
- [GPT2 / OPT] Fixed an issue in the parallel context encoding network where incorrect results could be generated due to incorrect masking logic.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Some configurations of Llama and Llama-2 inference models fail compilation with the error ``IndirectLoad/Save requires contiguous indirect access per partition``. This is fixed in the compiler version 2.10.0.35 (Neuron SDK 2.14.1).
- Some configurations of Llama and Llama-2 inference model fail compilation with the error ``Too many instructions after unroll for function sg0000``. To mitigate this, please try with ``-O1`` compiler option (or ``--optlevel 1``) by adding ``os.environ["NEURON_CC_FLAGS"] = "-O1"`` to your script or set in the environment. A complete fix will be coming in the future release which will not require this option. Note: Using -O1 in the Llama-2 13B tutorial results in about 50% increase in latency compared to Neuron SDK 2.13.2. If this is not acceptable, please use compiler version from Neuron SDK 2.13.2.

Release [0.6.106]
----------------------
Date: 08/28/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for Llama 2 (excluding grouped/multi-query versions, such as Llama 2 70B) [Beta]
- Improved the performance of BLOOM and Llama models [Beta]
- Reduced execution latency of token generation in tensor parallel models by improving thread synchronization. (supported in Llama only) 
- Added an optimized vector implementation of RoPE positional embedding. (supported in Llama only)
- Added support for faster context encoding on sequences of varying lengths. This is implemented by allowing multiple buckets for parallel context encoding. During inference the best fit bucket is chosen. (supported in Llama/GPT-2 only)
- Added the Neuron Persistent Cache for compilation to automatically load pre-compiled model artifacts. (supported by all models)
- Improved compilation time by compiling models used for different sequence length buckets in parallel. (not supported in GPT-NeoX/GPT-J)

Resolved Issues
~~~~~~~~~~~~~~~

- [Llama] Fixed an issue in the parallel context encoding network where incorrect results could be generated if the context length is shorter than the context length estimate
- [GPT2 / OPT] Fixed an issue in the parallel context encoding network where incorrect results could be generated

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The ``HuggingFaceGenerationModelAdapter`` class currently falls back to serial context encoding for models that have parallel context encoding (``GPT2ForSamplingWithContextBroadcasting``, ``LlamaForSampling``, etc. )
- Beam search can introduce memory issues for large models
- There can be accuracy issues for the GPT-J model for certain use-cases
  
Release [0.5.58]
----------------------
Date: 7/21/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added support for GPT-NeoX models [Beta].
- Added support for BLOOM models [Beta].
- Added support for Llama models [Alpha].
- Added support for more flexible tensor-parallel configurations to GPT2, OPT, and BLOOM. The attention heads doesn't need to be evenly divisible by `tp_degree` anymore. (Note: The `tp_degree` still needs to satisfy the runtime topologies constraint for collective communication (i.e Allreduce). For more details on supported topologies, see: `Tensor-parallelism-support`_ and https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/collective-communication.html.)
- Added multi-query / multi-group attention support for GPT2.

Resolved Issues
~~~~~~~~~~~~~~~

- Fixed NaN issues for GPT2 model.
- Fixed OPT/GPT-NeoX gibberish output.
- Resolved an issue where NaN values could be produced when the context_length argument was used in GPT2/OPT.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Missing cache reorder support for beam search.
- For more info, please see `features-support`_.

Release [0.4.0]
----------------------
Date: 6/14/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added ``int8`` weight storage for `GPT2`_ models.
- Improved prompt context encoding performance for `GPT2`_ models.
- Improved collective communications performance for tp-degrees 4, 8, and 24 on Inf2.
- Improved collective communications performance for tp-degrees 8 and 32 on Trn1.
- Support for the ``--model-type=transformer-inference`` compiler flag for optimized decoder-only LLM inference.

Resolved Issues
~~~~~~~~~~~~~~~

Incorrect `GPT-J`_ ``linear`` layer sharding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added padding to the `GPT-J`_ ``linear`` layer to correctly handle odd vocabulary sizes. 

Incorrect output with HuggingFace |beam_search|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Issues where the HuggingFace |generate| method produces incorrect results when
|beam_search| is used have been resolved.


Release [0.3.0]
----------------------
Date: 05/01/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added ``transformers-neuronx`` artifacts to PyPI repository.
- Added support for the HuggingFace |generate|.
- Added model serialization support for GPT2 models, including model saving, loading, and
  weight swapping.
- Added support for caching compiled artifacts.
- Improved performance by removing unnecessary KV-cache tensor resetting.
- Improved prompt context encoding performance (`OPT`_, `GPT2`_).

Resolved Issues
~~~~~~~~~~~~~~~

Incorrect `GPT-J`_ ``amp_callback`` import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fixed the `GPT-J`_ demo to import the correct ``amp_callback`` function.

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Incorrect output with HuggingFace |beam_search|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the HuggingFace |generate| method is configured to use |beam_search|, this
can produce incorrect results for certain configurations. It is recommended to
use other generation methods such as |sample| or |greedy_search|. This will be
fixed in a future Neuron release.
