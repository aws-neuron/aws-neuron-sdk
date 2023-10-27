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

Model support status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Definition of model support status
.. ----------------------------------

.. - Prototype (Alpha): An initial in-development version of a model that should be considered a preview of future functionality. A prototype may not be fully functional. A prototype model is not expected to perform well and may also have known accuracy issues. Prototype models may not maintain compatibility across versions.
.. - Experimental (Beta): A functional model which may still need performance & accuracy tuning. An experimental model should produce accurate results in most cases but is not yet considered stable. Prototype models may not maintain compatibility across versions.
.. - Stable: A model which has been validated for both accuracy and performance. Breaking changes to a stable models will occur with a deprecation notice in advance. 

.. .. list-table::
..    :widths: auto
..    :header-rows: 1
..    :align: left

..    * - Model Support
..      - Functional
..      - Performance Tuned
..      - Backwards Compatibility

..    * - Prototype
..      - No
..      - No
..      - No
   
..    * - Experimental
..      - Yes
..      - No
..      - No
   
..    * - Stable
..      - Yes
..      - Yes
..      - Yes

Current model support status
-----------------------------

-  `BLOOM <https://huggingface.co/docs/transformers/model_doc/bloom>`__: [Beta]
-  `GPT2 <https://huggingface.co/docs/transformers/model_doc/gpt2>`__: [Beta]
-  `GPT-J <https://huggingface.co/docs/transformers/model_doc/gptj>`__: [Beta]
-  `GPT-Neox <https://huggingface.co/docs/transformers/model_doc/gpt_neox>`__: [Beta]
-  `LLaMA <https://huggingface.co/docs/transformers/main/model_doc/llama>`__: [Beta]
-  `LLaMA 2 <https://huggingface.co/docs/transformers/main/model_doc/llama2>`__: [Beta]
-  `OPT <https://huggingface.co/docs/transformers/model_doc/opt>`__: [Beta]

--------------------------
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
     - No

   * - GPT2
     - Yes
     - Partial
     - Partial

   * - GPT-J
     - No
     - No
     - No

   * - GPT-NeoX
     - No
     - No
     - No

   * - LLaMA
     - Yes
     - Yes
     - No

   * - LLaMA 2
     - Yes
     - Yes
     - No

   * - OPT
     - Yes
     - No
     - No


Release [0.8.268]
----------------------
Date: 10/26/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- [LLaMA] [Experimental] Added support for ``int8`` quantization for LLaMA.
- [BLOOM] [Experimental] Added multi bucket context encoding support for BLOOM.
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

- Some configurations of LLaMA and LLaMA-2 inference models fail compilation with the error ``IndirectLoad/Save requires contiguous indirect access per partition``. This is fixed in the compiler version 2.10.0.35 (Neuron SDK 2.14.1).
- Some configurations of LLaMA and LLaMA-2 inference model fail compilation with the error ``Too many instructions after unroll for function sg0000``. To mitigate this, please try with ``-O1`` compiler option (or ``--optlevel 1``) by adding ``os.environ["NEURON_CC_FLAGS"] = "-O1"`` to your script or set in the environment. A complete fix will be coming in the future release which will not require this option. Note: Using -O1 in the LLaMA-2 13B tutorial results in about 50% increase in latency compared to Neuron SDK 2.13.2. If this is not acceptable, please use compiler version from Neuron SDK 2.13.2.

Release [0.6.106]
----------------------
Date: 08/28/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Experimental] Added support for LLaMA 2 (excluding grouped/multi-query versions, such as LLaMA 2 70b)
- [Experimental] Improved the performance of BLOOM and LLaMA models
- Reduced execution latency of token generation in tensor parallel models by improving thread synchronization. (supported in LLaMA only) 
- Added an optimized vector implementation of RoPE positional embedding. (supported in LLaMA only)
- Added support for faster context encoding on sequences of varying lengths. This is implemented by allowing multiple buckets for parallel context encoding. During inference the best fit bucket is chosen. (supported in LLaMA/GPT-2 only)
- Added the Neuron Persistent Cache for compilation to automatically load pre-compiled model artifacts. (supported by all models)
- Improved compilation time by compiling models used for different sequence length buckets in parallel. (not supported in GPT-NeoX/GPT-J)

Resolved Issues
~~~~~~~~~~~~~~~

- [LLaMA] Fixed an issue in the parallel context encoding network where incorrect results could be generated if the context length is shorter than the context length estimate
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

- [Experimental] Added support for GPT-NeoX models.
- [Experimental] Added support for BLOOM models.
- [Prototype] Added support for LLaMA models.
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
