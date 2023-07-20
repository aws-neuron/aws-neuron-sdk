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

Definition of model support status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Prototype (Alpha): An initial in-development version of a model that should be considered a preview of future functionality. A prototype may not be fully functional. A prototype model is not expected to perform well and may also have known accuracy issues. Prototype models may not maintain compatibility across versions.
- Experimental (Beta): A functional model which may still need performance & accuracy tuning. An experimental model should produce accurate results in most cases but is not yet considered stable. Prototype models may not maintain compatibility across versions.
- Stable: A model which has been validated for both accuracy and performance. Breaking changes to a stable models will occur with a deprecation notice in advance. 

.. list-table:: Model Support Status
   :widths: auto
   :header-rows: 1
   :align: left

   * - Model Support
     - Functional
     - Performance Tuned
     - Backwards Compatibility

   * - Prototype
     - No
     - No
     - No
   
   * - Experimental
     - Yes
     - No
     - No
   
   * - Stable
     - Yes
     - Yes
     - Yes
  
Release [0.5.0]
----------------------
Date: 7/19/2023

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
