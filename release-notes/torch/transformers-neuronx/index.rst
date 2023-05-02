.. _OPT: https://huggingface.co/docs/transformers/model_doc/opt
.. _GPT2: https://huggingface.co/docs/transformers/model_doc/gpt2
.. _GPT-J: https://huggingface.co/docs/transformers/model_doc/gptj

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

Release [x.x.x.x.x.x]
----------------------
Date: 04/xx/2023

Summary
~~~~~~~

What's new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added ``transformers-neuronx`` artifacts to PyPI repository.
- Added support for the HuggingFace |generate|.
- Added support for model serialization, including model saving, loading, and
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
use other generation methods such as |sample| or |greedy_search|.
