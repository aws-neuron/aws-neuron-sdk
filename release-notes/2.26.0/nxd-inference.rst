.. _neuron-2-26-0-nxd-inference:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Transformers for Inference component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: NxD Inference release notes
==================================================

**Date of release**:  September 18, 2025

**Version**: 0.6.10598

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Improvements
------------

Llama 4 model support (beta)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added beta support for Llama 4, which is a family of multi-modal MoE ope- weight LLMs by Meta that support text
and image inputs. Llama 4 is tested on ``Trn2``. Compatible models include:

- `Llama 4 Scout <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct>`__
- `Llama 4 Maverick <https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct>`__

In this beta release, Llama 4 model support has the following limitations:

- The model is tested to be accurate up to a sequence length of 8192.
- Model performance on Trn2 isn't fully optimized.
- To use Llama 4 with vLLM, you must compile the model outside of vLLM and specify
  the compiled model path using the ``NEURON_COMPILED_ARTIFACTS`` environment variable.

These limitations will be addressed in a future release.

For more information, see :ref:`/libraries/nxd-inference/tutorials/llama4-tutorial.ipynb`
and :ref:`nxdi-model-reference`.

FLUX.1 model support (beta)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added beta support for FLUX.1-dev, which is an open weight image generation model
by Black Forest Labs. Flux.1-dev is tested on Trn2. Compatible models include:

- `Flux.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`__

In this beta release, the model's performance isn't optimized.

For more information, see :ref:`/libraries/nxd-inference/tutorials/flux-inference-tutorial.ipynb`
and :ref:`nxdi-model-reference`.

Expert parallelism support (beta)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added support for expert parallelism, which distributes expert processing across multiple
NeuronCores. Expert parallelism improves performance for mixture-of-experts (MoE) models,
particularly for models with a large number of experts, such as Llama 4 Maverick. For more
information, see :ref:`nxd-inference-api-guide-moe-neuron-config`.

Context parallelism improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With this release, context parallelism is out of beta and includes several improvements.

- Added support for sliding window attention (SWA) with context parallelism.
- Added a strided context parallel flash attention kernel which includes compute elimination.
  This kernel is more performant than the existing content parallel flash attention kernel,
  especially at high sequence lengths. To use the kernel,
  enable ``strided_context_parallel_kernel_enabled`` in NeuronConfig.
- Fixed an accuracy issue in hybrid sharding configurations that use context parallelism
  and attention bias. Hybrid sharding refers to models with different sharding strategies
  for context encoding and token generation submodels, such as a configuration that uses
  context parallelism for context encoding and data parallelism for token generation.
..
  Sliding window attention (SWA)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
..
  Added support for sliding window attention, including support for attention sinks. Sliding window
  attention improves attention performance by attending to a subset of recent tokens, rather than the
  full context.
..
  NxD Inference uses the ``sliding_window`` attribute from the model config as the window size. The
  ``sliding_window`` attribute is typically set in the Hugging Face checkpoint config, so NxD Inference
  automatically enables sliding window attention for models trained with it.

On-device forward pipeline execution (Beta)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added support for a model-forward function that accepts both on-device abnd on-CPU input tensors. This feature improves performance in pipeline models by eliminating data transfer between device and CPU. For example, you can use this feature with Llama 4 (which accepts image and text inputs) to keep the vision encoder outputs on-device for the context encoding model to process.

To use pipeline execution, specify ``pipeline_execution=True`` when you initialize a ModelWrapper. For more information, see :ref:`how-to-use-fpem`.

Other improvements
^^^^^^^^^^^^^^^^^^

* Added support for PyTorch 2.8 and Python 3.11.
* Added support for sequence parallelism in mixture-of-experts (MoE) routers. This change improves
  context encoding latency for MoE models that use sequence parallelism.
* Enabled ``temperature=0`` as a valid option in dynamic on-device sampling. This temperature
  value specifies to use greedy sampling.
* Enabled ``top_k`` values of ``0`` and ``-1`` as valid options in dynamic on-device sampling.
  These ``top_k`` values specify to randomly pick a token from the vocabulary using a uniform
  distribution.

Bug fixes
---------

* Fixed an issue where HuggingFaceGenerationAdapter performs redundant CPU sampling for models that
  use on-device sampling and ``output_logits=True``. This fix improves the performance of models with
  this configuration.
* Other minor fixes and improvements.

Known issues
------------

* ``spmd_mode = True`` does not work when provided to the ``parallel_model_trace`` API. ``parallel_model_trace`` will be deprecated in the next Neuron SDK release.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-nxd-inference`
* :ref:`neuronx-distributed-inference-rn`
