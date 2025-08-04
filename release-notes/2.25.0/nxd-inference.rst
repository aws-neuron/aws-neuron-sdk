.. _neuron-2-25-0-nxd-inference:

.. meta::
   :description: The official release notes for the AWS Neuron SDK NxD Inference component, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0: NxD Inference release notes
==================================================

**Date of release**: July 31, 2025

**Version**: 0.5.9230

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.25.0 release notes home <neuron-2-25-0-whatsnew>`

Improvements
------------

*Improvements are significant new or improved features and solutions introduced this release of the AWS Neuron SDK. Read on to learn about them!*

Qwen3 (dense) model support
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add support for Qwen3 dense models, which are tested on Trn1. Compatible models include:

- `Qwen3-0.6B <https://huggingface.co/Qwen/Qwen3-0.6B>`__
- `Qwen3-1.7B <https://huggingface.co/Qwen/Qwen3-1.7B>`__
- `Qwen3-4B <https://huggingface.co/Qwen/Qwen3-4B>`__
- `Qwen3-8B <https://huggingface.co/Qwen/Qwen3-8B>`__
- `Qwen3-14B <https://huggingface.co/Qwen/Qwen3-14B>`__
- `Qwen3-32B <https://huggingface.co/Qwen/Qwen3-32B>`__

For more information, see :ref:`nxdi-model-reference`.

Other improvements
^^^^^^^^^^^^^^^^^^

- Added simplified functions that you can use to validate the accuracy of
  logits returned by a model. These new functions include
  ``check_accuracy_logits_v2`` and ``generated_expected_logits``, which provide more flexibility
  than ``check_accuracy_logits``. For more information, see :ref:`nxdi-evaluating-models`.
- Added ``scratchpad_page_size`` attribute to NeuronConfig. You can
  specify this attribute to configure the scratchpad page size used
  during compilation and at runtime. The scratchpad is a shared memory buffer
  used for internal model variables and other data. For more information, see :ref:`nxd-inference-api-guide-neuron-config`.
- Enabled `Chunked Attention <https://huggingface.co/blog/llama4-release#:~:text=Chunked%20attention%20(in%20RoPE%20layers)>`__ as a generic building block for
  any attention-based model. Chunked attention limits the KV cache size to chunk size and can be used to enable long-context inference where memory constraint is an issue. 
  NxDI now supports chunked attention for any model that defines ``attention_chunk_size`` in the model's HuggingFace ``config.json``,  such as `Llama 4 Scout <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E/blob/main/config.json#L11>`__,
  or in the model's InferenceConfig.
  Developers using NxDI can then pass ``attention_chunk_size`` to the attention module to enable chunked attention. See `modeling_llama.py <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/llama/modeling_llama.py>`__ for example.
- Published scripts to evaluate model accuracy and benchmark performance against Neuron. For more details, see :ref:`the corresponding documentation <libraries/nxd-inference/tutorials/generating-results-with-performance-cli>` or `go to the Neuron samples GitHub repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/inference-benchmarking>`_.
  
Breaking changes
----------------

*Sometimes we have to break something now to make the experience better in the longer term. Breaking changes are changes that may require you to update your own code, tools, and configurations.*

- Removed support for Meta checkpoint compatibility in Llama3.2 Multimodal modeling
  code. You can continue to use Hugging Face checkpoints. Hugging Face
  provides a `conversion
  script <https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/convert_mllama_weights_to_hf.py>`__
  that you can run to convert a Meta checkpoint to a Hugging Face checkpoint.

Bug fixes
---------

*We're always fixing bugs. It's developer's life!* Here's what we fixed in 2.25.0:

- Fixed accuracy issues when using Automatic Prefix Caching (APC) with
  EAGLE speculation.
- Fixed continuous batching for Llama3.2 Multimodal where the input batch size is less
  than the compiled batch size.
- Added support for continuous batching when running Neuron modeling code
  on CPU.
- Set a manual seed in ``benchmark_sampling`` to improve the stability
  of data-dependent benchmarks like speculation.
- Other minor fixes and improvements.
