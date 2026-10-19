.. meta::
    :description: Release notes for vLLM Neuron — high-performance LLM inference serving on AWS Trainium.
    :keywords: vllm neuron, vllm, inference, serving, release notes, aws neuron sdk
    :date-modified: 10/19/2026

.. _vllm-neuron_rn:

vLLM Neuron Release Notes
==========================

**Latest version (in 2.33.0)**: 0.29.0.1.2.0 (Beta)

vLLM Neuron is an open-source vLLM plugin for high-performance LLM inference
serving on AWS Trainium. For an introduction, see
:doc:`/vllm-neuron/docs/index`.

For the complete release notes for each version, see the `vLLM Neuron releases on GitHub <https://github.com/vllm-project/vllm-neuron/releases>`_.

.. _vllm-neuron-2-33-0-rn:

vLLM Neuron [0.29.0.1.2.0] (Neuron 2.33.0 Release)
--------------------------------------------------

vLLM Neuron 0.29.0.1.2.0 (Beta) moves to native PyTorch ``torch.compile`` as
the default compilation path and improves inference performance: GPT-OSS-120B
serves up to 2.5x more throughput than Neuron 2.32, and Qwen3-VL-32B up to 38%
more, with faster server startup. Model Runner V2 is now the default
for supported dense models. It adds Qwen3-32B dense model support and
Qwen3-ASR-1.7B for speech-to-text. This version ships with vLLM 0.29.0, and adds PyTorch 2.13,
Python 3.14, and Ubuntu 26.04 support.

For the full list of changes, see the `vLLM Neuron releases on GitHub <https://github.com/vllm-project/vllm-neuron/releases>`_.

.. _vllm-neuron-2-32-0-rn:

vLLM Neuron [0.24.0.1.1.0] (Neuron 2.32.0 Release)
--------------------------------------------------

Neuron 2.32.0 is the first Neuron SDK release to include vLLM Neuron (Beta), shipping 0.24.0.1.1.0 with
vLLM 0.24.0. For the release notes, see the
`0.24.0.1.1.0 release on GitHub <https://github.com/vllm-project/vllm-neuron/releases/tag/v0.24.0.1.1.0>`_.

.. _vllm-neuron-0-21-0-1-0-0-rn:

vLLM Neuron [0.21.0.1.0.0] (Beta, July 20, 2026)
------------------------------------------------

vLLM Neuron Beta (0.21.0.1.0.0) was released standalone on July 20, 2026, for Neuron SDK 2.31.0 and vLLM 0.21.0,
as an enhanced vLLM plugin for serving models on Trn2 and Trn3 that implements models directly within the
plugin, with no dependency on NxD Inference. Starting with this release, vLLM Neuron versions follow the
pattern ``<vLLM version>.<plugin version>`` (0.21.0.1.0.0 = vLLM 0.21.0, plugin 1.0.0); earlier releases
(0.1.0 through 0.5.3) were the NxD Inference-based plugin.

See :ref:`vLLM Neuron Beta now available <whats-new-2026-07-20-vllm-neuron-beta>` and the
`0.21.0.1.0.0 release on GitHub <https://github.com/vllm-project/vllm-neuron/releases/tag/v0.21.0.1.0.0>`_.
