.. meta::
    :description: Release notes for TorchTitan Neuron — large-model pre-training on AWS Trainium with TorchNeuron.
    :keywords: torchtitan neuron, torchtitan, training, pre-training, release notes, aws neuron sdk
    :date-modified: 10/19/2026

.. _torchtitan-neuron_rn:

TorchTitan Neuron Release Notes
===============================

**Latest version (in 2.33.0)**: 0.2.2+neuron.1 (Beta)

TorchTitan Neuron is an extension of `pytorch/torchtitan
<https://github.com/pytorch/torchtitan>`_ that adds AWS Trainium support with TorchNeuron. For an introduction and the
supported models, see :doc:`/training/torchtitan/about-torchtitan-neuron`.

.. _torchtitan-neuron-2-33-0-rn:

TorchTitan Neuron [0.2.2+neuron.1] (Neuron 2.33.0 Release)
----------------------------------------------------------

First release of TorchTitan Neuron (Beta). It adds Trainium support to
pytorch/torchtitan with TorchNeuron.

- **Supported models:** Llama3, Qwen3, DeepSeek-V3, and GPT-OSS.
- **Distributed strategies:** FSDP2 (``fully_shard``) and SimpleFSDP, tensor
  parallel (TP), context parallel (CP), expert parallel (EP) for MoE models,
  and pipeline parallel (PP).
- **Execution:** eager mode and ``torch.compile`` with the Neuron backend
  (``--compile.backend neuron``).
- **Hardware:** Trn2 and Trn3 instances.

Supported PyTorch and Python versions
--------------------------------------

TorchTitan Neuron runs on ``torch-neuronx`` 2.14. For the full TorchNeuron
support matrix, see :doc:`/release-notes/components/torchneuron`.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - ``torch-neuronx`` package family
     - PyTorch version
     - Python versions
   * - ``2.14.3.0.*``
     - 2.14.0
     - 3.12, 3.13, and 3.14
