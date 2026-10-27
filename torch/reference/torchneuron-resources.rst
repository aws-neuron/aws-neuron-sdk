.. meta::
   :description: Links to the TorchNeuron source repository, runnable examples, profiling tools, PyTorch documentation, and support channels for native PyTorch on AWS Trainium.
   :keywords: TorchNeuron, resources, open source, GitHub, PyTorch, AWS Neuron, Trainium, examples, TorchTitan
   :date-modified: 2026-10-13

.. _torchneuron-resources:

TorchNeuron resources
=====================

Links to the source code, examples, profiling tools, upstream PyTorch documentation, and support
channels for TorchNeuron (``torch-neuronx``).

Source code
-----------

`aws-neuron/torch-neuronx <https://github.com/aws-neuron/torch-neuronx>`_ contains the TorchNeuron
native PyTorch backend: the ``PrivateUse1`` device backend, ATen operator implementations, and the
TorchDynamo backend for ``torch.compile``.

Examples
--------

The `examples/ <https://github.com/aws-neuron/torch-neuronx/tree/main/examples>`_ directory
contains ready-to-run scripts, each with its own README. Start with the
`torch.compile quickstart <https://github.com/aws-neuron/torch-neuronx/blob/main/examples/compile/QUICKSTART.md>`_,
the `GPT-2 training loop <https://github.com/aws-neuron/torch-neuronx/tree/main/examples/gpt2-train-loop>`_
in eager mode, `fused attention <https://github.com/aws-neuron/torch-neuronx/tree/main/examples/fused_attention>`_
with NKI kernels, or `tensor parallelism with FSDP <https://github.com/aws-neuron/torch-neuronx/tree/main/examples/simple_mlp_tpxfsdp>`_.

Profile with Neuron Explorer
----------------------------

- :doc:`/tools/neuron-explorer/index`: what Neuron Explorer shows about a TorchNeuron workload.
- :doc:`/tools/neuron-explorer/how-to-profile-pytorch`: capture a profile from a PyTorch workload,
  including ``torch.compile`` versus eager mode.

Libraries built on TorchNeuron
------------------------------

- :doc:`vLLM Neuron </vllm-neuron/docs/index>` and :doc:`vLLM Omni Neuron </vllm-omni-neuron/docs/index>` for inference serving.
- :doc:`TorchTitan </training/torchtitan/index>` for large-scale pre-training.
- Hugging Face `transformers <https://huggingface.co/docs/transformers/index>`_,
  `diffusers <https://huggingface.co/docs/diffusers/index>`_, `PEFT <https://huggingface.co/docs/peft/index>`_,
  and `accelerate <https://huggingface.co/docs/accelerate/index>`_ run natively on TorchNeuron; no Neuron-specific
  compatibility layer is needed.

PyTorch community resources
---------------------------

- `torch.compile overview <https://docs.pytorch.org/docs/stable/torch.compiler.html>`_:
  ``torch.compile`` and TorchDynamo.
- `Custom backends for torch.compile <https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html>`_:
  how a backend such as ``backend="neuron"`` plugs into ``torch.compile``.
- `PrivateUse1 backend tutorial <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>`_:
  the out-of-tree device mechanism TorchNeuron uses.

Get help
--------

- :doc:`/torch/troubleshooting`: common issues and what to collect before reporting a problem.
- `aws-neuron/aws-neuron-sdk issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_:
  report bugs and request features.
- `AWS re:Post, aws-neuron tag <https://repost.aws/tags/TAjy-krivRTDqDPWNNBmV9lA/aws-neuron>`_:
  ask questions and search community answers.

Migrating from XLA-based torch-neuronx
--------------------------------------

Earlier ``torch-neuronx`` releases were built on PyTorch/XLA and are a separate product lineage
from TorchNeuron. The XLA-based documentation is in the
:doc:`archive </archive/frameworks/torch/index>`.
