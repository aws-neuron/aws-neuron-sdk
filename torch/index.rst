.. meta::
   :description: TorchNeuron is the native PyTorch backend for AWS Trainium. It supports eager mode execution, torch.compile, and standard distributed APIs for high-performance deep learning on Neuron hardware.
   :keywords: TorchNeuron, PyTorch, native PyTorch, torch.compile, eager mode, AWS Neuron, Trainium, distributed training, FSDP, NKI
   :date-modified: 2026-09-01

.. _torch-neuron-home:

TorchNeuron: Native PyTorch on AWS Neuron
==========================================

.. admonition:: Beta Release
   :class: note

   TorchNeuron is in public beta for the Neuron 2.33 release. It is supported on Trainium 2 (trn2)
   and Trainium 3 (trn3) instances. We welcome your feedback.

TorchNeuron is an open-source native PyTorch backend for AWS Trainium. It integrates through
PyTorch's standard `PrivateUse1 <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>`_ device backend mechanism, providing eager mode execution, ``torch.compile``
for JIT compilation, and standard distributed APIs including FSDP, DTensor, DDP, and Tensor Parallelism.

TorchNeuron uses standard PyTorch APIs. To run existing PyTorch code on Trainium, change the device
placement from ``cuda`` to ``neuron``.

What changes are needed?
-------------------------

**For eager mode:**

* Device placement: Change ``.to('cuda')`` to ``.to('neuron')``
* If your code uses ``torch.accelerator``, no changes are needed (automatic device detection)
* Mixed precision: Use ``torch.autocast(device_type="neuron")``
* Distributed training: ``torch.distributed`` works natively through the ``neuron`` backend and
  supports the PyTorch parallelism APIs: DDP, FSDP2, tensor and sequence parallelism with DTensor,
  pipeline parallelism, and context parallelism. Configure sharding based on NeuronCores rather
  than GPUs

**For torch.compile** (on top of the eager mode changes):

* Specify ``backend="neuron"`` (e.g., ``@torch.compile(backend="neuron")``)
* Remove CUDA-specific parameters like ``mode="max-autotune-no-cudagraphs"``

No ``import torch_neuronx`` is needed: PyTorch autoloads TorchNeuron on ``import torch`` through
the ``torch.backends`` entry point that the ``torch-neuronx`` package registers. This includes
``@nki.jit`` kernels, which run without the import. Import ``torch_neuronx`` only to call its own
APIs, such as ``nki_op`` or ``wrap_nki`` (see :doc:`/torch/guides/how-to-work-with-nki`).

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/about-pytorch-neuron
      :link-type: doc
      :class-card: sd-border-2

      **About TorchNeuron**
      :bdg-info:`overview`
      ^^^
      Understand TorchNeuron's architecture, execution modes (eager and ``torch.compile``), and
      how it integrates with the PyTorch ecosystem.

   .. grid-item-card::
      :link: /torch/get-started/index
      :link-type: doc
      :class-card: sd-border-2

      **Get started**
      :bdg-primary:`quickstart`
      ^^^
      Quickstart guides to get your first model running on Trainium in eager mode and with torch.compile.

   .. grid-item-card::
      :link: /torch/tutorials/index
      :link-type: doc
      :class-card: sd-border-2

      **Tutorials**
      :bdg-primary:`tutorial`
      ^^^
      End-to-end walk-throughs for simple and distributed training, Hugging Face
      fine-tuning, and Hugging Face inference with ``torch.compile``.

   .. grid-item-card::
      :link: /torch/guides/index
      :link-type: doc
      :class-card: sd-border-2

      **Guides**
      :bdg-primary:`how-to`
      ^^^
      How-to guides for ``torch.distributed`` on Neuron, working with NKI kernels from
      ``torch-neuronx``, profiling memory usage, and ``torch.compile`` on Neuron.

   .. grid-item-card::
      :link: inference-native-pytorch
      :link-type: ref
      :class-card: sd-border-2

      **Native PyTorch inference**
      :bdg-info:`overview`
      ^^^
      Run inference with native PyTorch on Trainium, serve native PyTorch models with vLLM
      Neuron, and see which package each path requires.

   .. grid-item-card::
      :link: /training/torchtitan/index
      :link-type: doc
      :class-card: sd-border-2

      **TorchTitan Neuron**
      :bdg-primary:`tutorial`
      ^^^
      Run TorchTitan pre-training on Trainium with TorchNeuron, with tutorials for Llama,
      DeepSeek (SimpleFSDP), and GPT-OSS.

   .. grid-item-card::
      :link: /torch/reference/index
      :link-type: doc
      :class-card: sd-border-2

      **Reference**
      :bdg-info:`reference`
      ^^^
      API compatibility, environment variables, best practices, memory management guidelines,
      and links to resources.

   .. grid-item-card::
      :link: /torch/troubleshooting
      :link-type: doc
      :class-card: sd-border-2

      **Troubleshooting**
      :bdg-info:`troubleshooting`
      ^^^
      Diagnose and resolve common issues with installation, eager mode, torch.compile, and
      distributed training.

.. toctree::
   :maxdepth: 1
   :hidden:

   About TorchNeuron <about-pytorch-neuron>
   Get started <get-started/index>
   Tutorials <tutorials/index>
   Guides <guides/index>
   Reference <reference/index>
   Troubleshooting <troubleshooting>
