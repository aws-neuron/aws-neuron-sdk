.. meta::
   :description: How-to guides for TorchNeuron on AWS Trainium. Covers distributed training, mixed precision, and other common tasks.
   :keywords: TorchNeuron, guides, how-to, distributed training, FSDP, PyTorch, Trainium, AWS Neuron
   :date-modified: 2026-09-01

.. _torch-guides:

TorchNeuron guides
===================

Task-oriented guides for common TorchNeuron workflows on AWS Trainium. Each guide walks you through
a specific task with step-by-step instructions.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/guides/how-to-work-with-nki
      :link-type: doc
      :class-card: sd-border-2

      **Work with NKI kernels from torch-neuronx**
      :bdg-primary:`how-to`
      ^^^
      Integrate NKI kernels into torch-neuronx workflows with ``@nki_op``, autograd, and
      ``torch.compile``.

   .. grid-item-card::
      :link: /torch/guides/torch-distributed
      :link-type: doc
      :class-card: sd-border-2

      **Use torch.distributed on Neuron**
      :bdg-primary:`how-to`
      ^^^
      Configure FSDP, DTensor, DDP, and functional collectives on Neuron.

   .. grid-item-card::
      :link: /torch/guides/memory-profiling-guide
      :link-type: doc
      :class-card: sd-border-2

      **Profile memory usage in torch-neuronx**
      :bdg-primary:`how-to`
      ^^^
      Use ``torch_neuronx.memory_*`` APIs and Neuron Explorer to measure and fix memory
      issues.

   .. grid-item-card::
      :link: /torch/compile/index
      :link-type: doc
      :class-card: sd-border-2

      **torch.compile on Neuron**
      :bdg-primary:`how-to`
      ^^^
      JIT-compile your model with ``backend="neuron"``. Covers the compilation
      pipeline, optimization, compiling without hardware, and the backend reference.

.. TODO: Add cards for additional guides as they are created:
   - How to use mixed precision training

.. toctree::
   :maxdepth: 1
   :hidden:

   Work with NKI kernels from torch-neuronx </torch/guides/how-to-work-with-nki>
   Use torch.distributed on Neuron </torch/guides/torch-distributed>
   Profile memory usage in torch-neuronx </torch/guides/memory-profiling-guide>
   torch.compile on Neuron </torch/compile/index>
