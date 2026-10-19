.. meta::
   :description: Reference materials for TorchNeuron on AWS Trainium. Includes torch.compile configuration, supported operations, and links to resources.
   :keywords: TorchNeuron, reference, torch.compile, configuration, supported operations, resources, PyTorch, Trainium, AWS Neuron
   :date-modified: 2026-09-01

.. _torch-reference:

TorchNeuron reference
======================

Reference materials for TorchNeuron configuration, supported operations, and external resources.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/reference/api/index
      :link-type: doc
      :class-card: sd-border-2

      **API reference**
      :bdg-info:`reference`
      ^^^
      Compatibility guide and supported operators for ``torch-neuronx`` on Neuron.

   .. grid-item-card::
      :link: /torch/reference/torchneuron-resources
      :link-type: doc
      :class-card: sd-border-2

      **TorchNeuron resources**
      :bdg-info:`reference`
      ^^^
      Links to Neuron native PyTorch open-source repositories, documentation, and relevant
      PyTorch community resources.

   .. grid-item-card::
      :link: /torch/reference/best-practices-pytorch-neuron
      :link-type: doc
      :class-card: sd-border-2

      **Best practices for TorchNeuron**
      :bdg-info:`best practice`
      ^^^
      Recommended methods for optimizing eager mode, torch.compile, distributed training, and
      mixed precision on Trainium.

   .. grid-item-card::
      :link: /torch/reference/env-vars
      :link-type: doc
      :class-card: sd-border-2

      **TorchNeuron environment variables**
      :bdg-info:`reference`
      ^^^
      Variables that configure ``torch-neuronx`` behavior: the ``neuron``
      backend for ``torch.compile``, caching, eager execution, distributed
      collectives, and the memory allocator.

   .. grid-item-card::
      :link: /torch/reference/guidelines-memory-management
      :link-type: doc
      :class-card: sd-border-2

      **Guidelines for memory management**
      :bdg-info:`reference`
      ^^^
      NUMA CPU affinity, Neuron caching allocator tuning, 64-bit dtype
      behavior, and cache placement.

.. toctree::
   :maxdepth: 1
   :hidden:

   API reference </torch/reference/api/index>
   Environment variables </torch/reference/env-vars>
   Best practices </torch/reference/best-practices-pytorch-neuron>
   Guidelines for memory management </torch/reference/guidelines-memory-management>
   TorchNeuron resources </torch/reference/torchneuron-resources>
