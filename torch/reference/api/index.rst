.. meta::
   :description: TorchNeuron API reference. Operator compatibility table and torch.compile reference for torch-neuronx on AWS Neuron.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium, API, reference, compatibility
   :date-modified: 2026-05-12

.. _torchneuron-api-reference:

============================
TorchNeuron API reference
============================

API reference material for ``torch-neuronx`` on AWS Neuron.

.. grid:: 1
   :gutter: 2

   .. grid-item-card::
      :link: https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md
      :link-type: url
      :class-card: sd-border-2

      **Operator compatibility table**
      ^^^
      Per-operator support status, dtype coverage, and CPU fallbacks for
      ``torch-neuronx``, maintained in the ``torch-neuronx`` repository.

   .. grid-item-card::
      :link: /torch/compile/reference-torch-compile
      :link-type: doc
      :class-card: sd-border-2

      **torch.compile backend reference**
      ^^^
      ``torch.compile`` parameter and ``torch.compiler`` API support, graph
      breaks, environment variables, and caching.


.. toctree::
   :maxdepth: 1
   :hidden:

   Operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>
