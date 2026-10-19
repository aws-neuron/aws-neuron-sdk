.. meta::
   :description: torch.compile on AWS Neuron with TorchNeuron. Overview, how-to guides, reference, and best practices for JIT-compiling PyTorch models on Trainium with backend="neuron".
   :keywords: TorchNeuron, torch.compile, backend neuron, TorchDynamo, NEFF, JIT compilation, AWS Neuron, Trainium
   :date-modified: 2026-08-21

.. _torch-compile-index:

=========================
torch.compile on Neuron
=========================

TorchNeuron provides native ``torch.compile`` support on AWS Trainium through a
custom TorchDynamo backend, selected with ``torch.compile(model, backend="neuron")``.
This section explains how compilation works on Neuron and how to use it to optimize
your PyTorch models.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/compile/about-torch-compile
      :link-type: doc
      :class-card: sd-border-2

      **How torch.compile works**
      :bdg-info:`deep dive`
      ^^^
      The compilation pipeline (TorchDynamo → AOT Autograd → StableHLO →
      ``neuronx-cc`` → NEFF), caching, graph breaks, and fallback behavior.

   .. grid-item-card::
      :link: /torch/compile/how-to-optimize-with-torch-compile
      :link-type: doc
      :class-card: sd-border-2

      **Optimize a model**
      :bdg-primary:`how-to`
      ^^^
      Apply ``backend="neuron"`` to a working eager model and tune for throughput.

   .. grid-item-card::
      :link: /torch/compile/reference-torch-compile
      :link-type: doc
      :class-card: sd-border-2

      **Backend reference**
      :bdg-info:`reference`
      ^^^
      Environment variables, the ``neuron_dynamo_backend`` helper API, and caching
      configuration.

.. toctree::
   :maxdepth: 1
   :hidden:

   How torch.compile works </torch/compile/about-torch-compile>
   Optimize a model </torch/compile/how-to-optimize-with-torch-compile>
   Backend reference </torch/compile/reference-torch-compile>
