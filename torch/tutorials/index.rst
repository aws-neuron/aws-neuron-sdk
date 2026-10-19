.. meta::
   :description: End-to-end tutorials for TorchNeuron on AWS Trainium. Covers simple training, distributed training with torch.distributed, and Hugging Face inference with torch.compile.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium, tutorial, training, torch.distributed, FSDP, DTensor, DDP, mixed mode
   :date-modified: 2026-10-15

.. _torchneuron-tutorials:

==========================
TorchNeuron tutorials
==========================

End-to-end walk-throughs on AWS Trainium that build on the quickstarts. Each
tutorial produces a working, realistic use case.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card::
      :link: /torch/tutorials/tutorial-simple-training
      :link-type: doc
      :class-card: sd-border-2

      **Simple training loop on Neuron**
      :bdg-primary:`tutorial`
      ^^^
      A complete, copy-paste training loop in eager mode and optional
      ``torch.compile``.

   .. grid-item-card::
      :link: /torch/tutorials/tutorial-torch-distributed
      :link-type: doc
      :class-card: sd-border-2

      **Distributed training with torch.distributed**
      :bdg-primary:`tutorial`
      ^^^
      Scale a training job across multiple NeuronCores and nodes using FSDP,
      DTensor, and DDP.

   .. grid-item-card::
      :link: /torch/tutorials/tutorial-hf-inference-torch-compile
      :link-type: doc
      :class-card: sd-border-2

      **Hugging Face inference with torch.compile**
      :bdg-primary:`tutorial`
      ^^^
      Compile Hugging Face ``generate()`` with ``torch.compile`` and the
      ``neuron`` backend, and compile a model that is sharded across
      NeuronCores.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple training loop on Neuron <tutorial-simple-training>
   Distributed training with torch.distributed <tutorial-torch-distributed>
   Hugging Face inference with torch.compile <tutorial-hf-inference-torch-compile>
