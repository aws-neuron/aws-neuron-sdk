.. meta::
   :description: NeuronX Compiler (neuronx-cc) documentation for Trn1, Trn1n, Inf2, and Trn2 — overview, CLI reference, developer guides, how-tos, and FAQ.
   :date-modified: 2026-07-30

.. _neuronx-cc-index:

================================
NeuronX Compiler for Trn* & Inf2
================================

The NeuronX Compiler (``neuronx-cc``) is the XLA-based Neuron Graph Compiler for
NeuronCores v2 to v4 (Trn1, Trn1n, Inf2, and Trn2). Use the guides below to learn what it
does, look up its command line, and work through common compilation tasks.

Get started
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: About neuronx-cc
      :link: /compiler/about-neuronx-cc
      :link-type: doc

      What the Neuron Graph Compiler is, how it fits into the Neuron workflow, and
      the files and features that make it up.

   .. grid-item-card:: CLI reference guide
      :link: /compiler/neuronx-cc/api-reference-guide/index
      :link-type: doc

      The complete ``neuronx-cc`` command line: the ``compile`` and
      ``list-operators`` commands, every option, and their valid values.

How-to guides
-------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: How to generate a NEFF file
      :link: /compiler/neuronx-cc/how-to-generate-neff
      :link-type: doc

      Compile an XLA HLO graph into a NEFF file with ``neuronx-cc``, the artifact the
      Neuron Runtime loads to run your model.

   .. grid-item-card:: How to use convolution kernels in UNet models
      :link: /compiler/neuronx-cc/how-to-convolution-in-unet
      :link-type: doc

      Modify UNet training models to use custom convolution kernels with NKI, to
      avoid out-of-memory errors on convolution-heavy models.

Guides and reference
--------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: FAQ
      :link: /compiler/neuronx-cc/faq
      :link-type: doc

      Common questions about compiling to NEFF, ``neuron-cc`` vs. ``neuronx-cc``,
      supported operators, precision, and recompilation.

.. toctree::
    :maxdepth: 1
    :hidden:

    API Reference Guide </compiler/neuronx-cc/api-reference-guide/index>
    How-to: Generate a NEFF file </compiler/neuronx-cc/how-to-generate-neff>
    How-to: Convolution </compiler/neuronx-cc/how-to-convolution-in-unet>
    FAQ </compiler/neuronx-cc/faq>
