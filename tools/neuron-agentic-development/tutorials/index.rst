.. meta::
   :description: Tutorials for Neuron Agentic Development. Step by step guides for using agents and skills on real tasks.
   :keywords: Neuron Agentic Development, tutorials, autoport, equivalence, NKI, model porting, model validation
   :date-modified: 2026-05-11

.. _neuron-agentic-dev-tutorials:

=========
Tutorials
=========

Step by step guides for using Neuron Agentic Development agents and skills on real tasks.

.. toctree::
   :maxdepth: 1
   :hidden:

   Port a model with Autoport </tools/neuron-agentic-development/tutorials/autoport-tutorial>
   Validate a port with Equivalence </tools/neuron-agentic-development/tutorials/equivalence-tutorial>

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Tutorial: Port a HuggingFace model using the Autoport skill
      :link: /tools/neuron-agentic-development/tutorials/autoport-tutorial
      :link-type: doc

      Walk through the full process of porting a HuggingFace model to NxD Inference
      using the Autoport agent. Covers parameter setup, invocation, and what to expect
      at each stage.

   .. grid-item-card:: Tutorial: Validate a model port using the Equivalence skill
      :link: /tools/neuron-agentic-development/tutorials/equivalence-tutorial
      :link-type: doc

      Walk through validating a ported NxD Inference model against its HuggingFace
      reference using the Equivalence agent. Covers R-ratio testing, fault localization,
      and end-to-end accuracy verification.
