.. meta::
   :description: Developer guides for Neuron Agentic Development skills. Deep dives on model porting, equivalence validation, and NKI kernel workflows.
   :keywords: Neuron Agentic Development, developer guides, autoport, equivalence, deep dive
   :date-modified: 2026-08-13

.. _neuron-agentic-dev-guides:

================
Developer Guides
================

These guides explain how Neuron Agentic Development skills work internally. Read them
to understand the architecture, decision logic, and validation strategies that the
agents use under the hood.

.. toctree::
   :maxdepth: 1
   :hidden:

   Autoport NxDI Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-framework-autoport>
   Autoport vLLM Neuron Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-framework-autoport-vllm-neuron>
   Equivalence Validation Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-framework-equivalence>

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Deep dive: Port HuggingFace models to Neuron with the Autoport skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-framework-autoport
      :link-type: doc

      How the Autoport skill analyzes model architectures, generates NxD Inference
      implementations, compiles to NEFF, and validates accuracy against the original model.

   .. grid-item-card:: Deep dive: Port GPU-compatible models to the vLLM Neuron backend with the Autoport skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-framework-autoport-vllm-neuron
      :link-type: doc

      How the Autoport vLLM Neuron skill researches model architectures, generates
      vLLM Neuron implementations for the Trainium2 backend, registers them, and
      validates logits and equivalence against the GPU-compatible reference implementation.

   .. grid-item-card:: Deep dive: Validate model ports with the Equivalence skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-framework-equivalence
      :link-type: doc

      How the Equivalence skill compares numerical output between a reference model and its
      port. Covers both NxD Inference and vLLM Neuron targets.
