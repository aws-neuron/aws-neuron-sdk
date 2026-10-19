.. meta::
   :description: Developer guides for Neuron Agentic Development skills. Deep dives on model porting, equivalence validation, profiling, performance planning, and NKI kernel workflows.
   :keywords: Neuron Agentic Development, developer guides, autoport, equivalence, profiling, profile analysis, parallelism planner, hardware entitlement, deep dive
   :date-modified: 2026-10-08

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
   Profiling vLLM Neuron Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-framework-profiling-vllm-neuron>
   Profile Analysis vLLM Neuron Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-framework-profile-analysis-vllm-neuron>
   Parallelism Planner Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-parallelism-planner>
   Hardware Entitlement Deep Dive </tools/neuron-agentic-development/developer_guides/neuron-hardware-entitlement>

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

   .. grid-item-card:: Deep dive: Capture device and system profiles from a vLLM Neuron server with the Profiling skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-framework-profiling-vllm-neuron
      :link-type: doc

      How the Profiling skill isolates the prefill and decode phases into separate device
      traces, manages the server lifecycle so a capture does not orphan device-holding
      workers, and records a reproducible input and output contract for the analysis skill.

   .. grid-item-card:: Deep dive: Localize performance bottlenecks with the Profile Analysis skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-framework-profile-analysis-vllm-neuron
      :link-type: doc

      How the Profile Analysis skill reads a capture's system and device traces together:
      session workload timing, roofline bounds, a collective breakdown, and a per-component
      attribution, and which of those outputs are measured rather than approximate.

   .. grid-item-card:: Deep dive: Plan inference parallelism with the Parallelism Planner skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-parallelism-planner
      :link-type: doc

      How the Parallelism Planner skill turns a model config and hardware constants into a
      TP, CP, and DP recommendation using an analytic entitlement model, before you run on
      hardware.

   .. grid-item-card:: Deep dive: Find the binding operator with the Hardware Entitlement skill
      :link: /tools/neuron-agentic-development/developer_guides/neuron-hardware-entitlement
      :link-type: doc

      How the Hardware Entitlement skill computes per-operator compute and HBM floors for a
      chosen config, and attributes a deployed graph's resident HBM back to those floors.
