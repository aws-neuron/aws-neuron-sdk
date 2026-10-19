.. meta::
   :description: Tutorials for Neuron Agentic Development. Step by step guides for using agents and skills on real tasks.
   :keywords: Neuron Agentic Development, tutorials, autoport, equivalence, profiling, profile analysis, parallelism planner, hardware entitlement, NKI, model porting, model validation
   :date-modified: 2026-10-08

.. _neuron-agentic-dev-tutorials:

=========
Tutorials
=========

Step by step guides for using Neuron Agentic Development agents and skills on real tasks.

.. toctree::
   :maxdepth: 1
   :hidden:

   Port a model with Autoport </tools/neuron-agentic-development/tutorials/autoport-tutorial>
   Port a model to vLLM Neuron </tools/neuron-agentic-development/tutorials/autoport-vllm-neuron-tutorial>
   Validate a port with Equivalence </tools/neuron-agentic-development/tutorials/equivalence-tutorial>
   Capture profiles with Profiling </tools/neuron-agentic-development/tutorials/profiling-vllm-neuron-tutorial>
   Analyze a profile with Profile Analysis </tools/neuron-agentic-development/tutorials/profile-analysis-vllm-neuron-tutorial>
   Plan a serving config with the Parallelism Planner </tools/neuron-agentic-development/tutorials/parallelism-planner-tutorial>
   Find the binding operator with Hardware Entitlement </tools/neuron-agentic-development/tutorials/hardware-entitlement-tutorial>

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Tutorial: Port a HuggingFace model using the Autoport skill
      :link: /tools/neuron-agentic-development/tutorials/autoport-tutorial
      :link-type: doc

      Walk through the full process of porting a HuggingFace model to NxD Inference
      using the Autoport agent. Covers parameter setup, invocation, and what to expect
      at each stage.

   .. grid-item-card:: Tutorial: Port a GPU-compatible model to vLLM Neuron using the Autoport skill
      :link: /tools/neuron-agentic-development/tutorials/autoport-vllm-neuron-tutorial
      :link-type: doc

      Walk through porting a GPU-compatible model to the vLLM Neuron Trainium2 backend
      using the Autoport vLLM Neuron agent. Covers invocation, what happens at each
      phase, validation, and serving the result.

   .. grid-item-card:: Tutorial: Validate a model port using the Equivalence skill
      :link: /tools/neuron-agentic-development/tutorials/equivalence-tutorial
      :link-type: doc

      Walk through validating a ported model against its reference using the Equivalence
      agent. Covers R-ratio testing, fault localization, end-to-end accuracy verification,
      and validating vLLM Neuron ports.

   .. grid-item-card:: Tutorial: Capture prefill and decode profiles using the Profiling skill
      :link: /tools/neuron-agentic-development/tutorials/profiling-vllm-neuron-tutorial
      :link-type: doc

      Capture two device profiles of a vLLM Neuron model on Trainium, one containing only
      prefill and one containing only steady-state decode, and confirm from the manifest
      that each trace is real before handing it to the analysis skill.

   .. grid-item-card:: Tutorial: Analyze a vLLM Neuron profile using the Profile Analysis skill
      :link: /tools/neuron-agentic-development/tutorials/profile-analysis-vllm-neuron-tutorial
      :link-type: doc

      Work interactively with the analysis agent to find where the time goes: which graphs
      ran, a step's roofline floors, its binding engine, its collective share, and which
      model component is responsible, with five checks that catch a plausible but wrong
      answer.

   .. grid-item-card:: Tutorial: Plan a serving config using the Parallelism Planner skill
      :link: /tools/neuron-agentic-development/tutorials/parallelism-planner-tutorial
      :link-type: doc

      Walk through choosing a parallelism config for Granite-4.1-8B on trn2 using the
      Parallelism Planner agent. Covers writing a run-spec, oracle validation, reading the
      sweep, and inspecting a single operating point.

   .. grid-item-card:: Tutorial: Find the binding operator using the Hardware Entitlement skill
      :link: /tools/neuron-agentic-development/tutorials/hardware-entitlement-tutorial
      :link-type: doc

      Walk through computing per-operator floors for the planner's recommended config using
      the Hardware Entitlement agent. Covers decode and prefill floors, how the binding moves
      with batch and context, and actual-versus-floor HBM attribution.
