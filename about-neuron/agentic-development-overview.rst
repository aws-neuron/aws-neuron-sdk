.. meta::
   :description: Overview of Neuron Agentic Development, an open-source suite of AI coding agents and skills that author, debug, profile, and analyze NKI kernels for AWS Trainium and Inferentia using Claude Code, Kiro, and other agentic IDEs.
   :date_updated: 2026-04-28
   :keywords: Neuron Agentic Development, agentic development, AI agents, AI skills, NKI, NKI kernels, Kiro, Claude Code, Trainium, Inferentia, agentic IDE

.. _neuron-agentic-development-overview:

===================================
Neuron Agentic Development Overview
===================================

Neuron Agentic Development is an open-source suite of AI capabilities starting with agents and skills that author, debug, profile, and analyze Neuron Kernel Interface (NKI) kernels for AWS Trainium and AWS Inferentia. The agents and skills run inside agentic coding environments such as Claude Code and Kiro, letting you drive kernel development with natural language or reference implementations in PyTorch or NumPy.

The goal is to meet you in your existing agentic IDE workflow: ask for a kernel in plain language or paste a PyTorch implementation, and the agent coordinates translation to NKI, on-device compilation, profiling, and targeted analysis without requiring you to learn Neuron-specific tooling one surface at a time.

.. note::
   Want to dive into working with agents on Neuron? Check out the full set of agents, skills, and samples in the open-source repo: `aws-neuron/neuron-agentic-development <https://github.com/aws-neuron/neuron-agentic-development>`_.

   Get `Amazon Kiro <https://kiro.dev/docs/getting-started/installation/>`__ and try it out!

Applies to
----------

Neuron Agentic Development applies to:

- Authoring new NKI kernels from PyTorch, NumPy, or natural-language specifications.
- Debugging NKI kernel compilation errors.
- Capturing and querying on-device execution profiles.
- Identifying a defined set of kernel inefficiencies such as intermediate-data spilling and redundant TensorEngine transposes.
- Looking up NKI documentation, API references, tutorials, and error codes from within an agent conversation.
- Common kernel-development use cases, including fused attention, custom normalization operations, and quantized matrix multiplications — any operation where the default compiler output leaves performance on the table.

What problem does it solve?
---------------------------

NKI is the programming interface for writing custom compute kernels that run directly on Trainium and Inferentia NeuronCores. It gives you fine-grained control over the hardware's tensor engines, vector engines, and DMA subsystem, so you can reach performance beyond what framework-level compilation alone can achieve.

Neuron Agentic Development makes it simpler to develop with Neuron with agents and skills.

How it works
------------

An agent runs inside your agentic IDE (for example, Claude Code or `Kiro <https://kiro.dev/docs/getting-started/installation/>`__ ) on a Trainium or Inferentia instance. When you ask for a kernel or point it at one, the agent calls one or more skills to complete the task:

- **Authoring skills** generate NKI kernel code and modify existing kernels.
- **Debugging skills** compile the kernel on-device, interpret compiler errors, and iteratively apply fixes.
- **Profiling skills** run the kernel, capture NEFF and NTFF execution profiles, and extract structured metrics.
- **Profile-query skills** let you ask questions about engine utilization and execution behavior against the captured profile.
- **Analysis skills** detect a curated set of kernel inefficiencies and attribute them to known patterns.
- **Documentation skills** route questions to the right NKI documentation, API references, tutorials, or compiler error-code pages.
- **Agent-ready code samples** provide important code context to help your AI tools produce cleaner, more optimized code for Neuron.

Skills chain automatically. A single natural-language prompt such as "profile my kernel and tell me what's slow" can trigger the full capture, metric-extraction, and analysis sequence. You can also call individual skills directly for targeted tasks.

The skills
~~~~~~~~~~

The package provides specialized skills that follow the natural kernel-development pipeline: write → debug → profile → analyze. Skills can be invoked individually for targeted tasks, or chained together by the top-level agent, which auto-selects the right workflow based on your request.

- ``neuron-nki-writing`` — the starting point for creating NKI kernels. Translates PyTorch, NumPy, or natural-language descriptions into correct NKI code. Covers tiling strategies that respect hardware constraints, memory access patterns, compute operations with explicit ``dst`` parameters, and efficiency guidelines for DMA sizing and SBUF reuse. The skill classifies your task by complexity and loads only the references it needs.
- ``neuron-nki-debugging`` — a systematic workflow for resolving NKI compilation and execution errors on Trainium and Inferentia hardware. Sets up the environment with the correct ``--target`` flags, resolves compiler errors using a categorized index of all 28 NCC error codes, and validates numerical correctness against CPU-computed references.
- ``neuron-nki-profiling`` — captures execution profiles on hardware. Configures runtime inspection environment variables, runs the kernel, identifies the correct Neuron Executable File Format (NEFF), captures the trace with Neuron Explorer (including DGE notifications for DMA-level detail), and extracts JSON metrics. Produces the NEFF and Neuron Trace File Format (NTFF) files that the profile-querying skill consumes.
- ``neuron-nki-profile-querying`` — ingests NEFF and NTFF files and runs SQL queries to compute performance bounds, identify bottleneck engines, and localize inefficiencies to specific NKI source lines. Supports three analysis approaches: the Neuron Explorer API server, DuckDB directly on Parquet, or Pandas for custom computation.
- ``neuron-nki-docs`` — used across all stages of development. During authoring, it provides API signatures and tutorials. During debugging, it explains error codes. During profiling, it clarifies hardware-architecture details. Useful for any ``nisa.*`` or ``nl.*`` API lookup, error-code explanation, tutorial search, or architecture guide for Trainium 1, 2, and 3.

The agents
~~~~~~~~~~

Skills provide the building blocks for individual tasks. Agents combine multiple skills into autonomous workflows, each as a specialized persona that handles multi-step development scenarios end-to-end.

- ``neuron-nki-agent`` — the unified entry point for NKI development. Automatically selects the right workflow based on your request (writing, debugging, profiling, or documentation lookup) and orchestrates the appropriate skills. Start here by default.
- ``neuron-nki-writing-agent`` — focuses exclusively on kernel authoring. Translates PyTorch, NumPy, or natural-language descriptions into NKI code and handles modifications to existing kernels.
- ``neuron-nki-debugging-agent`` — autonomously analyzes compiler errors, searches documentation for fixes, and applies corrections. Tracks iterations (up to 10) and progressively simplifies when stuck.
- ``neuron-nki-docs-agent`` — a lightweight documentation navigator for API signatures, error-code explanations, tutorials, and architecture details.
- ``neuron-nki-profile-analysis`` — captures execution profiles on hardware and identifies inefficiencies.

Supported agentic IDEs
----------------------

The agents and skills are distributed as a portable package that integrates with:

- **Claude Code.** Install as a plugin from the Neuron Agentic Development repository.
- **Kiro.** Copy the agents, skills, and context into your Kiro configuration directory.

Writing and documentation skills work in any supported IDE without hardware. Debugging, profiling, and analysis skills require access to Trainium or Inferentia hardware because they compile and execute on the NeuronCores.

Hardware and environment requirements
-------------------------------------

To use the full set of capabilities, you need:

- A Trainium or Inferentia instance (for example, ``trn1``, ``trn2``, or ``inf2``). The Amazon Linux 2023 Neuron DLAMI is a convenient starting point.
- The Neuron SDK and Neuron tools installed on the instance (pre-installed on Neuron DLAMIs, or install ``aws-neuronx-tools`` manually).
- A Python virtual environment containing Neuron packages such as ``neuronx-cc``, ``torch-neuronx``, and Neuron Explorer.
- Claude Code or Kiro installed on the same instance as the hardware. The agent and the hardware are co-located — there is no separate laptop-to-host file-transfer step.
- An Anthropic API key or equivalent credentials for the Claude model used by the agent.

For writing and documentation tasks only, you can run the agent on any machine with Claude Code or Kiro; hardware is not required.

Get started
-----------

Source, installation steps, and the current catalog of agents and skills are maintained in the Neuron Agentic Development repository:

- `Neuron Agentic Development on GitHub README <https://github.com/aws-neuron/neuron-agentic-development/README.md>`_

Follow the linked README for the supported installation paths (for example, Claude Code plugin install, Kiro setup script) and for the up-to-date list of available agents and skills.

Limitations
-----------

- **Hardware required for on-device skills.** Debugging, profiling, and kernel-level analysis require Trainium or Inferentia access. Writing and documentation skills do not.
- **Claude model support.** Current agents target Claude models. Support for additional model providers will depend on demand and capability parity.
- **NKI kernels only.** The agents and skills target NKI kernels on Neuron hardware. They do not generate or optimize NVIDIA CUDA kernels. If you have a GPU PyTorch model, you can use the authoring skill to translate PyTorch operations into NKI kernels as part of a Trainium migration.
- **Evolving capability set.** The agent and skill catalog is actively evolving. Treat the GitHub repository as the authoritative source for what is available today.

Responsible use
---------------

Agent-generated NKI code should be validated before you ship it. The debugging and analysis skills check compilation and surface known inefficiency patterns, but they do not replace numerical-correctness testing against a reference implementation on your target instance type, Neuron SDK version, and data types. When prompting the agent, include the target instance type, Neuron SDK version, and framework version so the generated code matches your environment.

Related concepts
----------------

- :ref:`What is AWS Neuron? <what-is-neuron>` — Overview of the Neuron SDK and its components.
- :ref:`amazon-q-dev` — Using AWS AI helper tools (Kiro, Amazon Q, Quick) with Neuron.

Further reading
---------------

- :doc:`/nki/index` — Neuron Kernel Interface documentation.
- :doc:`/tools/neuron-explorer/index` — Neuron Explorer, the profiling and analysis tool that agents invoke for on-device profile capture.
- `Kiro documentation <https://kiro.dev/docs/>`__ -- Learn about Kiro, AWS's agentic IDE.
