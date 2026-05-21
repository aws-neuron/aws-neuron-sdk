.. meta::
   :description: Neuron Agentic Development is a package of AI agents and skills for developing on AWS Trainium, including NKI kernel development and model porting to NxD Inference.
   :keywords: Neuron Agentic Development, AI agents, skills, NKI, model porting, autoport, Claude Code, Kiro, Trainium
   :date-modified: 2026-05-11

.. _neuron-agentic-development:
.. _neuron-agentic-development-overview:

==============================
Neuron Agentic Development
==============================

Neuron Agentic Development is an open source package of AI agents and skills for developing
on AWS Trainium. The agents and skills run inside agentic coding environments
like Claude Code and Kiro. You drive development with natural language, and the agent
coordinates the technical workflow for you.

The package covers two areas today.

1. **NKI kernel development.** Write, debug, profile, and analyze Neuron Kernel Interface
   kernels. Translate from PyTorch, NumPy, or plain language descriptions into NKI code
   that runs on NeuronCores.

2. **Model porting to NxD Inference.** Port HuggingFace transformer models to run on
   Trainium through NxD Inference. The agent handles architecture analysis,
   implementation, compilation, inference testing, and accuracy validation.

.. note::
   Source code, installation, and the full catalog of agents and skills live in the
   `Neuron Agentic Development GitHub repository <https://github.com/aws-neuron/neuron-agentic-development>`_.

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started </tools/neuron-agentic-development/getting-started>
   Developer Guides </tools/neuron-agentic-development/developer_guides/index>
   Tutorials </tools/neuron-agentic-development/tutorials/index>

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: /tools/neuron-agentic-development/getting-started
      :link-type: doc

      Install the package, set up your environment, and run your first agent.

   .. grid-item-card:: Developer Guides
      :link: /tools/neuron-agentic-development/developer_guides/index
      :link-type: doc

      Deep dives on how skills work internally. Architecture analysis, porting workflows,
      compilation pipelines, and validation strategies.

   .. grid-item-card:: Tutorials
      :link: /tools/neuron-agentic-development/tutorials/index
      :link-type: doc

      Step by step walkthroughs for using agents and skills on real tasks.


Skills
------

The package provides specialized skills that follow natural development pipelines.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Skill
     - Category
     - What it does
   * - ``neuron-nki-writing``
     - Authoring
     - Writes new NKI kernels or modifies existing ones from PyTorch, NumPy, or natural language.
   * - ``neuron-nki-debugging``
     - Debugging
     - Resolves NKI compilation errors using a categorized index of all 28 NCC error codes.
   * - ``neuron-nki-profiling``
     - Profiling
     - Captures execution traces on hardware and extracts JSON metrics.
   * - ``neuron-nki-profile-querying``
     - Analysis
     - Runs SQL queries against profile data to compute performance bounds and identify bottlenecks.
   * - ``neuron-nki-docs``
     - Documentation
     - Looks up API signatures, tutorials, error codes, and architecture details.
   * - ``neuron-framework-autoport``
     - Model porting
     - Ports HuggingFace models to NxD Inference with full compilation and validation.
   * - ``neuron-framework-equivalence``
     - Validation
     - Validates numerical equivalence between a HuggingFace model and its NxD Inference port.

Agents
------

Agents combine multiple skills into autonomous workflows. Each agent handles multi step
development scenarios end to end.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Agent
     - What it does
   * - ``neuron-nki-agent``
     - Unified entry point for NKI development. Auto selects the right workflow based on your request.
   * - ``neuron-nki-writer-agent``
     - Focuses on kernel authoring. Translates PyTorch, NumPy, or natural language into NKI code.
   * - ``neuron-nki-debugger-agent``
     - Autonomously analyzes compiler errors, searches docs for fixes, and applies corrections.
   * - ``neuron-nki-profile-analysis-agent``
     - Captures execution profiles on hardware and identifies inefficiencies.
   * - ``neuron-framework-autoport-agent``
     - Accepts model parameters and runs the full porting workflow from analysis through validation.

Supported environments
----------------------

The agents and skills run in these agentic coding environments.

- **Claude Code.** Install as a plugin from the Neuron Agentic Development repository.
- **Kiro.** Deploy using the provided setup script.

Writing and documentation skills work anywhere without hardware. Debugging, profiling,
analysis, and model porting skills require Trainium access.

Hardware and environment requirements
--------------------------------------

To use the full set of capabilities you need the following.

- A Trainium instance (``trn1`` or ``trn2``). The Amazon Linux 2023 Neuron DLAMI is a good starting point.
- The Neuron SDK installed on the instance (pre installed on Neuron DLAMIs, or install ``aws-neuronx-tools`` manually).
- A Python virtual environment with Neuron packages (``neuronx-cc``, ``torch-neuronx``, ``neuron-explorer``).
- Claude Code or Kiro installed on the same instance. The agent and hardware are co located.
- An Anthropic API key or equivalent credentials for the Claude model.

For writing and documentation tasks only, you can run the agent on any machine. Hardware is not required.

Limitations
-----------

- **Hardware required for on device skills.** Debugging, profiling, analysis, and model porting require Trainium access.
- **Claude model support.** Current agents target Claude models. Support for additional model providers depends on demand and capability parity.
- **Evolving capability set.** The agent and skill catalog is actively evolving. Treat the GitHub repository as the authoritative source for what is available today.

Responsible use
---------------

Agent generated code should be validated before you ship it. The debugging and analysis
skills check compilation and surface known issues, but they do not replace numerical
correctness testing against a reference implementation on your target instance type,
Neuron SDK version, and data types. Include the target instance type, Neuron SDK version,
and framework version in your prompts so the generated code matches your environment.

Related resources
-----------------

- :doc:`/nki/index` for Neuron Kernel Interface documentation.
- :doc:`/tools/neuron-explorer/index` for the profiling and analysis tool that agents invoke for profile capture.
- `Kiro documentation <https://kiro.dev/docs/>`__ for learning about Kiro.
- :ref:`nxdi-dev-ref-index` for NxD Inference developer guides.
