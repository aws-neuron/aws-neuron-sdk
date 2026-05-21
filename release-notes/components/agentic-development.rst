.. meta::
    :description: Complete release notes for Neuron Agentic Development across all AWS Neuron SDK versions.
    :keywords: neuron agentic development, agentic development, AI agents, AI skills, NKI, Kiro, Claude Code, release notes, aws neuron sdk
    :date-modified: 2026-05-21

.. _agentic-development_rn:

Component Release Notes for Neuron Agentic Development
=======================================================

**Latest version (in 2.30.0)**: 1.1

The release notes for Neuron Agentic Development, the open-source suite of AI
agents and skills that author, debug, profile, and analyze NKI kernels, port
HuggingFace models to NxD Inference, and validate numerical equivalence of
ported models on AWS Trainium from inside agentic IDEs such as Claude Code
and Kiro. Read these notes for the changes, improvements, and bug fixes in
each AWS Neuron SDK release. For an introduction to the feature, see
:ref:`neuron-agentic-development-overview`.

Source, installation instructions, and the current catalog of agents and
skills are maintained in the open-source repository:
`aws-neuron/neuron-agentic-development on GitHub
<https://github.com/aws-neuron/neuron-agentic-development>`_.

----

.. _agentic-development-2-30-0-rn:

Neuron Agentic Development (Neuron 2.30.0 Release)
--------------------------------------------------

New skills
~~~~~~~~~~

* **neuron-framework-autoport** — Ports HuggingFace transformer models to NxD Inference
  with full compilation and accuracy validation. The agent handles architecture analysis,
  implementation, compilation, inference testing, and greedy-token-match validation
  end to end.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-framework-autoport>`__)

* **neuron-framework-equivalence** — Validates numerical equivalence between a HuggingFace
  reference model and its NxD Inference port. Uses progressive 3-tensor R-ratio analysis,
  component-level testing, fault localization, and end-to-end accuracy verification.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-framework-equivalence>`__)

New and updated documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :ref:`Getting Started guide <neuron-agentic-development-getting-started>` covering
  installation, environment setup, and first agent invocation.
* Added :ref:`Autoport developer guide <neuron-framework-autoport>` — deep dive on how the
  Autoport skill works internally, including workflow stages, parameters, and environment setup.
* Added :ref:`Equivalence developer guide <neuron-framework-equivalence>` — deep dive on
  the Equivalence skill's 8-stage validation workflow and R-ratio methodology.
* Added :ref:`Autoport tutorial <autoport-tutorial>` — step-by-step walkthrough of porting
  a HuggingFace model using the Autoport agent.
* Added :ref:`Equivalence tutorial <equivalence-tutorial>` — step-by-step walkthrough of
  validating a ported model using the Equivalence agent.

----