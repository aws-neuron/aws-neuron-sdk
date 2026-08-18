.. meta::
    :description: Complete release notes for Neuron Agentic Development across all AWS Neuron SDK versions.
    :keywords: neuron agentic development, agentic development, AI agents, AI skills, NKI, Kiro, Claude Code, release notes, aws neuron sdk
    :date-modified: 2026-07-29

.. _agentic-development_rn:

Component Release Notes for Neuron Agentic Development
=======================================================

**Latest version (in 2.32.0)**: 1.3

The release notes for Neuron Agentic Development, the open-source suite of AI
agents and skills that author, debug, profile, and analyze NKI kernels, port
HuggingFace models to NxD Inference, and validate numerical equivalence of
ported models on AWS Trainium from inside agentic IDEs such as Claude Code
and Kiro. Read these notes for the changes, improvements, and bug fixes in
each AWS Neuron SDK release. For an introduction to the feature, see
:ref:`neuron-agentic-development`.

Source, installation instructions, and the current catalog of agents and
skills are maintained in the open-source repository:
`aws-neuron/neuron-agentic-development on GitHub
<https://github.com/aws-neuron/neuron-agentic-development>`_.

----

.. _agentic-development-2-32-0-rn:

Neuron Agentic Development (Neuron 2.32.0 Release)
--------------------------------------------------

Improvements
~~~~~~~~~~~~

* **New Autoport skill for vLLM Neuron.** The new ``neuron-framework-autoport-vllm-neuron``
  skill ports GPU-compliant transformer models to the vLLM Neuron backend on AWS Trainium.
  The agent researches the model architecture, sizes tensor and expert parallelism,
  generates the model implementation, registers it in the vLLM model registry, and
  validates accuracy — so you can bring a model to vLLM Neuron serving without writing the
  port by hand. See the
  :ref:`Autoport vLLM Neuron deep dive <neuron-framework-autoport-vllm-neuron>`.

* **Model Equivalence now validates vLLM Neuron ports.** The
  :ref:`neuron-framework-equivalence` skill adds a vLLM Neuron adapter so you can verify
  numerical equivalence for ports that target the vLLM Neuron backend, not just NxD
  Inference. Select it with ``target_stack: vllm_neuron``, or let the skill auto-detect it
  from the modeling file's imports. See the
  :ref:`vLLM Neuron section of the Equivalence deep dive <equiv-vllm-neuron>`.

* **NKI 0.6.0 skill support.** The NKI agentic skills are updated for compatibility with
  NKI 0.6.0. This includes migration guidance for the deprecated ``nl.dynamic_range`` and
  ``while reg:`` loop constructs — the agent can walk you through converting them to the
  new :func:`~nki.language.fori_loop` and :func:`~nki.language.while_loop` APIs.

New and updated documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :ref:`Autoport vLLM Neuron deep dive <neuron-framework-autoport-vllm-neuron>` —
  how the skill ports models to the vLLM Neuron backend and validates the result.
* Added :ref:`Autoport vLLM Neuron tutorial <autoport-vllm-neuron-tutorial>` — end-to-end
  walkthrough of porting a model to vLLM Neuron and serving it.
* Updated the :ref:`Equivalence developer guide <neuron-framework-equivalence>` with a
  vLLM Neuron adapter section.
* Updated the :ref:`Equivalence tutorial <equivalence-tutorial>` with a step for validating
  a vLLM Neuron port.

----

.. _agentic-development-2-31-0-rn:

Neuron Agentic Development (Neuron 2.31.0 Release)
--------------------------------------------------

Improvements
~~~~~~~~~~~~

* **NKI 0.5.0 skill support.** The NKI agentic skills are updated for compatibility with NKI 0.5.0.

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
