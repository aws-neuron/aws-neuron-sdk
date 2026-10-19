.. meta::
    :description: Complete release notes for Neuron Agentic Development across all AWS Neuron SDK versions.
    :keywords: neuron agentic development, agentic development, AI agents, AI skills, NKI, Kiro, Claude Code, release notes, aws neuron sdk
    :date-modified: 2026-10-07

.. _agentic-development_rn:

Component Release Notes for Neuron Agentic Development
=======================================================

**Latest version (in 2.33.0)**: 1.4

The release notes for Neuron Agentic Development, the open-source suite of AI
agents and skills that author, debug, profile, and analyze NKI kernels, port
HuggingFace models to NxD Inference, validate numerical equivalence of
ported models, and plan and analyze LLM inference performance on AWS Trainium from inside agentic IDEs such as Claude Code
and Kiro. Read these notes for the changes, improvements, and bug fixes in
each AWS Neuron SDK release. For an introduction to the feature, see
:ref:`neuron-agentic-development`.

Source, installation instructions, and the current catalog of agents and
skills are maintained in the open-source repository:
`aws-neuron/neuron-agentic-development on GitHub
<https://github.com/aws-neuron/neuron-agentic-development>`_.

----

.. _agentic-development-2-33-0-rn:

Neuron Agentic Development (Neuron 2.33.0 Release)
--------------------------------------------------

Improvements
~~~~~~~~~~~~

* **NKI 0.7.0 skill support.** The NKI agentic skills are updated for compatibility with NKI 0.7.0.

New skills
~~~~~~~~~~

* **neuron-parallelism-planner** — Plans tensor, context, and data parallelism (and expert
  parallelism for MoE models) and batch size for transformer LLM inference on Trainium
  before you run on hardware. Takes a run-spec (the model's HuggingFace ``config.json``,
  hardware constants, and decode ITL and prefill TTFT SLAs), validates its analytic
  entitlement model against a meta-device FLOP-counter oracle, sweeps every config and
  context bucket, and selects the config with the highest per-node throughput that meets
  both SLAs. Reports predicted ITL, TTFT, throughput, per-device HBM, and the binding
  constraint, and models KV-cache replication, FP8 weights and KV cache, and speculative
  decoding. Supports decoder-only softmax-attention transformers (MHA, GQA, MQA, MLA, and
  mixed sliding-window and global attention) with dense or MoE FFNs. Linear-attention,
  recurrent (for example, Gated DeltaNet), and state-space layers are not supported. Sweeps
  tensor, context, and data parallelism, and full expert parallelism for MoE models;
  pipeline parallelism is not supported. Plans single-node, colocated serving only.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-parallelism-planner>`__)

* **neuron-hardware-entitlement** — Computes per-operator compute and HBM floors for one
  chosen parallelism config and identifies which operator binds on the tensor engine,
  weight reads, or KV-cache reads. Rolls operator floors into serialized attention, FFN, and
  lm_head component floors that reconcile exactly with the Parallelism Planner, evaluates a
  config across a whole bucket regime, and attributes a deployed graph's resident HBM back to
  operator floors to rank overages. Shares the Parallelism Planner's run-spec, engine, and
  oracle, and supports the same architectures.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-hardware-entitlement>`__)

* **neuron-framework-profiling-vllm-neuron** — Captures device and system profiles from a
  running vLLM Neuron server on Trainium, isolating the prefill pass and steady-state decode
  into separate traces so each can be attributed on its own. Arms the profiler on a scheduler
  event rather than a timer, falls back to a decode-only server when a timed trigger cannot
  fire, manages the server lifecycle so a capture does not orphan device-holding workers, and
  records a run specification and manifest that make the capture reproducible.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-framework-profiling-vllm-neuron>`__)

* **neuron-framework-profile-analysis-vllm-neuron** — Turns a capture into a bottleneck
  localization. Ingests the device and system traces into queryable parquet, establishes the
  session workload and step time from the live dispatch records, computes memory and compute
  roofline floors from the capture's own device constants, partitions each step at its
  collectives, and rolls instructions up to model components so attention, MLP, and lm_head
  time and HBM traffic are separable. Pairs with ``neuron-hardware-entitlement`` to
  cross-validate a profiled step against an analytical floor.
  (`Source <https://github.com/aws-neuron/neuron-agentic-development/tree/main/skills/neuron-framework-profile-analysis-vllm-neuron>`__)

New agents
~~~~~~~~~~

* **neuron-parallelism-planner-agent** — Reconciles a joint latency and throughput SLA into a
  single-node ``DP × TP`` config, reports the full TP sweep and the binding constraint, and
  hands off to the profiling skills for on-hardware validation.

* **neuron-hardware-entitlement-agent** — Computes per-operator floors for a config handed off
  from the planner, flags likely collective-bound configs, and routes optimization effort to
  the operator with the most headroom.

New and updated documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added :ref:`Parallelism Planner developer guide <neuron-parallelism-planner>` — deep dive on
  the run-spec, the component cost model, KV-cache replication, oracle validation, and serving
  objectives.
* Added :ref:`Hardware Entitlement developer guide <neuron-hardware-entitlement>` — deep dive
  on operator floors, component serialization, reconciliation, and actual-versus-floor HBM
  attribution.
* Added :ref:`Parallelism Planner tutorial <parallelism-planner-tutorial>` — step-by-step
  walkthrough of planning a serving config for Granite-4.1-8B on trn2.
* Added :ref:`Hardware Entitlement tutorial <hardware-entitlement-tutorial>` — step-by-step
  walkthrough of finding the binding operator for the planner's recommended config.
* Added :ref:`Profiling vLLM Neuron developer guide <neuron-framework-profiling-vllm-neuron>`
  — deep dive on phase isolation, the server lifecycle, and the capture input and output
  contract.
* Added :ref:`Profile Analysis developer guide <neuron-framework-profile-analysis-vllm-neuron>`
  — deep dive on ingestion, session workload timing, roofline floors, collective
  partitioning, and component attribution.
* Added :ref:`Profiling vLLM Neuron tutorial <profiling-vllm-neuron-tutorial>` — step-by-step
  walkthrough of capturing separate prefill and decode profiles.
* Added :ref:`Profile Analysis tutorial <profile-analysis-vllm-neuron-tutorial>` —
  step-by-step walkthrough of analyzing a capture and confirming the result is sound.

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
