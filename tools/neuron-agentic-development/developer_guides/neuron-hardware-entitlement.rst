.. meta::
   :description: Use the Neuron Hardware Entitlement skill to compute per-operator compute and HBM floors for a chosen LLM inference parallelism config on AWS Trainium, and to attribute a deployed graph's resident HBM back to those floors.
   :keywords: hardware entitlement, operator floor, roofline, compute-bound, memory-bound, HBM, KV cache, Trainium, LLM inference, optimization
   :date-modified: 2026-10-07

.. _neuron-hardware-entitlement:

Deep dive: Find the binding operator with the Hardware Entitlement skill
========================================================================

**Why read this guide?** This guide is for ML and performance engineers who have chosen a
parallelism config for a transformer LLM on AWS Trainium and need to know where to spend
optimization effort. It explains the Hardware Entitlement skill. The skill is an 
agent-driven workflow that computes, for every matmul-class operator in the model, the
theoretical minimum time it must pay on the tensor engine and on HBM. It identifies which
operator and which ceiling binds, and compares a deployed graph's actual resident HBM against
those floors.

**How to use this guide:** Use this guide after you have picked a config, typically with the
:ref:`Parallelism Planner skill <neuron-parallelism-planner>`. Use it when you need to decide
whether a slow kernel is limited by compute or by memory bandwidth, or which operator to
optimize first. Skip to the :ref:`three capabilities <entitlement-capabilities>` if you already
understand operator floors and component serialization.

This topic explores the Hardware Entitlement skill in depth. It covers the operator floor,
component composition, reconciliation against the planner, single-config entitlement across
a bucket regime, and actual-versus-floor HBM attribution. You need experience with transformer
inference, tensor parallelism, and roofline analysis to fully understand this content.

Prerequisites
-------------

Before you start, you must be familiar with the following:

- **Roofline analysis:** Arithmetic intensity, the ridge point, and the difference between
  compute-bound and memory-bound work.
- **Tensor parallelism:** Column-parallel and row-parallel projections, and the collectives
  between them. See :ref:`nxdi-parallelism-user-guide`.
- **Neuron compilation artifacts:** For HBM attribution only. How a compiled graph's HLO ENTRY
  signature lists its parameter inputs.

Overview
--------

A faster kernel only helps end-to-end latency if it targets the operator that actually binds,
and only if it targets the ceiling that operator is actually hitting. A memory-bound projection
cannot be helped by better tiling. A compute-bound attention kernel cannot be helped by
lower-precision weights. The Hardware Entitlement skill tells you which case applies before you
spend compile cycles.

For one parallelism config, the skill computes per-operator **floors**: lower bounds on the time
each operator must take, given the work it is required to do. The gap between a measured kernel
and its floor is the optimization headroom. An operator already near its floor
(measured ÷ floor ≈ 1.0) does not need further optimization.

The skill is the **operator-level companion** to the Parallelism Planner. The planner answers
*which config to deploy*. This skill answers *which operator binds on what within that config*.
Both use the same analytic engine and oracle. This skill imports them from the planner rather
than copying them, so the two views cannot drift. Summing the per-operator floors reproduces the
planner's tensor-engine, weight, and KV totals exactly.

The skill provides three capabilities:

1. **Per-operator floor:** For one ``{TP, PP, CP, bucket, bs, prefill}`` cell, compute the
   compute floor, HBM floor, arithmetic intensity, and binding constraint for every operator,
   and the serialized component floors.
2. **Single-config entitlement:** For one ``{TP, PP, CP}`` config across a whole bucket regime,
   compute each cell's cost decomposition, binding, decode throughput or prefill latency bound,
   and peak HBM.
3. **Actual-versus-floor HBM attribution:** Attribute every resident tensor in a deployed graph
   to an operator class and rank the classes by how far they exceed their floor.

.. important::
   The floor is a **lower bound, not a latency prediction**. Measured time is always at least
   the floor and is usually higher, because of dispatch overhead, synchronization, small-tile
   inefficiency, and engines the floor does not model. The floor is also scoped to the tensor
   engine (TE) and HBM. See :ref:`entitlement-engine-scope`.

.. _entitlement-supported-architectures:

Supported architectures
-----------------------

This skill parses models with the Parallelism Planner's ``arch_from_hf`` parser and costs them
with the planner's engine and oracle, so it supports exactly the same architectures. For the
full list, including the config fields each feature reads, see
:ref:`planner-supported-architectures`.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Supported
     - Not supported
   * - Decoder-only causal transformers with softmax attention: MHA, GQA, MQA, and MLA
     - Linear attention and recurrent layers, including Gated DeltaNet (GDN) and hybrid
       linear/full-attention models
   * - Uniform sliding-window attention, and mixed sliding-window and global layers
       (``layer_types``)
     - State-space layers, such as Mamba
   * - ``attention_k_eq_v`` (V aliases K on global layers)
     - Encoder-only and encoder-decoder models
   * - Dense gated MLP
     - Per-layer FFN widths other than the dense-then-MoE split
   * - MoE: routed experts, shared experts, dense-then-MoE splits, and a routed pool in
       parallel with a dense MLP
     - MoE configs using unrecognized field names, such as ``moe_num_experts``, and
       ``n_routed_experts`` combined with mixed ``layer_types``
   * - Tied or untied embeddings
     - Vision and audio towers (ignored; only the text decoder is costed)

The operator rows reflect this coverage. An attention layer produces ``q_proj``, ``k_proj``,
``v_proj``, ``o_proj``, and ``sdpa`` rows. An FFN produces dense ``gate_proj``, ``up_proj``, and
``down_proj`` rows, plus ``moe_``-prefixed rows and a ``moe_router`` row for MoE layers. There
are no rows for recurrent-state updates, convolutions, or other non-matmul layer work.

.. warning::
   Unsupported architectures are **not rejected**. A ``linear_attention`` layer is silently
   costed as full softmax attention or dropped from the layer count, and an unrecognized MoE
   config is costed as dense. The reconciliation check does not catch this, because the
   floors and the planner use the same parse. Before trusting floors for a new model,
   confirm that every ``layer_types`` value is ``sliding_attention`` or ``full_attention``,
   that the instance count per device matches ``num_hidden_layers``, and
   that a known weight magnitude is right. For example, a routed-expert weight read should
   be much larger than a dense-sized one.

.. _entitlement-supported-parallelism:

Supported parallelism
^^^^^^^^^^^^^^^^^^^^^

The floors use the planner's sharding rules, so the same parallelism support applies. For the
full restrictions, see :ref:`planner-axes`.

- **TP:** Supported. Caps KV sharding at ``min(TP, n_kv_heads)``.
- **CP:** Supported. Must divide TP. In decode, it only reclaims KV that TP replicated past
  ``n_kv_heads``. It has no effect on sliding-window layers, or on prefill unless the run-spec
  sets ``apply_prefill_dcp``.
- **EP:** Supported for MoE models through ``--ep``, as full EP across ``TP × EP`` ranks.
  Only routed experts are EP-sharded, and the router stays replicated.
- **DP:** Independent replicas do not change per-device floors, so there is no DP option.
- **PP:** Not supported. ``--pp`` and ``"PP"`` in a regime file must be 1, and any other
  value raises an error.

.. _entitlement-prerequisites:

Hardware and software requirements
-----------------------------------

The floor computations do not need Neuron hardware.

* **Python:** 3.12.
* **Per-operator and single-config scripts:** ``torch`` (CPU build is enough; the oracle runs on
  the ``meta`` device).
* **HBM attribution script:** Python standard library only.
* **Parallelism Planner skill:** Must be installed side by side under the same ``skills/``
  directory. Deploying Neuron Agentic Development installs both.
* **Run-spec:** The same run-spec JSON the planner uses. See :ref:`planner-inputs`.
* **For HBM attribution only:** A text manifest of a compiled graph's ENTRY signature, captured
  from a deployment on a Trainium instance.

.. _entitlement-concepts:

Key concepts
------------

The operator floor
^^^^^^^^^^^^^^^^^^

For each operator, per device, for the given config:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Term
     - Definition
   * - **Compute floor**
     - Operator FLOPs ÷ tensor-engine peak.
   * - **HBM floor**
     - (Weight bytes + KV bytes) ÷ HBM bandwidth. Only ``sdpa`` reads KV bytes. Only
       projections own weights, so ``sdpa``'s HBM floor is pure KV.
   * - **Activation interface**
     - Input plus output activation volume for the operator. Reported for diagnosis. Under the
       default policy, it is not charged per operator (see below).
   * - **Arithmetic intensity (AI)**
     - FLOPs ÷ mandatory HBM bytes. Compare it against the ridge point,
       ``TE peak ÷ HBM bandwidth`` (about 221 FLOP/byte on trn2).
   * - **MUST-binding**
     - ``compute`` if the compute floor is at least the HBM floor, otherwise ``memory``.

Communication is not part of any operator's binding. Collectives are a per-layer property and
are reported in the planner cell's ``comm`` term.

Components and why they serialize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operator rows explain the work. They are not independent kernel latencies to add up. The
skill groups operators into **components**:

- An attention body or an FFN body is the compute interval between its gathered input and
  its pre-collective output. A component never contains a collective.
- The model ends with a terminal ``lm_head`` component.
- Under sequence-parallel prefill, the residual and norm interval between a reduce-scatter and
  the next all-gather is a separate bridge. It is outside the current floor.

Within a component, the engine assumes ideal fusion: internal operator interfaces stay on chip,
and TE work overlaps with HBM traffic. Each component pays one roofline:

.. code-block:: text

   component_floor(c) = max(TE_work(c) / TE_peak, mandatory_HBM_bytes(c) / HBM_bandwidth)

   mandatory_HBM_bytes(c) = weight_bytes(c) + KV_bytes(c)
                          + unique_component_input_bytes(c) + unique_component_output_bytes(c)

   model_floor = sum(component_floor(c)) + collective_time

Components are added, not overlapped, because each attention and FFN body ends at an
all-reduce or reduce-scatter whose result the next body consumes. A single model-wide
``max(TE, HBM)`` would assume overlap the graph does not allow. Components are also natural
optimization units: you can replace or tune an attention variant, an FFN path, or the lm_head
independently, and the component floor is the minimum latency available to that region.

Setting ``"activation_io_policy": "per_operator"`` in the run-spec replaces the component
boundary with every operator's input and output. Use it to model a decomposed or eager
execution.

.. _entitlement-engine-scope:

Engine scope
^^^^^^^^^^^^

Every report carries ``"engine_scope": ["te", "hbm"]``. The floor certifies tensor-engine work
and mandatory HBM traffic only. Vector engine (VE) and scalar engine (SC) work is not modeled
and is excluded from the binding maximum. That does not mean VE and SC work is zero. It means
the floor is valid but might be looser than a future floor that includes them. If a profile
shows a component bound on VE time, that is outside what this floor claims.

Reconciliation
^^^^^^^^^^^^^^

Every run of the per-operator and single-config scripts prints a reconciliation of the summed
operator floors against the planner's cell:

- **Weight read** versus the cell's ``weight_load``: matches exactly.
- **KV read** versus the cell's ``kv_load``: matches exactly.
- **Compute** versus the cell's ``matmul``: matches within a small, known oracle-versus-analytic
  delta, usually a few percent at most.

.. warning::
   Treat any reconciliation gap above about 1% as a stop condition. It means the operator
   breakdown and the planner disagree, usually because of a model the oracle does not fully
   cover or a duplicated formula. Do not report numbers until the gap is closed.

Data sources
^^^^^^^^^^^^

The meta-device oracle owned by the planner is exact for matmul shapes and parameter sizes,
so attention-projection, FFN, router, and lm_head FLOPs and weight bytes come from it. Two
quantities come from the analytic engine instead:

- **KV-cache bytes:** On ``meta``, attention decomposes into ``bmm`` and ``softmax``, so the
  oracle sees no KV reads.
- **Causal and windowed SDPA FLOPs:** The engine charges causal active pairs and applies
  sliding-window caps. The oracle counts a dense rectangle.

Activation shapes are derived from the sharding rules in the planner's ``planner.shard``
module.

.. _entitlement-capabilities:

The three capabilities
----------------------

The examples below run from your project directory:

.. code-block:: bash

   # ~/.claude for Claude Code, ~/.kiro for Kiro
   export NEURON_AGENTIC_DEVELOPMENT_ROOT=~/.claude
   SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-hardware-entitlement/scripts

Every script requires ``--runspec`` with a path to a run-spec file. Before analyzing a new
model, run the planner's ``validate_oracle.py`` on it. If the model-level counts do not
reconcile, the operator-level floors will not either.

.. _entitlement-operator-floor:

Capability 1: Per-operator floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Decode cell
   python3.12 $SKILL/operator_entitlement.py --runspec ./my-model/my-model-run.json \
     --tp 8 --bucket 8192 --bs 16

   # Prefill cell
   python3.12 $SKILL/operator_entitlement.py --runspec ./my-model/my-model-run.json \
     --tp 8 --bucket 8192 --bs 1 --prefill

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``--tp`` / ``--cp``
     - 8 / 1
     - Tensor- and context-parallel degrees for the cell. ``--cp`` must divide ``--tp``.
   * - ``--ep``
     - 1
     - Expert-parallel degree for MoE models. ``--ep E`` models the planner's full-EP
       config with ``DP = E``: routed experts are sharded across ``TP × E`` ranks.
       Use the EP degree the planner recommended.
   * - ``--pp``
     - 1
     - Must be 1. Pipeline parallelism is not supported, and any other value raises an
       error.
   * - ``--bucket``
     - 8192
     - Context length in tokens.
   * - ``--bs``
     - 8
     - Batch size.
   * - ``--prefill``
     - off
     - Evaluate prefill instead of decode.
   * - ``-o`` / ``--out``
     - ``agent_artifacts/tmp/operator_entitlement_<regime>.json``
     - Output JSON path.
   * - ``--project-root``
     - current directory
     - Where ``agent_artifacts/tmp/`` is created.

The output has three parts:

1. **Per-operator roll-up**, one row per (operator, attention group), sorted so the largest
   floors appear first. Columns: instances per device, compute floor, HBM floor (split into
   weight and KV), activation interface, arithmetic intensity, and MUST-binding.
2. **Reconciliation** against the planner's analytic cell.
3. **Component composition**: the number of component instances, the TE and HBM totals,
   and the serialized component floor.

**Row naming.** Feed-forward rows tell you which path you are reading. The always-on dense MLP,
including any shared experts, uses bare names (``gate_proj``, ``up_proj``, ``down_proj``). The
routed expert pool uses ``moe_``-prefixed names. A MoE layer also has a ``moe_router`` row. It
is the only row that is replicated rather than sharded, so its per-device weight stays flat as
TP grows.

.. _entitlement-config:

Capability 2: Single-config entitlement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3.12 $SKILL/config_entitlement.py --runspec ./my-model/my-model-run.json

This pins one ``{TP, PP, CP, EP}`` config, taken from the first row of the run-spec's
``configs`` array, and evaluates every (bucket, batch, regime) cell in that array at the
pinned config. For each cell it reports:

- The global cost decomposition: ``weight_load``, ``kv_load``, ``activation_load``, ``matmul``,
  and the additive ``comm`` term, in milliseconds.
- The binding constraint, for example ``weight HBM read``, ``KV HBM read``, or ``matmul``.
- Peak HBM per device.
- The operator-level reconciliation, with a flag on any cell that drifts past a few percent.

To evaluate a different config or set of cells, pass a regime file with ``--regime``:

.. code-block:: json

   {
     "TP": 8, "PP": 1, "CP": 1, "EP": 1,
     "cells": [
       {"category": "8k_decode",  "bucket": 8192, "bs": 16, "prefill": false},
       {"category": "8k_prefill", "bucket": 8192, "bs": 1,  "prefill": true}
     ]
   }

For MoE models, set ``EP`` to the value the planner recommended. The planner enables EP by
default for models with routed experts, and evaluating such a config at ``EP=1`` overstates
the per-device expert weight floor.

Use ``-o`` to set the output path.

.. _entitlement-hbm-attribution:

Capability 3: Actual-versus-floor HBM attribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This capability answers *which operators use more HBM than their floor, and by how much*.
It does not answer *why*. Root-causing an overage is a separate step that starts from this
table.

**Where resident bytes live.** A compiler's per-pass memory-allocation dumps track only the
allocator-managed scratch memory: collectives, matmul temporaries, and spills. They do not
track model weights or the paged KV pool. Weights and the KV pool enter the compiled graph as
fixed-address parameter inputs. The authoritative resident manifest is therefore the compiled
graph's **ENTRY signature**, which lists every weight and KV tensor with its sharded shape.

**Capture the manifest:**

1. Compile the smallest bucket and batch that reproduces the footprint of interest, so
   activations are at their minimum and weights and KV dominate.
2. Add compiler flags that dump the graph. For an over-budget config, also disable the
   HBM-usage asserts so the host compile finishes and writes the dump. Gate these behind an
   environment variable so the normal serving path is unaffected.
3. Bypass the compile cache. A cache hit skips the compiler and produces no dump.
4. Extract the ENTRY signature from the canonicalized HLO into a text file.

**Run the attribution** against a decode floor computed at the same TP, bucket, and batch:

.. code-block:: bash

   python3.12 $SKILL/operator_entitlement.py --runspec ./my-model/my-model-run.json \
     --tp 32 --bucket 65536 --bs 1

   python3.12 $SKILL/hbm_attribution.py \
     --entry-hlo ./captured_entry_manifest.hlo \
     --floor-json agent_artifacts/tmp/operator_entitlement_decode.json \
     --runspec ./my-model/my-model-run.json \
     -o agent_artifacts/tmp/actual_vs_floor.json

The script prints the parsed tensor count and total size. That total should match the
"Unallocated Tensors" figure in the compiler's HBM-usage log, which confirms the manifest is
complete.

**How attribution works.** No parameter names survive in the ENTRY signature, so tensors are
matched by **shape signature**. Signatures are derived from the model's architecture and the
floor's TP degree. A deployed graph often fuses projections the oracle reports separately (for
example, ``qkv`` or ``gate_up``), so the script registers both fused and split signatures and
folds the matching floors together.

**Reading the diff.** Each operator class reports actual bytes, floor bytes, overage, and ratio,
ranked by absolute overage.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Result
     - Interpretation
   * - Weight class at ×1.00
     - At entitlement. No action needed.
   * - ``embed_or_lm_head`` at ×2.00
     - The graph stores separate embedding and lm_head tables while the model ties them. A
       real, expected finding.
   * - Fused ``qkv`` class slightly above ×1.00
     - The fused weight stores Q, K, and V even where the model aliases V to K. Expected.
   * - KV rows with a large ratio
     - **Not an operator overage.** The actual KV figure is the paged-pool reservation set by
       the memory-utilization policy. The floor is the per-request KV for one bucket and
       batch. Every KV row is flagged.

.. _entitlement-reading:

Reading the output
------------------

**Decode is memory-bound. Prefill is compute-bound.** At small decode batches, each weight read
is amortized over only a few tokens, so projections land at an arithmetic intensity roughly
equal to the batch size, far below the ridge point. ``sdpa`` has no weights to amortize, so it
is lowest of all. Whole-prompt prefill reuses each weight across every prompt token, so
projection arithmetic intensity reaches the prompt length and binds on compute. The lm_head
projects only sampled positions and stays memory-bound in prefill.

**The binding is the routing signal.**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Binding
     - Lever
   * - ``memory`` (weight read)
     - Fewer or larger HBM reads, lower-precision weights, or a larger batch. A faster matmul
       kernel will not help.
   * - ``memory`` (KV read)
     - FP8 KV cache, or a TP degree that does not replicate KV heads. If TP already exceeds
       ``n_kv_heads``, context parallelism reclaims the replicated cache.
   * - ``compute``
     - Tiling, occupancy, and tensor-engine utilization work.
   * - Measured ≈ floor
     - Already at entitlement. Spend effort elsewhere.

**Check for collectives before blaming a kernel.** At high TP, and especially under
sequence-parallel ``all_gather → matmul → reduce_scatter`` patterns, collective wall time can
dominate device time. In one validated case, an 8B prefill-only embedding model at TP=16 on
trn2, the TE floor was about 1.4 ms, but measured device time was about 18 ms, and collectives
accounted for 43% of the device window, almost entirely serialized with compute. Compare the
planner's ``comm_ms`` with collective wall time in a system trace before concluding that an
operator is inefficient.

**Kernel reachability depends on the config.** Whether a fused kernel activates can depend on
tiling constraints at the chosen TP degree. The floor does not change, but how closely an
implementation can approach it does.

.. _entitlement-file-organization:

File organization
-----------------

All three scripts write under ``<project root>/agent_artifacts/tmp/``. The project root is the
current directory unless you set ``--project-root`` or ``$NEURON_PROJECT_ROOT``. The planner
honors the same variable, so a planning run and an entitlement run share one evidence trail.

.. code-block:: text

   <project root>/agent_artifacts/tmp/
   ├── operator_entitlement_decode.json    # Per-operator floor, decode cell
   ├── operator_entitlement_prefill.json   # Per-operator floor, prefill cell
   ├── config_entitlement.json             # Single-config entitlement across cells
   └── actual_vs_floor.json                # HBM attribution diff

.. _entitlement-tools-reference:

Tools reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Role
   * - ``scripts/operator_entitlement.py``
     - Per-operator floor engine. Produces per-instance rows and a per-(operator, group)
       roll-up for one device, reconciles against the planner's cell, and exposes the
       planner's component activation boundaries.
   * - ``scripts/config_entitlement.py``
     - Single-config entitlement over a bucket regime: per-cell decomposition, binding,
       bound, peak HBM, and attached reconciliation.
   * - ``scripts/hbm_attribution.py``
     - Parses an ENTRY-signature manifest, attributes tensors to operator classes by shape,
       and diffs actual against floor.
   * - ``scripts/_planner.py``
     - Locates the sibling ``neuron-parallelism-planner/scripts`` directory, puts its engine
       and oracle on the import path, and loads the run-spec. The single seam between the two
       skills.

.. _entitlement-debugging:

Common issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Solution
   * - ``error: run-spec not found``
     - ``--runspec`` requires a file path. Model names are not resolved.
   * - ``ModuleNotFoundError: torch``
     - ``operator_entitlement.py`` and ``config_entitlement.py`` need ``torch``. A CPU build is
       enough.
   * - Cannot import the planner engine
     - The Parallelism Planner skill must be installed beside this skill under the same
       ``skills/`` directory.
   * - Reconciliation gap above about 1%
     - Stop. Check the architecture parse, layer count, and MoE field names, then rerun the
       planner's ``validate_oracle.py``.
   * - MoE ``moe_gate_proj`` weight looks dense-sized
     - The MoE structure was not parsed. Check the config's expert field names.
   * - Instance count per device is lower than the model's layer count, or a hybrid model
       shows only softmax-attention rows
     - ``layer_types`` contains an unsupported value, such as ``linear_attention``. See
       :ref:`entitlement-supported-architectures`.
   * - Floor far below measured time at high TP
     - Likely collective-bound. Compare against collective wall time in a system trace.
   * - Measured kernel beats its own floor
     - A hardware constant is wrong. Calibrate it with microbenchmarks; see
       ``references/microbench-calibration.md``.
   * - Attribution total does not match the HBM-usage log
     - The manifest is incomplete. Recapture the ENTRY signature with the compile cache
       bypassed.

.. _entitlement-guidelines:

Project guidelines
------------------

The Hardware Entitlement skill enforces these rules:

- **Do not edit the scripts to analyze a new model.** Point them at a new run-spec.
- **Extend the oracle in the planner skill**, not here. This skill inherits the fix.
- **Validate the planner's engine against the oracle first** for any new model.
- **Run every scenario in the run-spec**, not just the first one that produces output.
- **Confirm the binding operator before recommending an optimization.** Rank operators by
  measured time, not just floor time.
- **Flag likely collective-bound configs** explicitly so downstream profiling tests that
  hypothesis.
- **Report floors as analytic** until they are confirmed by on-hardware profiling.

.. _entitlement-knowledge-base:

Knowledge base
--------------

The skill's ``references/`` directory contains:

- ``operator-floors.md``: The floor definition, data sources, sharding, the roll-up,
  component serialization, reconciliation, and caveats.
- ``actual-vs-floor-hbm.md``: Where resident bytes live, how to capture the manifest, and how
  attribution and the diff work.
- ``microbench-calibration.md``: How to calibrate the tensor-engine peak and HBM bandwidth
  constants by microbenchmarking.
- ``worked-example.md``: Decode and prefill floors, single-config entitlement, and HBM
  attribution for a 31B mixed-attention dense model on trn2.
- ``worked-example-embedding-prefill.md``: A prefill-only embedding model that is
  compute-floor-dominated at low TP and collective-bound at high TP.

Related resources
-----------------

- :ref:`neuron-parallelism-planner`: Choose the config this skill analyzes.
- :ref:`hardware-entitlement-tutorial`: Step-by-step tutorial for this skill.
- :ref:`nxdi-parallelism-user-guide`: Tensor and sequence parallelism in NxD Inference.
- :doc:`/tools/neuron-explorer/index`: The profiling tool used to validate floors against
  measured time.
- `Neuron Agentic Development <https://github.com/aws-neuron/neuron-agentic-development>`_: Source for the Hardware Entitlement skill.
