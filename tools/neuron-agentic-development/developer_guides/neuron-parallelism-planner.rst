.. meta::
   :description: Use the Neuron Parallelism Planner skill to choose tensor, context, and data parallelism and batch size for transformer LLM inference on AWS Trainium before running on hardware.
   :keywords: parallelism planner, tensor parallelism, context parallelism, data parallelism, expert parallelism, LLM inference, Trainium, entitlement, roofline, ITL, TTFT, KV cache
   :date-modified: 2026-10-07

.. _neuron-parallelism-planner:

Deep dive: Plan inference parallelism with the Parallelism Planner skill
========================================================================

**Why read this guide?** This guide is for ML engineers who need to decide how to shard a
transformer LLM for serving on AWS Trainium. It covers how many NeuronCores to use, which tensor,
context, and data parallelism degrees to pick, and what batch size to run at. It explains the
Parallelism Planner skill. The skill is an agent-driven workflow that turns a model's
HuggingFace ``config.json`` and a set of hardware constants into a ranked parallelism
recommendation, using an analytic hardware-entitlement model. You get the recommendation
before you compile anything or reserve a Trainium instance.

**How to use this guide:** Read this guide before your first on-hardware sweep for a new model,
or when a deployed config misses its latency or throughput target and you want to know whether
a different sharding could meet it. Skip to the :ref:`workflow <planner-workflow>` if you
already understand run-specs and serving objectives. For per-operator analysis of one chosen
config, see the :ref:`Hardware Entitlement skill <neuron-hardware-entitlement>`.

This topic explores the Parallelism Planner skill in depth. It covers the run-spec input, the
three-layer analytic engine, oracle validation, the serving objectives and SLA gates, the
parallelism axes the planner sweeps, and how to read its output. You need experience with
transformer inference, tensor parallelism, and basic roofline analysis to fully understand this
content.

Prerequisites
-------------

Before you start, you must be familiar with the following:

- **Transformer inference phases:** The difference between prefill (processing the prompt) and
  decode (generating one token per step), and what time-to-first-token (TTFT) and inter-token
  latency (ITL) measure.
- **Parallelism concepts:** How tensor, context, and data parallelism split work across
  NeuronCores. See :ref:`nxdi-parallelism-user-guide`.
- **Roofline analysis:** The difference between compute-bound and memory-bandwidth-bound work.

Overview
--------

Picking a parallelism config by trial and error on hardware is slow. Each candidate needs a
compile, a warm-up, and a benchmark. The Parallelism Planner skill narrows the search before
you touch hardware. It computes a theoretical lower bound, called the *entitlement*, on decode
ITL, prefill TTFT, throughput, and per-device HBM for every candidate config. It then selects
the config that best meets your serving objective.

The planner is an **entitlement model, not a measurement**. Real latencies are higher because
of dispatch overhead, synchronization, idle gaps, and kernel inefficiency. Use the planner's
recommendation to decide where to point the first on-hardware sweep, then validate the
prediction with the profiling skills.

The skill handles:

- **Dense and Mixture of Experts (MoE) models:** Including always-on shared experts and
  dense-then-MoE layer splits.
- **Attention variants:** Standard multi-head attention (MHA), grouped-query attention (GQA),
  Multi-head Latent Attention (MLA), and mixed sliding-window and global attention.
- **Precision options:** BF16 or FP8 weights, FP8 KV cache, and FP8 compute.
- **Trainium generations:** trn1, trn2, and trn3. Each generation is described by a
  hardware-constants file.

Linear-attention, recurrent, and state-space layers are not supported. See
:ref:`planner-supported-architectures` for the full list.

The workflow has four stages:

1. **Run-spec authoring:** Describe the model (pristine HF ``config.json``), the hardware
   constants, and your SLAs in a run-spec JSON file.
2. **Oracle validation:** Confirm the analytic engine's FLOP and weight counts for this model
   match a shape-exact meta-device oracle.
3. **Sweep and select:** Evaluate every config in the parallelism × context-bucket grid at
   its maximum feasible batch size, gate on the SLAs, and rank the survivors.
4. **Cell inspection:** Break a single operating point down per component and layer type to
   see what binds, and hand off to profiling for validation.

.. important::
   The planner models **single-node, colocated serving only**. One replica serves both
   prefill and decode, and every config must fit inside one node
   (``TP × DP ≤ cores_per_node``). Disaggregated prefill and decode, split pools, and any
   multi-node or multi-instance topology are out of scope. The engine raises an error if you
   request them. If no single colocated config meets your SLAs, the planner reports the
   best achievable config and the gap instead of proposing a different topology.

.. _planner-supported-architectures:

Supported architectures
-----------------------

The planner models **decoder-only, causal transformer LLMs with softmax attention**. The
architecture is read from the HF ``config.json`` by the ``arch_from_hf`` parser. The
following table lists what the parser and oracle model today.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Area
     - Supported
     - Config fields read
   * - Attention
     - Multi-head (MHA), grouped-query (GQA), and multi-query (MQA) softmax attention.
     - ``num_attention_heads``, ``num_key_value_heads``, ``head_dim`` (derived from
       ``hidden_size`` when absent)
   * - Latent attention
     - Multi-head Latent Attention (MLA) with a compressed KV cache, as in DeepSeek-V2/V3.
     - ``kv_lora_rank``, ``q_lora_rank``, ``qk_nope_head_dim``, ``qk_rope_head_dim``,
       ``v_head_dim``
   * - Sliding-window attention
     - Uniform sliding window, and mixed sliding-window and full-context layers with
       separate KV-head count and head size for the global layers, as in Gemma.
     - ``sliding_window``, ``layer_types`` (``sliding_attention`` / ``full_attention``),
       ``num_global_key_value_heads``, ``global_head_dim``
   * - K/V aliasing
     - Global layers where V aliases K.
     - ``attention_k_eq_v``
   * - Dense FFN
     - Gated MLP (gate, up, down projections).
     - ``intermediate_size``
   * - Mixture of Experts
     - Routed experts with top-k routing, always-on shared experts, dense-then-MoE layer
       splits, and a routed pool in parallel with a dense MLP.
     - ``n_routed_experts`` / ``num_experts`` / ``num_local_experts``,
       ``num_experts_per_tok`` / ``top_k_experts``, ``moe_intermediate_size``,
       ``n_shared_experts`` / ``shared_expert_intermediate_size``,
       ``first_k_dense_replace``, ``enable_moe_block``
   * - Embeddings
     - Tied or untied embedding and lm_head tables.
     - ``tie_word_embeddings``, ``vocab_size``
   * - Config layout
     - Flat configs, and multimodal configs with a nested ``text_config``. Only the text
       decoder is modeled.
     - ``text_config``

Models the planner has been exercised on include Granite-4.1-8B (dense GQA, tied
embeddings), Gemma-4-31B (mixed sliding and global attention, ``attention_k_eq_v``),
Gemma-4-26B-A4B (MoE in parallel with a dense MLP), and Devstral-2-123B (untied embeddings,
FP8 weights).

Not supported
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Architecture
     - Status
   * - **Linear attention and recurrent layers**, including Gated DeltaNet (GDN) and other
       hybrid linear/full-attention models
     - Not modeled. The engine has no representation for recurrent state or the non-tensor-
       engine work these layers do.
   * - **State-space layers** (for example, Mamba)
     - Not modeled.
   * - **Encoder-only and encoder-decoder models**
     - Not modeled. The engine assumes causal self-attention with a KV cache, and has no
       cross-attention.
   * - **Vision and audio towers** in multimodal models
     - Ignored. Only the text decoder in ``text_config`` is costed.
   * - **Per-layer FFN widths** other than the dense-then-MoE split
     - Not modeled.
   * - **MoE with** ``n_routed_experts`` **combined with mixed** ``layer_types``
     - Raises ``NotImplementedError``.
   * - **MoE configs that use other field names** (for example, ``moe_num_experts`` and
       ``moe_top_k``)
     - Not recognized. The model is parsed as dense.
   * - **Operators that exist only in a kernel language**
     - Must be expressed in the oracle by hand before validation. See
       ``references/oracle-construction.md``.

.. warning::
   The parser does **not** reject unsupported architectures. It mis-models them without an
   error. In ``layer_types``, it recognizes only ``sliding_attention`` and
   ``full_attention``. Depending on the other layer types present, it then either treats
   unrecognized layers, such as ``linear_attention``, as full softmax attention, or drops
   them from the layer count. An unrecognized MoE config is planned as a dense model.
   Before you trust any plan for a new model, confirm that every value in ``layer_types``
   is ``sliding_attention`` or ``full_attention``, that the parsed layer count equals
   ``num_hidden_layers``, and that the corrected weight total matches the model card. Do
   not plan only the layers that happen to parse. To support a new architecture, extend
   the engine and oracle first.

.. _planner-prerequisites:

Hardware and software requirements
-----------------------------------

The planner does not need Neuron hardware. It runs on any machine, including a laptop or a
CPU-only instance.

* **Python:** 3.12 (the skill was developed and tested with ``python3.12``).
* **Planner CLI:** Python standard library only.
* **Oracle validation:** ``torch`` (CPU build is enough; the oracle runs on the ``meta``
  device) and ``pytest`` for the tests.
* **Figures (optional):** ``matplotlib`` for the ``--visualize`` output.
* **Model config:** The model's unmodified HuggingFace ``config.json``. Model weights are
  not needed.

.. note::
   Hardware is needed only when you validate the planner's prediction. Use the
   ``neuron-framework-profiling-vllm-neuron`` and
   ``neuron-framework-profile-analysis-vllm-neuron`` skills on a Trainium instance for that
   step.

.. _planner-inputs:

Inputs: the run-spec
--------------------

Everything the planner computes is driven by a **run-spec**, a JSON file that points at the
model config and the hardware constants and states your serving targets. The skill ships no
default model. If you omit the run-spec, the planner stops with an error rather than silently
analyzing some other model.

A run-spec resolves ``hf_config`` and ``hardware`` relative to its own location, so keep the
three files in one directory.

.. code-block:: json

   {
     "model": "granite-4.1-8B (dense)",
     "hardware_name": "trn2",
     "hf_config": "granite4-8b-hf.json",
     "hardware": "trn2_constants.json",
     "serving": "colocated",
     "decode_itl_ceiling_ms": 50,
     "prefill_itps_floor": 30000
   }

Model, hardware, and SLA fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``model``
     - (required)
     - Display name printed in every report.
   * - ``hardware_name``
     - (required)
     - Short hardware label, for example ``trn2``.
   * - ``hf_config``
     - (required)
     - Path to the model's **pristine** HuggingFace ``config.json``. Do not edit it. Put all
       serving choices in the run-spec instead.
   * - ``hardware``
     - (required)
     - Path to a hardware-constants JSON file. The skill ships trn2 constants at
       ``references/trn2_constants.json``.
   * - ``serving``
     - ``colocated``
     - Serving mode. Only ``colocated`` is accepted; any other value raises an error.
   * - ``decode_itl_ceiling_ms``
     - 50
     - Maximum acceptable decode inter-token latency, in milliseconds.
   * - ``prefill_itps_floor``
     - 30000
     - Prefill SLA expressed as an input-tokens-per-second floor. The TTFT ceiling scales
       with context: ``ttft_ceiling(S) = S / floor``.
   * - ``prefill_ttft_ceiling_ms``
     - (unset)
     - Alternative prefill SLA: a flat TTFT ceiling in milliseconds at every context.
       Mutually exclusive with ``prefill_itps_floor``.

Physical serving fields
^^^^^^^^^^^^^^^^^^^^^^^

These fields change the system being modeled, not what the planner optimizes for. All are
optional.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``weight_dtype_bytes``
     - model dtype
     - Served weight element size (``1`` = FP8, ``2`` = BF16). Decode is weight-HBM-bound, so
       this is the largest single decode lever for a quantized checkpoint.
   * - ``lm_head_dtype_bytes``
     - ``weight_dtype_bytes``
     - Element size for a separate, **untied** lm_head. A tied model rejects a conflicting
       value.
   * - ``kv_dtype_bytes``
     - weight dtype
     - KV-cache element size, for example ``1`` for an FP8 KV cache.
   * - ``compute_precision``
     - ``bf16``
     - ``bf16`` or ``fp8``. ``fp8`` scales the tensor-engine peak by
       ``fp8_te_peak_multiplier`` (default 2.0), which speeds up compute-bound prefill.
   * - ``apply_prefill_dcp``
     - ``false``
     - Apply DCP attention sharding on the prefill pass.
   * - ``activation_io_policy``
     - ``component_boundary``
     - How activation HBM traffic is charged. See :ref:`planner-cost-model`.
   * - ``usable_hbm_fraction``
     - 0.90
     - Fraction of per-core HBM available for weights, KV cache, and activations. Lower it to
       model a tighter memory budget.

.. note::
   When you model an FP8 deployment, set ``weight_dtype_bytes: 1`` **and**
   ``compute_precision: "fp8"`` together. Halving weight bytes without modeling the FLOP
   speedup understates FP8 prefill by about 2×. Both factors are applied inside the engine.
   Do not scale the planner's output by hand.

Hardware constants
^^^^^^^^^^^^^^^^^^

The hardware-constants file supplies the numbers every planner result divides by. The shipped
trn2 file describes one LNC2 logical NeuronCore:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Constant
     - trn2 value
     - Meaning
   * - ``te_peak_flops``
     - 158 TFLOP/s
     - BF16 tensor-engine peak per logical core.
   * - ``hbm_bw_bytes_per_s``
     - 716 GB/s
     - HBM bandwidth per logical core.
   * - ``hbm_capacity_bytes_per_core``
     - 24 GiB
     - HBM capacity per logical core (96 GiB device ÷ 4 logical cores).
   * - ``cores_per_node``
     - 64
     - Logical cores per trn2.48xlarge at LNC2. This is the single-node cap.
   * - ``link_bw_within_node_bytes_per_s``
     - 1.28 TB/s
     - Link bandwidth used by the analytic collective model.

A stale or wrong constant produces a silently wrong plan. If a result looks implausible,
calibrate the constants by microbenchmarking. See the Hardware Entitlement skill's
``references/microbench-calibration.md``.

.. _planner-concepts:

Key concepts
------------

Entitlement and the roofline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each piece of work, the planner computes two lower bounds and takes the larger:

.. code-block:: text

   compute time = FLOPs / tensor-engine peak
   load time    = (weight bytes + KV bytes + activation bytes) / HBM bandwidth
   floor        = max(compute time, load time)

The ratio ``tensor-engine peak / HBM bandwidth`` is the *ridge point*, about 221 FLOP/byte on
trn2. Work whose arithmetic intensity (FLOPs per byte loaded) is below the ridge point is
memory-bound. Work above it is compute-bound. Small-batch decode sits far below the ridge
point. Whole-prompt prefill sits far above it.

.. _planner-cost-model:

The component cost model
^^^^^^^^^^^^^^^^^^^^^^^^

The planner does not apply a single roofline to the whole model, and it does not sum
per-operator latencies. It applies the roofline to each **component**: an attention body, an
FFN body, or the terminal lm_head. It then adds the component floors in sequence and adds the
collective time:

.. code-block:: text

   component_floor(c) = max(TE_work(c) / TE_peak, HBM_bytes(c) / HBM_bandwidth)
   step_time          = sum(component_floor(c)) + collective_time

Components are serialized because in a tensor-parallel transformer graph each attention and
FFN body ends at an all-reduce or reduce-scatter whose result the next body consumes. Inside
a component, the model assumes ideal fusion: internal operator interfaces stay on chip, and
only one gathered input read and one pre-collective output write are charged as activation
HBM traffic. This is the default ``component_boundary`` policy. Setting
``activation_io_policy`` to ``per_operator`` instead charges every operator's input and output,
which models an unfused or eager execution.

Collective time is modeled separately from the TE/HBM floor using a topology-aware ring model
(TP all-reduce, CP K/V ring). It is added after the component floors, not overlapped with them.

KV cache replication
^^^^^^^^^^^^^^^^^^^^

This is the most error-prone part of planning tensor parallelism for a GQA or MQA model:

- **The TP ceiling is the query-head count**, not the KV-head count. TP shards query heads,
  one per rank at most.
- **The KV cache does not shard past the KV-head count. It replicates.** Per-device KV bytes
  divide by ``min(TP, n_kv_heads)`` for each attention group. Past that point, extra ranks
  hold redundant copies of the cache.

For example, a model with 32 query heads and 8 KV heads can run at TP=32, but at TP=16 every
KV head is stored twice, and at TP=32 four times. That replication makes high-TP decode go
KV-read-bound early and wastes HBM that could hold more sequences or a prefix cache.

Context parallelism is the planner's remedy. A CP group is carved out of the TP group, and
during decode it splits the sequence dimension of the replicated KV cache across those
otherwise-redundant ranks. CP therefore recovers exactly the replication factor and no more:
the effective CP for a group is ``min(CP, TP / n_kv_heads)``. See :ref:`planner-axes`.

Oracle validation
^^^^^^^^^^^^^^^^^

The analytic engine is cross-checked against a **meta-device oracle**: a hand-written,
shape-exact ``torch`` module built from the same parsed architecture and run on the ``meta``
device under a FLOP counter. Because it runs on ``meta``, it needs no accelerator and no
weights.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Quantity
     - Authority
   * - Projection, FFN, router, and lm_head FLOPs
     - Oracle (exact).
   * - Weight bytes
     - Oracle. The analytic ``corrected_weight_bytes`` must match to well under 1%.
   * - KV-cache bytes
     - Analytic. The oracle reports zero because SDPA decomposes on ``meta``.
   * - Prefill SDPA FLOPs
     - Analytic causal active-pair count. The oracle's raw dense count is reported separately
       and is expected to be higher.

.. warning::
   The oracle validates that the engine and the oracle agree, not that the architecture
   was parsed correctly. Both are built from the same parse, so a mis-parsed model can still
   read +0.0%. Sanity-check the corrected weight total against the model card's parameter
   count, and confirm the layer count matches ``num_hidden_layers``.

Serving objectives
^^^^^^^^^^^^^^^^^^

Selection is **gate-then-rank**. SLA ceilings decide which configs are valid, and a
throughput metric ranks the survivors.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Objective
     - Behavior
   * - **Maximize OTPS under SLAs** (default)
     - Keep configs that meet both the prefill TTFT ceiling and the decode ITL ceiling, then
       pick the highest decode output-tokens-per-second (OTPS) per node.
   * - **Minimize prefill TTFT**
     - Pick the lowest TTFT, ignoring decode throughput. Favors configs with more cores,
       because prefill parallelizes almost perfectly.
   * - **Maximize decode OTPS**
     - Pick the highest per-node OTPS without the ITL ceiling. Favors large-batch configs.

Per-node throughput is ``otps × cores_per_node / cores``. This assumes the node is filled
with independent data-parallel replicas of the chosen config, so a small replica that holds
the weights scores well even though each replica uses few cores.

Throughput-capacity versus interactivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sweep evaluates every config at its **maximum feasible batch**, the largest batch whose
peak HBM fits the usable budget. Those cells answer the throughput-capacity question: how
much throughput can this footprint hold? They favor low TP with DP scale-out.

Low-ITL operating points live **below** maximum batch and favor higher TP. The
``--visualize`` per-node Pareto figure sweeps batch size below the maximum for each config
and draws the ITL versus per-node-throughput frontier. Use that figure, not the max-batch
table alone, when interactivity matters.

.. _planner-axes:

Supported parallelism
---------------------

The planner sweeps **TP, CP, DP, and EP** on a single node. **PP is not supported.** A
config's core count is ``world = TP × DP``. CP is carved out of TP and adds no cores.

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - Axis
     - Status
     - Behavior and restrictions
   * - **TP**
     - Supported, swept
     - * Candidate degrees: 4, 6, 8, 12, 16, 24, 32, and 48. TP=1 and TP=2 are not in the
         grid.
       * A degree is kept only if it does not exceed the query-head count, and divides
         exactly: every attention group's query-head count, ``vocab_size``, the dense FFN
         intermediate size, and the MoE expert intermediate size. A TP that needs padding is
         never recommended.
       * Divides compute and weights by TP. Divides the KV cache by ``min(TP, n_kv_heads)``.
       * Adds a per-layer all-reduce whose cost grows with TP.
       * Sequence parallelism is folded into TP for prefill. It is not a separate axis.
   * - **CP**
     - Supported, swept
     - * Candidate degrees: 1, 2, 4, and 8. CP must divide TP (``TP % CP == 0``).
       * **Decode:** shards the KV cache of full-context layers by
         ``min(CP, TP / n_kv_heads)``, which only recovers the cache TP replicated past
         ``n_kv_heads``. At ``TP ≤ n_kv_heads``, CP has no effect. For MLA, the limit is TP.
       * **Sliding-window layers:** never CP-sharded, because their cache is a constant
         window.
       * **Prefill:** no effect by default. With ``"apply_prefill_dcp": true`` in the run-spec,
         CP also splits prefill attention work and the attention input across the CP ranks.
       * Adds a K/V ring collective.
   * - **DP**
     - Supported
     - * Candidate degrees: 1, 2, 4, 8, and 16, subject to ``TP × DP ≤ cores_per_node``.
       * **Dense models:** not swept. DP is pinned to 1 in the grid, and the node is assumed
         to be filled with independent replicas. That is the per-node throughput metric.
         Replicas share no communication.
       * **MoE models with EP enabled:** swept, because DP widens the expert-sharding grid.
   * - **EP**
     - Supported, MoE only
     - * Turned on automatically when the model has routed experts. For dense models it
         does nothing.
       * Full EP over the whole world: the EP degree always equals ``TP × DP``, and each
         rank holds ``num_experts / (TP × DP)`` whole experts. The planner does not sweep a
         separate EP degree or a hybrid split that also tensor-shards each expert.
       * Only routed experts are EP-sharded. Attention, the always-on dense MLP, shared
         experts, and the embedding and lm_head stay sharded by TP. The router is replicated
         on every rank.
       * Communication is modeled as a wider FFN all-reduce over the expert grid. There is
         no all-to-all token dispatch.
       * Pass ``--disable-ep`` to plan a MoE model without EP.
   * - **PP**
     - **Not supported**
     - * Not implemented by the Neuron inference stack, so the engine has no PP axis and the
         grid has no PP degree.
       * The Hardware Entitlement scripts accept a ``--pp`` option only for compatibility.
         Any value other than 1 raises an error.

Other topology restrictions:

- **Single node only.** ``TP × DP`` can never exceed ``cores_per_node``, which is 64 logical
  cores on trn2.48xlarge at LNC2. Multi-node and multi-instance configs are never produced.
- **Colocated only.** One replica serves both prefill and decode. Disaggregated prefill and
  decode is not modeled.
- **Weights must fit.** A config whose per-device weights alone exceed usable HBM is dropped
  from the grid.
- **Speculative decoding is not a parallelism axis.** It is modeled separately; see
  :ref:`planner-tools-reference`.

The candidate degrees (``TP_CHOICES``, ``CP_CHOICES``, ``DP_CHOICES``), the number of context
buckets (``N_BUCKETS``), and batch granularity are constants in ``scripts/planner/plan.py``.
Edit them only to retune the search. To exclude an axis for one run, use the
``--disable-tp``, ``--disable-cp``, ``--disable-dp``, or ``--disable-ep`` flags, which pin that
axis to 1 (EP: off).

.. _planner-workflow:

Workflow
--------

The examples below run from your project directory with the skill's ``scripts/`` directory on
``PYTHONPATH``. Do not ``cd`` into the skill. The scripts refuse to write artifacts inside the
skill package.

.. code-block:: bash

   # ~/.claude for Claude Code, ~/.kiro for Kiro
   export NEURON_AGENTIC_DEVELOPMENT_ROOT=~/.claude
   SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/scripts

.. _planner-stage1:

Stage 1: Write the run-spec
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a directory containing the model's pristine ``config.json``, a copy of the hardware
constants, and a run-spec that points at both. The skill's
``samples/granite4-8b-walkthrough.md`` contains a complete, known-good example you can copy.

Before trusting results for a new model, check the parse:

1. Confirm the config's field names are ones ``arch_from_hf`` recognizes. For example, MoE
   configs that use ``moe_num_experts`` instead of ``n_routed_experts`` have previously been
   modeled as dense.
2. Confirm every ``layer_types`` value is ``sliding_attention`` or ``full_attention``, and
   that the parsed layer count equals ``num_hidden_layers``. Other values, such as
   ``linear_attention``, are silently mis-modeled or dropped.
3. If the architecture is not in :ref:`planner-supported-architectures` (for example, a new
   attention variant or recurrent linear attention), extend the engine and oracle first. See
   ``references/oracle-construction.md``.

.. _planner-stage2:

Stage 2: Validate the engine against the oracle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3.12 $SKILL/validate_oracle.py ./my-model/my-model-run.json

The script prints, per segment, the oracle and analytic FLOPs and weight bytes for a decode
cell and a prefill cell.

**Pass criteria:**

- Weight bytes match to well under 1% on every segment.
- Decode FLOPs match within a few percent.
- Prefill attention FLOPs may differ. The oracle counts the dense score rectangle, while the
  engine charges causal active pairs. Projection, FFN, and lm_head prefill FLOPs should still
  match.

A large gap localizes a missing term to the segment where it appears. Fix it before sweeping.

.. _planner-stage3:

Stage 3: Sweep and select
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./my-model/my-model-run.json

The planner:

1. Builds the config grid, rejecting configs that do not shard evenly, whose per-rank
   weights exceed usable HBM, or whose ``TP × DP`` exceeds ``cores_per_node``.
2. Derives the context buckets from the model's ``max_position_embeddings``. It keeps six
   buckets, pinning the full context as the last one.
3. Solves each cell's maximum feasible batch in closed form. Peak HBM is affine in batch
   size, and the result is rounded down to an even number.
4. Costs every cell for decode and prefill, gates on the SLAs, and selects a config per
   bucket.

For each bucket, the output prints the selected config and the full decode landscape:

.. code-block:: text

   --- 8k ---
     SELECTED TP8            cores=8   TTFT=0.22s ITL=35ms sla_met=True tput/node=28089
         TP4            cores=4   bs=56   ITL=  33.5ms tput/node=    26768 bind=attention:kv_load
         TP8            cores=8   bs=124  ITL=  35.3ms tput/node=    28089 bind=attention:kv_load
         TP16           cores=16  bs=130  ITL=  37.4ms tput/node=    13895 bind=attention:kv_load
         ...

``bind`` names the component and term that dominates the step: ``kv_load``, ``weight_load``,
``activation_load``, ``matmul``, or ``comm``. When no config meets both SLAs, the planner falls
back to the lowest-TTFT config among those that meet the ITL ceiling, or the lowest-TTFT config
overall if none do, and marks it ``sla_met=False``. That is a documented miss, not an error.

Useful options:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Effect
   * - ``--decode-itl-ceiling-ms N``
     - Override the run-spec's ITL ceiling.
   * - ``--prefill-itps-floor N`` / ``--prefill-ttft-ceiling-ms N``
     - Override the prefill SLA (mutually exclusive).
   * - ``--usable-fraction F``
     - Override the usable HBM fraction. Precedence: CLI flag, then run-spec, then hardware
       JSON, then 0.90.
   * - ``--disable-<axis>`` / ``--enable-<axis>``
     - Pin an axis (``tp``, ``cp``, ``dp``, ``ep``) to degree 1, or allow it.
   * - ``--emit``
     - Write ``target_config.json``, ``pareto_frontier.json``, and ``entitlement.json``.
   * - ``--visualize``
     - Write the roofline, Pareto, per-node Pareto, and prefill-footprint figures (PNG).
   * - ``--project-root DIR``
     - Write artifacts under ``DIR/agent_artifacts/tmp/`` instead of the current directory.

.. _planner-stage4:

Stage 4: Inspect a single cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compare against one on-hardware run, evaluate the exact operating point you plan to
deploy:

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./my-model/my-model-run.json \
     --cell --tp 8 --cp 1 --bs 16 --ctx 8192

``--cell`` skips the sweep and prints a per-(component, layer type) breakdown: layer count,
weight, KV, and activation megabytes, FLOPs, load time, compute time, the component floor, and
its binding term. Add ``--prefill`` to evaluate the prefill regime instead of decode.

The cell is the hypothesis a profile confirms or refutes. Hand it to
``neuron-framework-profiling-vllm-neuron`` to capture a trace at the same operating point,
then to ``neuron-framework-profile-analysis-vllm-neuron`` to compare measured per-segment
time against the planner's expected time. For a per-operator breakdown of the same cell, use
the :ref:`Hardware Entitlement skill <neuron-hardware-entitlement>`.

.. _planner-file-organization:

File organization
-----------------

The planner writes every artifact under ``<project root>/agent_artifacts/tmp/``. The project
root is the current directory unless you set ``--project-root`` or ``$NEURON_PROJECT_ROOT``.
The Hardware Entitlement skill honors the same variable, so one analysis keeps one evidence
trail.

.. code-block:: text

   <project root>/agent_artifacts/tmp/
   ├── sweep_results.json                   # Every decode and prefill cell, plus the batch sweep
   ├── target_config.json                   # Selected config per bucket (--emit)
   ├── pareto_frontier.json                 # Per-bucket frontier (--emit)
   ├── entitlement.json                     # Weight totals and minimum footprint (--emit)
   ├── roofline_<model>.png                 # Decode cells on the TE/HBM roofline (--visualize)
   ├── pareto_<model>.png                   # OTPS vs ITL, area = cores (--visualize)
   ├── pareto_per_node_<model>.png          # Interactivity Pareto, DP-normalized (--visualize)
   └── prefill_footprint_<model>.png        # Prefill TTFT vs footprint with SLA (--visualize)

.. _planner-tools-reference:

Tools reference
---------------

Planner CLI
^^^^^^^^^^^

**Module:** ``scripts/planner/__main__.py`` (``python3.12 -m planner``)

Runs the full sweep and selection for a run-spec, or evaluates a single cell with ``--cell``.

Engine package
^^^^^^^^^^^^^^

**Package:** ``scripts/planner/``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Role
   * - ``segments.py``
     - Aggregates the model into per-(class, layer type) weight, KV, and FLOP segments.
   * - ``shard.py``
     - Owns every TP, CP, DP, and EP divisor rule, including KV replication.
   * - ``activation.py``
     - Builds the ordered component graph and the activation boundary tensors.
   * - ``cost.py``
     - Applies the per-component roofline and serializes components into a step time.
   * - ``plan.py``
     - Builds the grid, solves maximum feasible batch, selects buckets, and applies
       gate-then-rank selection.
   * - ``build.py``
     - Loads the run-spec. ``cell(...)`` evaluates one (config, bucket, batch, regime) point.
   * - ``emit.py`` / ``visualize.py``
     - Write the JSON artifacts and PNG figures.

Model and hardware layer
^^^^^^^^^^^^^^^^^^^^^^^^

**Module:** ``scripts/entitlement_formulas.py``

Parses the HF config (``arch_from_hf``) and hardware constants, and provides the accounting
primitives and the topology-aware collective model the engine builds on.

Oracle and validation
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Role
   * - ``scripts/oracle_model.py``
     - The shape-exact ``torch`` oracle, built from the same parsed architecture.
   * - ``scripts/entitlement_counter.py``
     - FLOP, weight, and KV counter (subclasses ``torch.utils.flop_counter``) and the
       speculative-decoding cost model.
   * - ``scripts/validate_oracle.py``
     - Diffs the engine's analytic totals against the oracle per segment.

Speculative decoding model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Speculative decoding is not a sharding axis. It changes the cost per accepted token. The model
in ``entitlement_counter.py`` takes the draft length γ and the acceptance rate α:

.. code-block:: text

   expected accepted tokens  N = (1 - α^(γ+1)) / (1 - α)
   weight-read speedup         = N / (1 + γ · draft_weight / target_weight)

The planner infers nothing about the draft model. You must supply its weight bytes, KV bytes,
and FLOPs explicitly, either directly, from a measured report, or from the draft's HF config.

.. _planner-debugging:

Common issues
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Solution
   * - ``error: a run-spec path is required``
     - Pass a path to a run-spec JSON. There is no default model.
   * - ``error: run-spec not found``
     - The argument must be a file path. Model names are not resolved.
   * - Error when ``serving`` is not ``colocated``
     - Only colocated, single-node serving is supported. Remove the field or set it to
       ``colocated``.
   * - Scripts refuse to write artifacts
     - You are running from inside the skill package. Run from your project directory, or pass
       ``--project-root``.
   * - ``validate_oracle.py`` weight gap above 1%
     - A weight term is missing or double counted. Check tied embeddings, MoE expert counts,
       and the ``attention_k_eq_v`` flag.
   * - Model planned as dense but it is MoE
     - The config uses MoE field names ``arch_from_hf`` does not recognize. Extend the parser.
   * - Parsed layer count is lower than ``num_hidden_layers``, or a hybrid model plans like a
       pure-attention model
     - ``layer_types`` contains a value other than ``sliding_attention`` or
       ``full_attention``, such as ``linear_attention``. The architecture is not supported.
       See :ref:`planner-supported-architectures`.
   * - Every config shows ``sla_met=False``
     - No colocated config meets both SLAs at maximum batch. Check the per-node Pareto figure
       for lower-batch points that meet the ITL ceiling, or relax an SLA.
   * - CP rows look identical to their TP row
     - Expected when ``TP ≤ n_kv_heads``. CP only reclaims KV replicated past the KV-head
       count, so it has no effect there. It also has no effect on sliding-window layers, or
       on prefill unless ``apply_prefill_dcp`` is set.

.. _planner-guidelines:

Project guidelines
------------------

The Parallelism Planner skill enforces these rules:

- **Single-node, colocated serving only.** Never propose disaggregated prefill and decode,
  split pools, or multi-node topologies, even when no colocated config meets the SLA.
- **Always pass an explicit run-spec path.** No environment variable or default selects a
  model.
- **Keep the HF config pristine.** Put precision, KV dtype, and memory budget in the
  run-spec.
- **Validate against the oracle before trusting a new model.**
- **Report the full TP sweep, not just the winner.** For collective-bound models, the
  latency optimum is often an interior TP degree, not the maximum.
- **Treat every result as a prediction** until it is profiled on hardware.

.. _planner-knowledge-base:

Knowledge base
--------------

The skill's ``references/`` and ``samples/`` directories contain:

- ``methodology.md``: The full sweep methodology, the findings that drive config selection,
  the two serving objectives, prefill, and the prefix-cache tradeoff.
- ``parallelism-cost-model.md``: How each parallelism axis scales compute, weight, KV, and
  communication cost, and the speculative-decoding cost model.
- ``tp-feasibility-and-kv-replication.md``: Why the TP ceiling is the query-head count and
  how KV replication works. Read before trusting any high-TP number.
- ``oracle-construction.md``: How the oracle is built, and how to extend it for a new
  architecture.
- ``design.md``: The engine's design: aggregation, sharding, the cost model, and selection.
- ``trn2_constants.json``: The shipped trn2 hardware constants.
- ``samples/granite4-8b-walkthrough.md``: An end-to-end walkthrough for Granite-4.1-8B on trn2,
  including oracle validation, the sweep, a cell breakdown, and a comparison against measured
  hardware.

Related resources
-----------------

- :ref:`neuron-hardware-entitlement`: Per-operator floors for a chosen config.
- :ref:`parallelism-planner-tutorial`: Step-by-step tutorial for this skill.
- :ref:`nxdi-parallelism-user-guide`: Tensor and sequence parallelism in NxD Inference.
- :ref:`nxdi-vllm-user-guide-v1`: Deploying models with vLLM on Neuron.
- `Neuron Agentic Development <https://github.com/aws-neuron/neuron-agentic-development>`_: Source for the Parallelism Planner skill.
