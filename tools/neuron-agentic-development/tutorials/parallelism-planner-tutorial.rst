.. meta::
   :description: Tutorial for planning tensor, context, and data parallelism for LLM inference on Trainium using the Neuron Agentic Development Parallelism Planner skill.
   :keywords: parallelism planner, tutorial, tensor parallelism, context parallelism, LLM inference, Neuron Agentic Development, Trainium, ITL, TTFT
   :date-modified: 2026-10-07

.. _parallelism-planner-tutorial:

Tutorial: Plan a serving config using the Parallelism Planner skill
====================================================================

This tutorial walks you through choosing a parallelism config for serving Granite-4.1-8B on a
trn2.48xlarge instance with the Parallelism Planner agent. You write a run-spec that describes
the model, the hardware, and your latency targets. The agent validates its analytic engine
against an oracle, sweeps every candidate config, and recommends one. You then inspect the
recommended config in detail.

By the end of this tutorial you will have a recommended tensor-parallel and data-parallel
config with predicted decode ITL, prefill TTFT, per-node throughput, and the binding
constraint, ready to validate on hardware.

Prerequisites
-------------

Choose a machine
^^^^^^^^^^^^^^^^

The planner is an analytic model. It does not need Neuron hardware, model weights, or a
compiled model. You can run this tutorial on a laptop, a CPU-only instance, or the Trainium
instance you will later deploy on.

Install Neuron Agentic Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have not already installed the package, follow the
:ref:`Getting Started guide <neuron-agentic-development-getting-started>`.

Make sure the deploy step completed.

.. code-block:: bash

   # For Claude Code
   deploy-neuron-agentic-development-to-claude

   # For Kiro
   deploy-neuron-agentic-development-to-kiro

Set up Python
^^^^^^^^^^^^^

The planner CLI needs only the Python 3.12 standard library. Oracle validation needs
``torch``. The CPU build is enough because the oracle runs on the ``meta`` device. The
optional figures need ``matplotlib``.

.. code-block:: bash

   python3.12 -m venv ~/planner-venv
   source ~/planner-venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install matplotlib

Point at the skill
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # ~/.claude for Claude Code, ~/.kiro for Kiro
   export NEURON_AGENTIC_DEVELOPMENT_ROOT=~/.claude
   SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/scripts

Step 1. Write the run-spec
--------------------------

The run-spec is the planner's only input. It names three files that must sit in the same
directory, because paths resolve relative to the run-spec.

Create a project directory and a model directory inside it.

.. code-block:: bash

   mkdir -p ~/planner-tutorial/granite
   cd ~/planner-tutorial

**Hardware constants.** Copy the trn2 constants that ship with the skill.

.. code-block:: bash

   cp $NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-parallelism-planner/references/trn2_constants.json granite/

**Model config.** Save the model's unmodified HuggingFace ``config.json`` as
``granite/granite4-8b-hf.json``. For Granite-4.1-8B, the full config is also written out at the
end of the skill's ``samples/granite4-8b-walkthrough.md``. The fields the planner uses are:

.. code-block:: json

   {
     "architectures": ["GraniteForCausalLM"],
     "hidden_size": 4096,
     "intermediate_size": 12800,
     "num_hidden_layers": 40,
     "num_attention_heads": 32,
     "num_key_value_heads": 8,
     "vocab_size": 100352,
     "max_position_embeddings": 131072,
     "tie_word_embeddings": true,
     "torch_dtype": "bfloat16"
   }

Granite is a dense GQA model, which the planner supports. If you adapt this tutorial to your
own model, first check that its architecture is listed in
:ref:`planner-supported-architectures`. Linear-attention, recurrent, and state-space layers
are not supported, and the parser does not reject them. Such a model would be silently
mis-modeled.

**Run-spec.** Save the following as ``granite/granite4-8b-run.json``.

.. code-block:: json

   {
     "model": "granite-4.1-8B (dense)",
     "hardware_name": "trn2",
     "hf_config": "granite4-8b-hf.json",
     "hardware": "trn2_constants.json",
     "serving": "colocated",
     "decode_itl_ceiling_ms": 50,
     "prefill_itps_floor": 30000,
     "configs": [
       {"label": "TP=8", "regime": "decode_heavy", "TP": 8, "PP": 1, "CP": 1,
        "bucket": 8192, "bs": 8}
     ]
   }

The two SLA fields say: every decode step must finish within 50 ms, and prefill must process at
least 30,000 input tokens per second, which is a 0.27 s TTFT ceiling at 8k context. The
``configs`` entry is optional for the planner, but the Hardware Entitlement tutorial reuses this
run-spec and reads it.

.. note::
   The planner plans **single-node, colocated** serving only: one replica serves both prefill
   and decode, within one trn2.48xlarge (64 logical NeuronCores at LNC2). It will not propose
   disaggregated or multi-node configs.

Step 2. Invoke the Parallelism Planner agent
--------------------------------------------

Open your agentic IDE in ``~/planner-tutorial``.

For Kiro, start the agent directly.

.. code-block:: bash

   kiro-cli chat --agent neuron-parallelism-planner-agent

For Claude Code, start ``claude`` in the project directory. The agent is selected automatically
from your request.

Ask for a plan.

.. code-block:: text

   Plan a serving config for Granite-4.1-8B on trn2.48xlarge using the run-spec at
   granite/granite4-8b-run.json. I need decode ITL under 50 ms and at least 30,000
   input tokens/s for prefill. Maximize throughput per node.

The agent confirms the run-spec and the SLAs, then works through the steps below.

Step 3. What happens during planning
------------------------------------

**Architecture parse.** The agent parses the HF config into a model description. Granite parses
as 40 layers of full attention with 32 query heads and 8 KV heads (head_dim 128), a dense FFN,
and tied embeddings. That is 16.76 GB of BF16 weights. The agent checks this against the model
card's 8B parameter count.

**Oracle validation.** The agent runs ``validate_oracle.py``, which compares the analytic
engine's FLOP and weight counts against a shape-exact ``torch`` oracle on the ``meta`` device.

.. code-block:: bash

   python3.12 $SKILL/validate_oracle.py ./granite/granite4-8b-run.json

.. code-block:: text

   === decode S4096 bs8 ===
   segment                     oracle FLOP    analytic FLOP      d%    oracle wt(MB)  analytic wt(MB)      d%
   attention/global         48,318,382,080   48,318,382,080   +0.0%           3355.4           3355.4   +0.0%
   ffn/dense               100,663,296,000  100,663,296,000   +0.0%          12582.9          12582.9   +0.0%
   lm_head                   6,576,668,672    6,576,668,672   +0.0%            822.1            822.1   +0.0%
   TOTAL                   155,558,346,752  155,558,346,752   +0.0%          16760.4          16760.4   +0.0%

   === prefill S2048 bs1 ===
   attention/global       9,620,726,743,040 8,247,008,296,960  +16.7%           3355.4           3355.4   +0.0%
   ...

Weights and decode FLOPs match exactly. The +16.7% on prefill attention is expected: the oracle
counts the full score rectangle, while the engine charges only the causal (lower-triangle)
pairs.

**Sweep.** The agent sweeps 15 TP and CP configs across six context buckets, from 8k up to the
model's 128k maximum. Each config runs at its maximum feasible batch, the largest batch whose
weights, KV cache, and activations fit in 90% of HBM.

**Selection.** For each bucket, the agent keeps the configs that meet both SLAs and picks the
one with the highest throughput per node, assuming the node is filled with independent replicas.

Step 4. Read the recommendation
-------------------------------

You can run the same sweep yourself.

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./granite/granite4-8b-run.json

The 8k bucket looks like this (CP rows trimmed):

.. code-block:: text

   === granite-4.1-8B (dense) on trn2 [serving=colocated single-node, TTFT floor=30000 ITPS, ITL ceiling=50 ms, usable HBM=90%, weights=2B/param, compute=bf16] ===
   grid=15 configs, buckets=[8, 16, 28, 44, 76, 128]k

   --- 8k ---
     SELECTED TP8            cores=8   TTFT=0.22s ITL=35ms sla_met=True tput/node=28089
         TP4            cores=4   bs=56   ITL=  33.5ms tput/node=    26768 bind=attention:kv_load
         TP8            cores=8   bs=124  ITL=  35.3ms tput/node=    28089 bind=attention:kv_load
         TP16           cores=16  bs=130  ITL=  37.4ms tput/node=    13895 bind=attention:kv_load
         TP16 CP2       cores=16  bs=262  ITL=  39.5ms tput/node=    26500 bind=attention:kv_load
         TP32           cores=32  bs=134  ITL=  41.6ms tput/node=     6441 bind=attention:kv_load
         TP32 CP4       cores=32  bs=538  ITL=  48.0ms tput/node=    22420 bind=attention:kv_load

How to read it:

- **The recommendation is TP=8, deployed as 8 data-parallel replicas** across the node's 64
  cores. Each replica runs a batch of up to 124 at about 35 ms ITL and 0.22 s TTFT, and the
  node delivers about 28,000 output tokens per second.
- **Throughput per node falls sharply past TP=8.** Granite has 8 KV heads. At TP=16, each
  KV head is stored on two ranks, and at TP=32 on four. The extra cores hold redundant KV
  instead of buying batch, so per-node throughput halves at each step.
- **CP recovers that loss, and only that loss.** At TP16, CP2 splits the duplicated KV
  cache's sequence across the two ranks that would otherwise each hold a full copy. That
  doubles the batch the replica can hold compared to plain TP16. CP4 and CP8 add nothing more,
  because there is only 2× replication to reclaim. At TP8, the CP rows are within 0.1% of
  plain TP8, because nothing is replicated.
- **Every cell binds on attention KV reads** (``attention:kv_load``) at maximum batch: the
  batch-scaled KV read dominates the decode step.
- **ITL grows with TP at maximum batch**, because a larger footprint holds a larger batch.
  This table answers the throughput-capacity question, not the lowest-latency one.

Further down, the selection changes as context grows. At 44k, the TTFT ceiling is
45,056 ÷ 30,000 = 1.50 s. TP8 needs 1.61 s for a 44k prompt, so it fails the gate, and the pick
moves to TP16 CP2 (1.12 s).

Step 5. Inspect the recommended config
--------------------------------------

Before you deploy, look at one exact operating point. Batch 16 at 8k context is a conservative
first point to compare with hardware.

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./granite/granite4-8b-run.json \
     --cell --tp 8 --cp 1 --bs 16 --ctx 8192

.. code-block:: text

   === granite-4.1-8B (dense) on trn2 | cell TP8 CP1 DP1 bs16 ctx8k regime=decode ===
   cores(world)=8  step_ms=8.569  comm_ms=1.864  binding=attention:kv_load
   decode otps=1867/replica  tput/node(x8)=14938

   component/type                       layers       W_MB      KV_MB     Act_MB    FLOP_G  load_ms  flop_ms  time_ms  bind
   attention/global                         40     419.43     2684.4       10.5    17.448    4.350    0.110    4.350  kv_load
   ffn/dense                                40    1572.86        0.0       10.5    25.166    2.211    0.159    2.211  weight_load
   lm_head                                   1     102.76        0.0        0.5     1.644    0.144    0.010    0.144  weight_load
   model total                                    2095.05     2684.4       21.5    44.258

The predicted decode step is ``4.350 + 2.211 + 0.144 = 6.705 ms`` of component floors plus
1.864 ms of collectives, for 8.569 ms in total. Attention is KV-read-bound, the FFN is
weight-read-bound, and every component's compute time is tiny by comparison. A faster matmul
kernel would not move this config. Fewer HBM bytes would, for example an FP8 KV cache or FP8
weights.

Step 6. Explore the latency and throughput tradeoff (optional)
--------------------------------------------------------------

**Tighten the ITL ceiling.** Rerun with a 30 ms ITL ceiling.

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./granite/granite4-8b-run.json --decode-itl-ceiling-ms 30

.. code-block:: text

   --- 8k ---
     SELECTED TP32           cores=32  TTFT=0.14s ITL=42ms sla_met=False tput/node=6441

No config meets 30 ms **at maximum batch**, so the planner falls back to the config with the
lowest TTFT and marks it ``sla_met=False``. That does not mean 30 ms is unreachable. It means
you have to run below maximum batch.

**Draw the interactivity frontier.** ``--visualize`` sweeps batch size below the maximum for
every config and writes figures.

.. code-block:: bash

   PYTHONPATH=$SKILL python3.12 -m planner ./granite/granite4-8b-run.json --visualize

Open ``agent_artifacts/tmp/pareto_per_node_granite_4_1_8b_dense.png``. Each curve is one config
swept over batch size. The bold frontier is the best per-node throughput available at each ITL.
For Granite at 8k, the frontier is all TP8. Moving along it from low batch to high batch trades
lower ITL for higher throughput. Pick the point on the frontier that meets your ITL target.

**Model FP8.** Copy the run-spec, add ``"weight_dtype_bytes": 1`` and
``"compute_precision": "fp8"``, and rerun to compare against the BF16 baseline.

Step 7. Check the results
-------------------------

.. code-block:: text

   ~/planner-tutorial/
   ├── granite/
   │   ├── granite4-8b-run.json                     # Your run-spec
   │   ├── granite4-8b-hf.json                      # Pristine HF config
   │   └── trn2_constants.json                      # Hardware constants
   └── agent_artifacts/tmp/
       ├── sweep_results.json                       # Every decode and prefill cell
       ├── roofline_granite_4_1_8b_dense.png        # --visualize
       ├── pareto_granite_4_1_8b_dense.png          # --visualize
       ├── pareto_per_node_granite_4_1_8b_dense.png # --visualize: interactivity frontier
       └── prefill_footprint_granite_4_1_8b_dense.png

Add ``--emit`` to also write ``target_config.json`` (the selected config per bucket),
``pareto_frontier.json``, and ``entitlement.json``.

Next steps
----------

The recommendation is a prediction. To validate it:

1. Run :ref:`hardware-entitlement-tutorial` on the same run-spec to get per-operator floors for
   the chosen config.
2. Deploy TP=8 with vLLM on a trn2.48xlarge, then profile it with the
   ``neuron-framework-profiling-vllm-neuron`` skill.
3. Compare measured per-segment time against the ``--cell`` breakdown with the
   ``neuron-framework-profile-analysis-vllm-neuron`` skill.

Troubleshooting
---------------

**The agent asks for missing SLAs.** The default objective needs both an ITL ceiling and a
prefill SLA. Provide them, or say which single metric you care about: lowest TTFT, or highest
decode throughput with no latency limit.

**"error: a run-spec path is required" or "run-spec not found".** Pass the path to the run-spec
file. The planner has no default model and does not look up models by name.

**"refusing to write generated artifacts inside the skill package".** You are running from inside
the skill directory. ``cd`` to your project directory, or pass ``--project-root``.

**ModuleNotFoundError: torch.** Only ``validate_oracle.py`` needs ``torch``. Install the CPU
build into the interpreter you use.

**Weight bytes in validate_oracle.py differ by more than 1%.** The architecture was not parsed
as expected. Check tied embeddings, MoE field names, and that the parsed layer count matches
``num_hidden_layers``.

**The agent proposes nothing that meets my SLA.** The planner never falls back to disaggregated
or multi-node serving. It reports the best single-node config and the gap. Relax an SLA, use the
per-node Pareto figure to pick a lower batch, or model FP8.

Related resources
-----------------

- :doc:`/tools/neuron-agentic-development/developer_guides/neuron-parallelism-planner` for the deep dive on how the Parallelism Planner skill works internally.
- :doc:`/tools/neuron-agentic-development/tutorials/hardware-entitlement-tutorial` for per-operator floors of the config you chose.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- :ref:`nxdi-vllm-user-guide-v1` for deploying the chosen config with vLLM.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_ for source code and issue tracker.
