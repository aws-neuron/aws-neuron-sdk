.. meta::
   :description: Tutorial for computing per-operator compute and HBM floors for an LLM inference config on Trainium using the Neuron Agentic Development Hardware Entitlement skill.
   :keywords: hardware entitlement, tutorial, operator floor, roofline, memory-bound, compute-bound, Neuron Agentic Development, Trainium, optimization
   :date-modified: 2026-10-07

.. _hardware-entitlement-tutorial:

Tutorial: Find the binding operator using the Hardware Entitlement skill
=========================================================================

This tutorial walks you through computing the hardware-entitlement floor for Granite-4.1-8B at
TP=8 on trn2 with the Hardware Entitlement agent. TP=8 is the config the
:ref:`Parallelism Planner tutorial <parallelism-planner-tutorial>` recommends. You give the
agent the run-spec and the operating point. The agent computes the minimum time every operator
must pay on the tensor engine and on HBM, identifies which operator and which ceiling binds,
and shows how the binding shifts with batch size, context length, and phase.

By the end of this tutorial you will know which operator to optimize first for this config,
and whether it is limited by compute, weight reads, or KV-cache reads.

Prerequisites
-------------

Complete the Parallelism Planner tutorial setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial reuses the project directory, the Python environment, and the run-spec from the
:ref:`Parallelism Planner tutorial <parallelism-planner-tutorial>`. Complete its Prerequisites
and Step 1 first. You need:

.. code-block:: text

   ~/planner-tutorial/granite/
   ├── granite4-8b-run.json
   ├── granite4-8b-hf.json
   └── trn2_constants.json

This skill supports the same architectures as the planner: decoder-only softmax-attention
transformers, dense or MoE. Linear-attention, recurrent, and state-space layers are not
supported. See :ref:`entitlement-supported-architectures`.

The floor scripts need ``torch``. The CPU build is enough. Like the planner, they do not need
Neuron hardware. Only the optional HBM attribution step in Step 7 needs a compiled model from a
Trainium instance.

Point at the skill
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd ~/planner-tutorial
   source ~/planner-venv/bin/activate
   # ~/.claude for Claude Code, ~/.kiro for Kiro
   export NEURON_AGENTIC_DEVELOPMENT_ROOT=~/.claude
   SKILL=$NEURON_AGENTIC_DEVELOPMENT_ROOT/skills/neuron-hardware-entitlement/scripts

The Hardware Entitlement skill imports the Parallelism Planner's engine. Both skills must be
deployed side by side, which the deploy step does for you.

Step 1. Choose the operating point
----------------------------------

The skill analyzes one already-chosen config. A config cell is
``{TP, PP, CP, bucket, bs, prefill}``.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Value
     - Why
   * - TP / PP / CP
     - 8 / 1 / 1
     - The planner's recommendation at 8k context.
   * - Bucket (context)
     - 8192
     - The 8k bucket.
   * - Batch size
     - 16
     - A conservative first point to profile on hardware.
   * - Phase
     - decode
     - The phase that sets ITL.

Step 2. Invoke the Hardware Entitlement agent
---------------------------------------------

Open your agentic IDE in ``~/planner-tutorial``.

For Kiro, start the agent directly.

.. code-block:: bash

   kiro-cli chat --agent neuron-hardware-entitlement-agent

For Claude Code, start ``claude`` in the project directory. The agent is selected automatically
from your request.

Ask for the floor.

.. code-block:: text

   Compute the per-operator hardware-entitlement floor for Granite-4.1-8B using the run-spec
   at granite/granite4-8b-run.json, at TP=8, 8k context, batch 16, decode. Which operator
   binds, and on what? Then show me how it changes at batch 64 and for an 8k prefill.

The agent confirms the run-spec and the cell, then works through the steps below.

Step 3. What happens during the analysis
----------------------------------------

**Engine check.** The agent confirms the planner's engine reconciles with the oracle for this
model. If you ran ``validate_oracle.py`` in the planner tutorial, this is already done.

**Per-operator floor.** For every operator in the model, the agent computes, per device:

- the **compute floor**: FLOPs ÷ tensor-engine peak (158 TFLOP/s per logical core on trn2),
- the **HBM floor**: weight and KV bytes ÷ HBM bandwidth (716 GB/s per logical core),
- the **arithmetic intensity**: FLOPs per byte, compared with the ridge point of about
  221 FLOP/byte, and
- the **binding**: ``compute`` or ``memory``.

**Reconciliation.** The agent sums the per-operator floors and checks that they match the
planner's totals for the same cell. A mismatch above about 1% stops the analysis.

**Component floors.** The agent groups operators into attention, FFN, and lm_head components,
applies one roofline to each, and adds them up. Collectives are reported separately.

Step 4. Read the decode floor
-----------------------------

You can run the same analysis yourself.

.. code-block:: bash

   python3.12 $SKILL/operator_entitlement.py --runspec ./granite/granite4-8b-run.json \
     --tp 8 --bucket 8192 --bs 16

.. code-block:: text

   === operator entitlement: decode TP8 PP1 CP1 EP1 bucket=8192 bs=16 (granite-4.1-8B (dense)) ===
   op           group    inst/dev   compute       HBM (    wt     kv)  act-iface      AI  MUST-bound
   sdpa         full         40.0    0.068m    3.749m ( 0.00m  3.75m)      1.2MiB     4.0  memory
   gate_proj    full         40.0    0.053m    0.732m ( 0.73m  0.00m)      7.0MiB    16.0  memory
   up_proj      full         40.0    0.053m    0.732m ( 0.73m  0.00m)      7.0MiB    16.0  memory
   down_proj    full         40.0    0.053m    0.732m ( 0.73m  0.00m)      7.0MiB    16.0  memory
   q_proj       full         40.0    0.017m    0.234m ( 0.23m  0.00m)      5.6MiB    16.0  memory
   o_proj       full         40.0    0.017m    0.234m ( 0.23m  0.00m)      5.6MiB    16.0  memory
   lm_head      head          1.0    0.010m    0.144m ( 0.14m  0.00m)      0.5MiB    16.0  memory
   k_proj       full         40.0    0.004m    0.059m ( 0.06m  0.00m)      5.2MiB    16.0  memory
   v_proj       full         40.0    0.004m    0.059m ( 0.06m  0.00m)      5.2MiB    16.0  memory

   reconciliation vs analytic cell (binding=kv_load):
     compute_vs_matmul        rel_err=0.0000%
     weight_vs_weight_load    rel_err=0.0000%
     kv_vs_kv_load            rel_err=0.0000%

   component composition (TE/HBM certified scope; collectives excluded):
     components=81 in 3 reported records (3 cost records)  TE=0.280 ms  HBM=6.705 ms
     global-engine diagnostic=6.705 ms  serialized component floor=6.705 ms

Each row is one operator summed over its 40 layer instances on one device. Times are in
milliseconds.

How to read it:

- **Every operator is memory-bound.** Projections sit at an arithmetic intensity of 16, which
  is the batch size: each weight byte is read once and used for 16 tokens. That is far below the
  ridge point of about 221.
- **Attention (``sdpa``) is the binding operator.** It owns no weights, so its HBM floor is all
  KV-cache reads: 3.75 ms, more than every other operator combined. Its arithmetic intensity is
  just 4.
- **Compute is negligible.** The whole model needs 0.28 ms of tensor-engine time against 6.7 ms
  of HBM traffic.
- **Reconciliation is 0.00%**, so the per-operator breakdown is exactly the planner's cell,
  opened up.
- **The component floor is 6.705 ms**, made of 81 serialized components: 40 attention bodies,
  40 FFN bodies, and one lm_head. Adding the planner's 1.86 ms of collectives gives the planner's
  8.57 ms decode step.

**What to optimize:** A faster matmul kernel would not help, because nothing here is
compute-bound. The lever is fewer HBM bytes, and the largest share is KV. An FP8 KV cache halves
the ``sdpa`` floor. To model it, copy the run-spec, add ``"kv_dtype_bytes": 1``, and rerun.

Step 5. See how the binding moves
---------------------------------

Larger batch
^^^^^^^^^^^^

.. code-block:: bash

   python3.12 $SKILL/operator_entitlement.py --runspec ./granite/granite4-8b-run.json \
     --tp 8 --bucket 8192 --bs 64

.. code-block:: text

   op           group    inst/dev   compute       HBM (    wt     kv)  act-iface      AI  MUST-bound
   sdpa         full         40.0    0.272m   14.996m ( 0.00m 15.00m)      5.0MiB     4.0  memory
   gate_proj    full         40.0    0.212m    0.732m ( 0.73m  0.00m)     27.8MiB    64.0  memory
   ...

Quadrupling the batch quadruples KV reads, from 3.75 ms to 15.00 ms, because every sequence has
its own cache. Weight reads stay at 0.73 ms because weights are shared across the batch. FFN
arithmetic intensity rises to 64 but is still below the ridge point, so the FFN remains
memory-bound. Attention's share of the step grows.

Prefill
^^^^^^^

.. code-block:: bash

   python3.12 $SKILL/operator_entitlement.py --runspec ./granite/granite4-8b-run.json \
     --tp 8 --bucket 8192 --bs 1 --prefill

.. code-block:: text

   === operator entitlement: prefill TP8 PP1 CP1 EP1 bucket=8192 bs=1 (granite-4.1-8B (dense)) ===
   op           group    inst/dev   compute       HBM (    wt     kv)  act-iface      AI  MUST-bound
   gate_proj    full         40.0   27.183m    0.732m ( 0.73m  0.00m)   3560.0MiB  8192.0  compute
   up_proj      full         40.0   27.183m    0.732m ( 0.73m  0.00m)   3560.0MiB  8192.0  compute
   down_proj    full         40.0   27.183m    0.732m ( 0.73m  0.00m)   3560.0MiB  8192.0  compute
   sdpa         full         40.0   17.399m    0.234m ( 0.00m  0.23m)    640.0MiB 16386.0  compute
   q_proj       full         40.0    8.699m    0.234m ( 0.23m  0.00m)   2880.0MiB  8192.0  compute
   o_proj       full         40.0    8.699m    0.234m ( 0.23m  0.00m)   2880.0MiB  8192.0  compute
   k_proj       full         40.0    2.175m    0.059m ( 0.06m  0.00m)   2640.0MiB  8192.0  compute
   v_proj       full         40.0    2.175m    0.059m ( 0.06m  0.00m)   2640.0MiB  8192.0  compute
   lm_head      head          1.0    0.001m    0.144m ( 0.14m  0.00m)      0.0MiB     1.0  memory

   component composition (TE/HBM certified scope; collectives excluded):
     components=81 in 3 reported records (3 cost records)  TE=120.697 ms  HBM=18.157 ms
     global-engine diagnostic=120.697 ms  serialized component floor=120.840 ms

The picture flips. Each weight is reused across all 8,192 prompt tokens, so projection
arithmetic intensity equals the prompt length, and **every projection is compute-bound**. The
FFN projections now dominate. The lm_head is the one exception: it projects only the sampled
last token, so it stays memory-bound.

The serialized component floor (120.840 ms) is slightly larger than the global TE total
(120.697 ms). That is because the memory-bound lm_head is its own component and cannot hide
under the compute-bound attention and FFN components. Adding collectives gives about 216 ms,
which matches the planner's 0.22 s TTFT for this config.

**What to optimize:** For prefill, tensor-engine utilization in the FFN matmuls is the lever:
tiling, occupancy, or FP8 compute.

Step 6. Evaluate the config across several cells
------------------------------------------------

To see the whole regime at once, describe the cells in a regime file. Save the following as
``granite/tp8-regime.json``.

.. code-block:: json

   {
     "TP": 8, "PP": 1, "CP": 1, "EP": 1,
     "cells": [
       {"category": "8k_decode",  "bucket": 8192,  "bs": 8,  "prefill": false},
       {"category": "8k_decode",  "bucket": 8192,  "bs": 16, "prefill": false},
       {"category": "8k_decode",  "bucket": 8192,  "bs": 64, "prefill": false},
       {"category": "32k_decode", "bucket": 32768, "bs": 16, "prefill": false},
       {"category": "8k_prefill", "bucket": 8192,  "bs": 1,  "prefill": true}
     ]
   }

.. code-block:: bash

   python3.12 $SKILL/config_entitlement.py --runspec ./granite/granite4-8b-run.json \
     --regime granite/tp8-regime.json

.. code-block:: text

   === config entitlement: TP8 PP1 CP1 EP1 (weights 16.76 GB, granite-4.1-8B (dense)) ===
   certified component-engine scope: TE/HBM
   category         bs   bucket      wl      kv     act         mm     comm  bound                         mem
   8k_decode         8     8192    2.93    1.88    0.01       0.14     1.77  weight HBM read              3.2G
   8k_decode        16     8192    2.93    3.75    0.03       0.28     1.86  KV HBM read                  4.5G
   8k_decode        64     8192    2.93   15.00    0.12       1.12     2.41  KV HBM read                 11.9G
   32k_decode       16    32768    2.93   15.00    0.03       0.48     1.86  KV HBM read                 11.9G
   8k_prefill        1     8192    2.93    0.23   15.00     120.70    95.63  matmul                       2.1G

Columns are milliseconds per step: ``wl`` is weight reads, ``kv`` is KV reads, ``act`` is
activation traffic, ``mm`` is matmul compute, and ``comm`` is collectives. ``mem`` is peak HBM
per device.

How to read it:

- **Decode flips from weight-bound to KV-bound between batch 8 and batch 16.** At batch 8, the
  2.93 ms of weight reads exceed 1.88 ms of KV reads. Doubling the batch doubles KV, and KV takes
  over.
- **Longer context has the same effect as larger batch.** 32k at batch 16 holds the same KV as
  8k at batch 64.
- **Collectives matter.** Even at batch 16, ``comm`` is 1.86 ms against a 6.7 ms floor. At
  prefill, ``comm`` is 95.63 ms against 120.70 ms of compute. If a profile shows much more time
  than the floor at this config, check collective wall time before blaming a kernel.

Step 7. Compare against a deployed graph (optional)
---------------------------------------------------

This step needs a compiled model, so it runs on a Trainium instance after you deploy. It
answers: *which operators actually hold more HBM than their floor?*

1. Compile the config with graph dumps enabled and the compile cache bypassed, then extract the
   compiled graph's ENTRY signature into a text manifest. See
   :ref:`entitlement-hbm-attribution` for the capture procedure.
2. Compute the decode floor at the same TP, bucket, and batch as the compiled graph.
3. Run the attribution.

.. code-block:: bash

   python3.12 $SKILL/hbm_attribution.py \
     --entry-hlo ./captured_entry_manifest.hlo \
     --floor-json agent_artifacts/tmp/operator_entitlement_decode.json \
     --runspec ./granite/granite4-8b-run.json \
     -o agent_artifacts/tmp/actual_vs_floor.json

The script ranks operator classes by how far actual resident bytes exceed the floor. Read it
this way:

- **Weight classes at ×1.00** are at entitlement.
- **A tied embedding at ×2.00** means the deployed graph stores the embedding and lm_head
  tables separately.
- **KV rows with large ratios are not operator waste.** The deployed KV figure is the paged
  pool the runtime reserved, while the floor is the KV for one bucket and batch. Every KV row is
  flagged.

For a full example of this table on a 31B model, see the skill's
``references/worked-example.md``.

Step 8. Check the results
-------------------------

.. code-block:: text

   ~/planner-tutorial/agent_artifacts/tmp/
   ├── operator_entitlement_decode.json    # Step 4 and 5: per-operator decode floor
   ├── operator_entitlement_prefill.json   # Step 5: per-operator prefill floor
   ├── config_entitlement.json             # Step 6: single-config entitlement
   └── actual_vs_floor.json                # Step 7 (optional): HBM attribution

Each rerun of ``operator_entitlement.py`` for the same phase overwrites the matching JSON file.
Pass ``-o`` to keep several cells side by side.

Next steps
----------

The floors tell you where the headroom is. To measure how much of it a deployment captures:

1. Deploy TP=8 with vLLM and capture a profile with the
   ``neuron-framework-profiling-vllm-neuron`` skill.
2. Segment the profile by collective boundaries with the
   ``neuron-framework-profile-analysis-vllm-neuron`` skill, and compare each attention and FFN
   segment against its component floor.
3. Optimize the component with the largest measured-to-floor gap first.

Troubleshooting
---------------

**"error: run-spec not found".** ``--runspec`` requires a file path. It does not look up models
by name.

**ModuleNotFoundError: torch.** ``operator_entitlement.py`` and ``config_entitlement.py`` need
``torch``. Install the CPU build into the interpreter you use.

**The script cannot find the planner engine.** The Parallelism Planner skill must be deployed
beside this skill under the same ``skills/`` directory. Rerun the deploy command.

**Reconciliation shows a gap above about 1%.** Stop and check the architecture parse. Confirm
the layer count and, for MoE models, the expert field names, then rerun the planner's
``validate_oracle.py``.

**The floor is far below my measured time.** At high TP, collective wall time can dominate.
Compare the ``comm`` column with collective time in a system trace. Also check that your
hardware constants are calibrated; see ``references/microbench-calibration.md``.

**HBM attribution total does not match the compiler's HBM log.** The manifest is incomplete,
often because a compile-cache hit skipped the dump. Recapture with the cache bypassed.

Related resources
-----------------

- :doc:`/tools/neuron-agentic-development/developer_guides/neuron-hardware-entitlement` for the deep dive on how the Hardware Entitlement skill works internally.
- :doc:`/tools/neuron-agentic-development/tutorials/parallelism-planner-tutorial` for choosing the config analyzed here.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- :doc:`/tools/neuron-explorer/index` for capturing and analyzing the profiles you compare floors against.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_ for source code and issue tracker.
