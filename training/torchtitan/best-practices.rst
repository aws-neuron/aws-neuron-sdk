.. meta::
   :description: Best practices for running TorchTitan on AWS Neuron. Covers eager vs. compile mode, correctness-first validation, FSDP tuning, scale-out, and anti-patterns for TorchTitan-on-Neuron workloads.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, best practices, FSDP, performance, eager, compile
   :date-modified: 2026-10-15

.. _torchtitan-neuron-best-practices:

=========================================
Best practices for TorchTitan Neuron
=========================================

This topic covers best practices for running TorchTitan on AWS Neuron. Learn
when to use eager versus ``torch.compile`` mode, how to validate correctness
before scaling, and how to tune FSDP and scale out across Trainium instances.

Overview
--------

The best practices here help you reach predictable throughput and convergence
with TorchTitan on Neuron, and avoid the common problems teams hit on a first
large-model run. They assume you have a working install
(:doc:`/training/torchtitan/get-started/setup-torchtitan-neuron`) and have run
the basic tutorial (:doc:`/training/torchtitan/tutorials/tutorial-basic-training`).

Choosing eager vs. compile mode
-------------------------------

TorchTitan runs in two execution modes, and the right choice depends on where
you are in the development cycle:

- **Eager mode — for faster iteration.** PyTorch's eager mode executes
  operations as soon as they are encountered, making debugging iterations
  faster at the expense of steady-state throughput. Use it when
  bringing a model up for the first time, debugging a config, or iterating on
  model code — the shorter edit-run loop is worth more than peak throughput
  while you are still getting the pipeline correct. The basic tutorial uses
  eager mode for exactly this reason.
- **Compile mode — for performance.** Enabling ``torch.compile`` lets the
  Neuron backend fuse and optimize the graph, which raises throughput for
  steady-state training runs. Switch to compile mode once the pipeline is
  validated and you are running real training. The
  :doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training` tutorial
  covers the compile path and its Neuron-specific flags.

A practical workflow is: bring the model up in eager mode, confirm loss is
finite and trending down, then re-run the same config with compile enabled for
the production run.

General principles
------------------

- **Enable before you tune.** Get a model running correctly in a minimal
  configuration before touching parallelism or precision settings. Reduce the
  model size so it fits on a single node for faster debugging iteration, then
  scale back up once the pipeline is correct. A correct eager run is the
  baseline everything else is measured against.
- **Validate correctness by adding one dimension at a time.** Start from a
  trusted 1D FSDP configuration, then layer in tensor, context, or pipeline
  parallelism one dimension at a time, comparing against the simpler baseline
  at each step. This isolates which change caused a divergence.
- **Profile before optimizing.** Use the memory and execution profilers to
  debug slow performance or unexpected out-of-memory issues before hand-tuning
  configuration. See
  :doc:`/training/torchtitan/troubleshooting` for the TorchTitan profiler
  interface.

TorchTitan Neuron best practices
--------------------------------

**1:** Validate the pipeline with a small, short run before scaling up

- Follow this practice when: Onboarding a new TorchTitan workload to
  Neuron for the first time.
- Does not apply to: Incremental tuning of an already-validated pipeline.

Run a trusted config for a handful of steps in eager mode and confirm the run
steps, emits finite loss, and writes a checkpoint before you commit to a long
or multi-node run. You can shorten any config from the command line without
editing it:

.. code-block:: bash

   NGPU=64 MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_tp4_fsdp16_sac \
   ./run_train.sh --training.steps 10

**2:** Add one parallelism dimension at a time, keeping tokens-per-step fixed

- Follow this practice when: Validating a new parallelism or distributed
  technique (TP, CP, PP) against a known-good baseline.
- Does not apply to: Re-running an already-validated multi-dimensional
  configuration.

Use a 1D FSDP run as the trusted control, then introduce the new dimension on
top of it. Keep the input deterministic and hold the global batch size (and
tokens per iteration) constant across the two runs — fix
``training.local_batch_size`` and the data-parallel degree — so a loss
difference is attributable to the new technique and not to a changed batch.
Adding the dimensions one at a time makes a divergence easy to localize.

**3:** Use the shipped per-family configs as your baseline

- Follow this practice when: Pre-training a supported model family
  (for example Qwen3).
- Does not apply to: Custom model definitions not covered by a shipped
  config.

TorchTitan registers tuned configurations in its config registry, and you
select one by name through the ``CONFIG`` environment variable (for example
``qwen3_8b_tp4_fsdp16_sac``). The shipped config already encodes the
parallelism layout, batch sizing, sequence length, and activation-checkpointing
policy for that model/instance combination, so start from it and override
individual values on the command line only as needed, rather than assembling a
configuration from scratch.

Prefer the Neuron-tuned configs over upstream GPU configs of the same model.
Upstream GPU sharding configurations typically assume a world size of 8 with
higher memory per rank, whereas a typical Trainium configuration is 64 ranks
with lower memory per rank. Fitting the same model therefore requires a higher
combined TP × FSDP degree on Trainium, which the shipped Neuron configs already
account for.

**4:** Enable activation checkpointing before lowering batch size

- Follow this practice when: Hitting device-memory pressure on
  large-model configurations.
- Does not apply to: Small models where the recomputation overhead costs
  more step time than it is worth.

When a run is close to the memory limit, prefer enabling (or widening)
activation checkpointing over dropping the batch size — recomputation trades a
modest step-time cost for a large memory saving, and keeps the effective batch
size intact. Shipped configs set an activation-checkpointing policy; check the
config before you override it. Full activation checkpointing saves more memory
at a higher recomputation cost than selective (SAC).

**5:** Pin and log exact versions for every run

- Follow this practice when: Any run whose results you may need to
  reproduce — in practice, every training run.
- Does not apply to: Throwaway local experiments where reproducibility
  does not matter.

Record the versions of TorchTitan, torch-neuronx, PyTorch, and the Neuron SDK,
along with the model and dataset revision, with each run. Exact reproducibility
across different distributed layouts is not guaranteed — RNG consumption and
accumulation order differ by sharding — so capturing the environment is what
lets you explain a difference later. In particular, changing the data-parallel
degree changes which batches the dataloader produces, so do not expect
bit-exact loss curves when you modify it.

Performance best practices
--------------------------

- **Compile for steady-state runs.** Once a config is validated in eager mode,
  enable ``torch.compile`` for the production run (see
  :doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training`).
- **Consider SimpleFSDP once compiled.** As a further optimization beyond
  ``torch.compile``, SimpleFSDP expresses FSDP in a compile-friendly way so the
  backend can overlap and fuse the sharding collectives with compute. See
  :doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training`.
- **Keep the global batch size constant when comparing configs.** Use
  ``--training.global_batch_size`` (gradient accumulation) to hold the
  effective batch fixed while you vary parallelism, so throughput comparisons
  are comparable.

Scale-out best practices
------------------------

Move from single-node to multi-node only after the single-node run is
validated. Scale by adding parallelism dimensions in order of overhead, not by
hand-editing process counts (the tutorial covers how to launch across nodes):

- **Start with FSDP.** It shards parameters, gradients, and optimizer state
  across data-parallel ranks and is the lowest-overhead way to fit a larger
  model.
- **Add tensor parallelism (TP) if the model still does not fit.** TP shards
  individual layers across ranks, reducing per-rank memory beyond what FSDP
  alone provides.
- **Add pipeline parallelism (PP) last.** PP partitions the model into stages
  across nodes; reach for it once FSDP and TP together cannot fit the model.

For MoE models, the torchtitan-neuron fork supports expert parallelism (EP);
EP documentation is planned for a later release.

For how to launch across nodes, see
:ref:`Single-node vs. multi-node on Neuron <torchtitan-neuron-tutorial-basic-single-multi-node>`
in the basic training tutorial. ``run_train.sh`` reads ``NNODES`` and
``RDZV_ENDPOINT`` and passes them to ``torchrun``; an external launcher such as
MPI starts it once per node.

When scaling to more ranks, set the data-parallel replicate degree
deliberately. By default ``parallelism.data_parallel_shard_degree`` is ``-1``,
so FSDP sharding absorbs every rank left after the other parallelism
dimensions -- adding nodes silently widens the FSDP shard group, which past a
point costs more collective traffic than the memory saving is worth. Pin
``--parallelism.data_parallel_replicate_degree`` above 1 to replicate instead
of shard across those ranks, keeping the shard group sized for performance
rather than growing it with every node you add.

Anti-patterns
-------------

- **Tuning before validating.** Reaching for parallelism or precision knobs
  before a minimal eager run is correct. Get a correct baseline first, then
  tune.
- **Changing several variables between runs.** Varying batch size, sequence
  length, and parallelism at once makes a divergence or OOM impossible to
  attribute. Change one thing at a time.
- **Overriding mixed precision per module without a reason.** Inconsistent
  per-module precision is a common source of shape/dtype errors; keep the
  config's mixed-precision settings unified unless you have a specific need.

Additional recommendations
--------------------------

- Set a checkpoint cadence that bounds lost work to an acceptable amount of
  wall-clock time for your run length.
- Keep logging and metrics conventions consistent across runs so a later
  comparison is possible. TorchTitan logs loss, memory, throughput (tokens/s),
  and MFU.
- When a run fails, hangs, or diverges, see
  :doc:`/training/torchtitan/troubleshooting`.

.. include:: /includes/torch/trn3-sizing-note.txt

Further reading
---------------

- :doc:`/training/torchtitan/about-torchtitan-neuron`
- :doc:`/training/torchtitan/tutorials/index`
- :doc:`/training/torchtitan/troubleshooting`
- :doc:`/torch/reference/best-practices-pytorch-neuron` — TorchNeuron-level
  best practices that apply beneath TorchTitan.
