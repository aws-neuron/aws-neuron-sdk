.. meta::
   :description: Choose efficient FSDP, tensor, pipeline, and hybrid parallel layouts for training workloads on Trainium2.
   :keywords: AWS Neuron, Trainium2, training, best practices, DeviceMesh, DTensor, FSDP, tensor parallelism, pipeline parallelism, HSDP, PyTorch, NKI, nki-library
   :date-modified: 2026-10-16

.. _training-best-practices:

.. JIRA: NDOC-262

=========================================
Best practices for training on AWS Neuron
=========================================

Choose a layout for your model's memory requirements and communication pattern
on ``trn2.48xlarge``. Use :doc:`/torch/guides/torch-distributed` for APIs and
launch instructions.

Understand the unit of sharding
-------------------------------

A ``trn2.48xlarge`` has 16 Trainium2 chips with eight physical NeuronCores and
96 GiB of HBM each. The default Logical NeuronCore configuration 2 (LNC2)
combines pairs of physical cores into four logical NeuronCores per chip:
**64 ranks per instance** and approximately
**24 GiB of HBM per rank**, before runtime overhead. The layouts below use
one process per logical core.

Size the job against the per-rank memory budget. Persistent parameter,
gradient, and optimizer shards share that budget with gathered weights,
activations, prefetched layers, and temporary buffers.

Keep TP4 within a chip. Use NeuronLink-v3's 4 x 4 torus for FSDP between chips,
and EFA for communication between separate instances.

For the hardware details, see :ref:`Amazon EC2 Trn2 architecture <aws-trn2-arch>`
and :ref:`Logical NeuronCore configuration <logical-neuroncore-config>`.
Trn2 UltraServers use NeuronLink between their constituent instances and have
a different topology.

Choose a starting layout
------------------------

Start with the least model parallelism that leaves memory headroom. These dense
model sizes are **selection heuristics**; sequence length, batch size, layer
width, precision, and checkpointing determine the final layout.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Workload
     - Starting layout
     - Reason to change it
   * - Small model whose complete training state fits on one rank
     - DDP, without parameter sharding
     - Use FSDP if replicated parameters, gradients, optimizer state, and
       activations leave too little memory for useful batches.
   * - Dense model below roughly 14B parameters
     - One node: FSDP64, TP1
     - Use this while gathered layers and activations fit per rank. Switch to
       TP4 when they constrain memory or it improves measured throughput.
   * - Dense model around 14B to 32B parameters
     - One node: TP4 x FSDP16
     - Add nodes to the FSDP group if model state still needs more memory.
   * - Larger dense model, such as a 70B model
     - Start with TP4 x FSDP across a few nodes; four nodes give FSDP64.
     - If gathered layers or activations still do not fit, use activation
       checkpointing, reduce the local batch, or select a supported wider TP layout.
   * - A layout that fits within one node and needs more data throughput
     - Replicate it across nodes with hybrid sharded data parallelism (HSDP).
     - Expand the sharding group if one replica needs more memory.
   * - Deep model whose layer working set or cross-node FSDP traffic limits training
     - Use PP with TP4 and FSDP inside each stage.
     - Use HSDP instead when a replica fits within one node and only throughput
       needs to grow.

Mixture-of-experts models additionally need expert placement and dispatch;
active parameter count alone does not determine memory.

Use FSDP alone while the per-rank working set fits
--------------------------------------------------

Use FSDP64 while gathered layers and activations fit per rank. Each rank
stores a shard of the persistent model state and processes its own input batch.
FSDP gathers the parameters of an execution unit, such as a transformer block,
before computing it, then reduces and scatters its gradients.

More FSDP ranks reduce persistent storage, but each rank still holds the full
gathered execution unit. Distinguish these two memory limits: increasing
sharding helps parameter, gradient, and optimizer storage; it does not split
the gathered layer or its computation.

Start with block-level sharding to control the gathered working set.
Leave room for activations, prefetched weights, and temporary buffers.
Use activation checkpointing when activation memory limits the local batch,
and account for its recomputation cost when comparing throughput.

Add TP4 when layer or activation memory becomes limiting
--------------------------------------------------------

Use **TP4 x FSDP16** when gathered layers or activations limit useful batches.
TP partitions supported layer weights and computation across four ranks.
Tensor and sequence parallel plans determine which activations are also
distributed. FSDP then shards each TP partition across the 16 chips.

Place each TP4 group on **one chip** to keep frequent TP collectives local.
With consecutive ranks, TP groups are ``[0, 1, 2, 3]``, ``[4, 5, 6, 7]``,
and so on. FSDP joins corresponding TP positions across chips, such as
``[0, 4, 8, ..., 60]``. This separates frequent layer-level communication
from parameter gathering and gradient reduction.

Compare TP1 and TP4 for long sequences, wide layers, or a large vocabulary.
Increase TP beyond four only when memory or measured throughput justifies
crossing chip boundaries. Check rank-to-core placement with custom launchers
or core visibility settings.

Think in mesh axes
------------------

Use PyTorch DTensor and DeviceMesh to express the layout. A mesh axis names
one parallelism role; ranks differing only along that axis form its
communication group:

* ``tp`` ranks compute different pieces of the same layer on the same examples.
* ``shard`` ranks store different pieces of model state and process different
  examples.
* ``replicate`` ranks hold matching shards in different replicas and synchronize
  their gradients.
* ``pp`` ranks execute different layer stages on the same examples.

DTensor placements describe how a tensor is distributed over those groups.
``Shard(dim)`` partitions a tensor dimension; ``Replicate()`` keeps a copy.
FSDP manages parameter gathering and gradient reduction around execution;
a ``Shard`` placement alone does not provide that behavior.

For dense workloads without context parallelism:

.. code-block:: text

   world_size = pp_degree * replicate_degree * shard_degree * tp_degree

Mesh dimensions describe parallelism, rather than the physical torus shape.
With consecutive local ranks mapped to consecutive logical cores, keep TP
innermost and replication outside sharding. Put PP outermost when each stage
occupies a separate node:

.. list-table::
   :header-rows: 1
   :widths: 35 10 20 35

   * - Layout
     - Nodes
     - Mesh shape
     - Mesh axis names
   * - FSDP only
     - 1
     - ``(64,)``
     - ``("shard",)``
   * - TP4 with FSDP within a node
     - 1
     - ``(16, 4)``
     - ``("shard", "tp")``
   * - TP4 with FSDP spanning four nodes
     - 4
     - ``(64, 4)``
     - ``("shard", "tp")``
   * - TP4 with node-local FSDP and four replicas
     - 4
     - ``(4, 16, 4)``
     - ``("replicate", "shard", "tp")``
   * - Two pipeline stages with stage-local FSDP and TP4
     - 2
     - ``(2, 16, 4)``
     - ``("pp", "shard", "tp")``

For the single-node ``(16, 4)`` mesh, each row is a chip-local TP group.
Each column is an FSDP group spanning the 16 chips:

.. code-block:: text

                     tp axis
                   0   1   2   3
   shard axis  0  [0,  1,  2,  3]   chip 0
               1  [4,  5,  6,  7]   chip 1
              ...       ...
              15 [60, 61, 62, 63]   chip 15

Apply tensor parallelism to the TP submesh and FSDP to the shard submesh.
For HSDP, give FSDP the two-dimensional ``("replicate", "shard")`` submesh.
In the row-major ``(4, 16, 4)`` mesh,
``rank = 64 * replicate + 4 * shard + tp``. The shard group
``[0, 4, ..., 60]`` stays within a node; the replicate group
``[0, 64, 128, 192]`` synchronizes matching shards across nodes.
Grow the shard axis for more model-state memory, or the replicate axis for
more data throughput once a replica fits.

Check collective support for the complete mesh
-----------------------------------------------

Check TP and data-parallel groups together. These single-node layouts were
checked with 64 ranks at LNC2:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Layout
     - Placement
     - Observed result and recommendation
   * - TP1 x FSDP64
     - Default ``(64,)`` shard mesh
     - Training matched distributed CPU results.
   * - TP4 x FSDP16
     - Default ``(16, 4)`` mesh; each TP group occupies one chip
     - Training matched distributed CPU results.
   * - TP8 x FSDP8
     - Default ``(8, 8)`` mesh
     - Unsupported DP collective topology. Use the non-default placement below,
       or reduce TP to four.
   * - TP8 x FSDP8
     - Non-default ``(8, 8)`` mesh below
     - Collectives and training matched expected results. Measure throughput
       before preferring it to TP4.

**The default TP8 x FSDP8 rank mapping creates an unsupported DP collective topology.**
On one ``trn2.48xlarge`` at LNC2, a row-major ``("dp", "tp")`` mesh of
``(8, 8)`` creates these groups when rank IDs match logical core IDs:

.. code-block:: text

   TP: [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, ..., 15], ...
   DP: [0, 8, 16, 24, 32, 40, 48, 56], [1, 9, ..., 57], ...

TP groups span adjacent chips; DP groups skip every other chip. The chip IDs
occupy this grid, with wraparound links on both axes:

.. code-block:: text

                     chip column
                   0        1        2        3
   chip row 0    chip 0   chip 1   chip 2   chip 3
            1    chip 4   chip 5   chip 6   chip 7
            2    chip 8   chip 9   chip 10  chip 11
            3    chip 12  chip 13  chip 14  chip 15

Grid entries are chip IDs. At LNC2,
``rank = 4 * chip_id + logical_core_position``, where the logical core position
is ``0`` through ``3``. Chip 3 (row 0, column 3) contains ranks ``12`` through
``15``; chip 12 (row 3, column 0) contains ranks ``48`` through ``51``.

Rank 0 on chip 0 (row 0, column 0) and rank 8 on chip 2 (row 0, column 2)
need an intermediate hop through chip 1 or chip 3. Neither intermediate chip
participates in this DP group; its two columns have no direct links between
participants.
The collective library cannot select an algorithm for this topology:
``failed (2) to init a collective algorithm. reason: no_hier no_mesh``.
TP all-reduce succeeds, but FSDP parameter all-gather fails before training.

EFA connects separate instances; it does not repair this intra-node placement.
For algorithm and connectivity details, see
:doc:`/neuron-runtime/explore/intranode-collective-comm` and
:doc:`/neuron-runtime/about/collectives`.

**Unblock TP8 x FSDP8 with a non-default mesh.** Keep the same TP and DP
degrees, pairing adjacent chips for TP as shown in the code below.
Pairs ``(0,12)`` through ``(3,15)`` use wraparound links. DP groups select one
core per chip from the connected regions ``0-7`` and ``8-15``.

For example, the first TP group becomes ``[0,1,2,3,48,49,50,51]``.
The first DP group becomes ``[0,4,8,12,16,20,24,28]``.

After initializing the 64-rank process group, create groups in the same order
on every rank. ``sort_ranks=False`` preserves mesh order for shard and batch
assignment.

.. code-block:: python

   import torch
   import torch.distributed as dist
   from torch.distributed.device_mesh import DeviceMesh

   chip_pairs = [(0, 12), (1, 13), (2, 14), (3, 15),
                 (4, 8), (5, 9), (6, 10), (7, 11)]
   ranks = torch.tensor([
       [4 * chip + core for chip in pair for core in range(4)]
       for pair in chip_pairs
   ])
   groups = {}
   for name, axis_groups in (("dp", ranks.T), ("tp", ranks)):
       for members in axis_groups.tolist():
           group = dist.new_group(ranks=members, sort_ranks=False)
           if dist.get_rank() in members:
               groups[name] = group
   mesh = DeviceMesh.from_group(
       [groups["dp"], groups["tp"]],
       device_type="neuron",
       mesh=ranks,
       mesh_dim_names=("dp", "tp"),
   )

This mesh passed collective checks and ten compiled training steps against
distributed CPU results on all 64 ranks. Measure throughput before choosing
it over chip-local TP4. To keep a default mesh, use **TP4 x FSDP16** instead.

Scale out by replicating a sharding group
-----------------------------------------

When a replica fits within one instance, use **node-local FSDP with replication
across nodes** (HSDP). Parameter all-gathers stay local; matching gradient
shards synchronize across nodes over EFA. Each node holds a complete model
replica distributed over its local shard and TP groups.

For example, four replicas of TP4 x FSDP16 use a ``(4, 16, 4)`` mesh and
256 ranks across four nodes. The replicate axis adds independent training
examples and DDP-style gradient synchronization without extending each
parameter all-gather across those nodes.

Extend FSDP across nodes when persistent state needs more memory. TP4 x FSDP64
uses four instances, with parameter gathers and gradient reductions crossing
node boundaries. It does not reduce gathered-layer or activation memory.
Once that sharding group fits, replicate the group to add throughput.
Compare wider FSDP with HSDP at the same global batch: the choice trades
additional sharding headroom against inter-node parameter traffic.

Install EFA software on **every node** before multi-node training. Follow
**Step 3: Install EFA** in the `Neuron manual setup instructions
<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html#installation-steps>`_,
for Trn2.

Use PP at node boundaries
-------------------------

Use PP across nodes when a deep model needs more capacity or cross-node FSDP
parameter traffic limits throughput. EFA is the slowest tier in this hierarchy.

For two nodes, a ``("pp", "shard", "tp")`` mesh of ``(2, 16, 4)`` places one
stage per node with TP4 and FSDP16 inside it. This mesh confines cross-node
model communication to stage-boundary activations and their gradients, keeping
TP collectives and FSDP parameter gathers local. Prefer this placement over
extending TP or repeated parameter all-gathers across the inter-node links.

Choose kernel implementations for the workload
-----------------------------------------------

Choose kernels for the shapes and dtypes produced by your sharding plan:

* **Use supported PyTorch operations** before specializing kernels.
* **Check the NKI Library first.** The :doc:`NKI Library </nki/library/index>` and the
  `aws-neuron/nki-library repository
  <https://github.com/aws-neuron/nki-library>`_ provide reusable kernels. Check
  shape, dtype, backward support, and compiler compatibility against the local
  tensors on each rank.
* **Write an NKI kernel for a measured bottleneck** when existing implementations
  are insufficient. Design tiling and data movement around the NeuronCore's
  compute engines, HBM, on-chip scratch buffer (SBUF), and partial-sum memory
  (PSUM). See the :doc:`NKI documentation </nki/index>`,
  :doc:`language guide </nki/get-started/nki-language-guide>`, and
  :doc:`PyTorch integration </torch/guides/how-to-work-with-nki>`.

Check outputs and gradients against a reference, then measure performance
with the workload's sharded tensors. Recheck the kernel's shape contracts
when changing TP degree or tensor placement, and provide the backward
implementation required for training.

Measure before keeping an optimization
--------------------------------------

Compare steady-state tokens per second, step time, and peak memory over at
least ten steps after compilation and warmup. Hold sequence length, precision,
checkpointing, and global batch fixed. Check loss and checkpoint restore after
changing the layout.

Use :doc:`/tools/neuron-explorer/index` and
:doc:`/torch/guides/memory-profiling-guide` to distinguish the limiting resource:

* **Persistent model state:** increase FSDP sharding.
* **Gathered weights or activations:** use TP4, checkpointing, or a smaller
  local batch with accumulation.
* **Inter-node parameter traffic:** keep FSDP local and scale through replicas
  when the replica fits.
* **Compute idle during communication:** use supported prefetching and HostCC
  stream overlap from the distributed guide.
* **Host overhead:** compare eager execution with ``torch.compile`` using the
  same mesh and workload.

For compiled collectives, see :doc:`/torch/tutorials/tutorial-torch-distributed`.
Exclude compilation and cache population from throughput measurements, and
reuse a compatible compilation cache between runs.

Further reading
---------------

* :doc:`/torch/reference/best-practices-pytorch-neuron` for execution-mode
  guidance and general PyTorch practices.
* `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__
  for supported APIs and release-specific restrictions.
