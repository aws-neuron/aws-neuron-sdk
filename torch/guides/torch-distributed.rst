.. meta::
   :description: Launch Neuron workers, run a collective, use DDP, FSDP2, DTensor, and pipeline parallelism, and overlap communication with compute using HostCC and streams.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium, torch.distributed, process group, collectives, EFA, DDP, FSDP, DTensor, pipeline parallelism, HostCC, streams
   :date-modified: 2026-10-15

.. _torchneuron-torch-distributed-guide:

======================================
How to use torch.distributed on Neuron
======================================

Task overview
-------------

Launch workers and run a collective, then choose DDP, FSDP2, DTensor, or
pipeline parallelism for your model. Use HostCC and streams to schedule communication alongside
independent compute. See the `API compatibility guide <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`_ for supported APIs and their limits.

Prerequisites
-------------

- A working native TorchNeuron installation and at least two allocated logical
  NeuronCores.
- **For multi-node execution: EFA installed and configured on every node.**
  Follow the `Step 3: Install EFA in the Neuron manual setup instructions <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html#installation-steps>`_.
  Use the same package versions and script on all nodes, and ensure network
  connectivity between workers.

Instructions
------------

**1: Write a basic collective**

Save this as ``test_distributed.py``:

.. code-block:: python

   import os

   import torch
   import torch.distributed as dist
   import torch_neuronx  # Registers the Neuron device and distributed backend.

   rank = int(os.environ["RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   dist.init_process_group(
       backend="neuron",
       init_method="env://",
       rank=rank,
       world_size=world_size,
   )
   try:
       process_group = dist.group.WORLD
       value = torch.tensor([float(rank + 1)], device=torch.device("neuron"))
       dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
       print(
           f"rank={rank} local_rank={os.environ['LOCAL_RANK']} "
           f"world_size={world_size}: all_reduce={value.cpu().item()}",
           flush=True,
       )
   finally:
       dist.destroy_process_group()

``init_process_group`` creates the default ``ProcessGroup``, available as
``dist.group.WORLD``. ``all_reduce`` updates the tensor in place with the sum
from every rank in that group. Every worker must call the collective.

Initialize the group before allocating Neuron tensors. The Neuron backend
uses ``LOCAL_RANK`` and ``LOCAL_WORLD_SIZE`` to assign each worker a visible
core; tensor allocation then uses that worker's current Neuron device.

**2: Launch two workers on one node**

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 test_distributed.py

``torchrun`` sets the following environment variables. The script reads the
global rank and world size; the backend reads the local values. See the
upstream `torchrun environment variables <https://docs.pytorch.org/docs/2.14/elastic/run.html#environment-variables>`_.
When using a custom launcher, set both ``LOCAL_RANK`` and ``LOCAL_WORLD_SIZE``
consistently on each node, along with the global rank and rendezvous variables.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Variable
     - Meaning
     - Two-worker example
   * - ``RANK``
     - Unique worker index across the whole job.
     - ``0`` or ``1``
   * - ``WORLD_SIZE``
     - Total number of workers across all nodes.
     - ``2``
   * - ``LOCAL_RANK``
     - Worker index on the current node.
     - ``0`` or ``1``
   * - ``LOCAL_WORLD_SIZE``
     - Number of workers on the current node.
     - ``2``
   * - ``MASTER_ADDR``, ``MASTER_PORT``
     - Shared rendezvous address and port used by ``env://``.
     - Set by ``--standalone``.

Use a process count within your allocation. To select an allocated core range,
prefix the launch command with, for example, ``NEURON_RT_VISIBLE_CORES=4-5``.
The backend assigns one entry in that range to each local worker.

**3: Launch across two nodes (optional)**

Follow the `Step 3: Install EFA in the Neuron manual setup instructions <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html#installation-steps>`_
on both nodes before launching the workers.
Run the following on each node, using the first node's reachable private
address for ``MASTER_ADDR``. Set ``NODE_RANK=0`` on the first node and
``NODE_RANK=1`` on the second:

.. code-block:: bash

   export MASTER_ADDR="FIRST_NODE_PRIVATE_ADDRESS"
   export NODE_RANK=0
   torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
       --master-addr="$MASTER_ADDR" --master-port=29503 \
       test_distributed.py

This launches four workers: global ranks ``0, 1`` on the first node and
``2, 3`` on the second. ``WORLD_SIZE=4`` everywhere, while ``LOCAL_RANK`` is
``0`` or ``1`` on each node. ``NODE_RANK`` identifies the node; it is not a
worker's ``RANK``.

Confirm your work
-----------------

With two workers, both print ``all_reduce=3.0``. With four workers, every
worker prints ``all_reduce=10.0``. The launcher should exit successfully.


Choose a model parallelism API
------------------------------

Run these setup snippets after process-group initialization. Create the
optimizer after wrapping or sharding the model.

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - API
     - Model layout
     - Communication
   * - DDP
     - A complete model replica on each worker.
     - All-reduce gradient buckets during backward.
   * - FSDP2
     - Parameters, gradients, and optimizer state sharded across workers.
     - All-gather parameters and reduce-scatter gradients.
   * - DTensor / tensor parallelism
     - Tensors or individual layers partitioned across workers.
     - Collectives when operations require a different tensor layout.
   * - Pipeline parallelism
     - Consecutive model sections assigned to different stages.
     - Point-to-point activation and gradient transfers between stages.

DDP: replicate a model
~~~~~~~~~~~~~~~~~~~~~~

Use ``DDP`` when the model fits on one logical NeuronCore. Each worker processes
its own batch; DDP synchronizes gradients.

.. code-block:: python

   from torch.nn.parallel import DistributedDataParallel as DDP

   model = torch.nn.Linear(16, 16).to("neuron")
   model = DDP(model)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

Partition the training data with a ``DistributedSampler`` and call
``sampler.set_epoch(epoch)`` each epoch. All workers must execute matching
training steps.

DDP schedules all-reduces as gradient buckets become ready. Use
`register_comm_hook <https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook>`_
to customize bucket reduction. The hook must return a Future containing the
reduced bucket tensor and handle gradient normalization; DDP does not divide
the bucket by the world size before calling the hook.

Start with the default DDP options; the current Neuron tests
mark unused-parameter, static-graph, and bucket-capacity cases as expected
failures.

FSDP2: shard a model and prefetch parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``fully_shard`` on individual blocks before the root model. This gives FSDP separate
parameter groups, so a later block's all-gather can run while the current
block computes.

.. code-block:: python

   from torch.distributed.device_mesh import init_device_mesh
   from torch.distributed.fsdp import fully_shard

   mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("dp",))
   torch.manual_seed(0)  # Start from identical parameters on all ranks.
   model = torch.nn.Sequential(
       torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()),
       torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()),
   ).to("neuron")
   for block in model:
       fully_shard(block, mesh=mesh, reshard_after_forward=True)
   fully_shard(model, mesh=mesh)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

For overlap, :ref:`enable HostCC <torchneuron-distributed-overlap>` before
starting workers. PyTorch 2.14 FSDP2 creates its own all-gather and reduce-scatter streams and
manages their dependencies. With streams and HostCC enabled, its non-default
stream collectives use the HostCC path. Start with FSDP's implicit prefetching.
For a fixed block order, explicit prefetching can issue the next request
earlier from the CPU:

.. code-block:: python

   model[0].set_modules_to_forward_prefetch([model[1]])
   model[1].set_modules_to_backward_prefetch([model[0]])

Prefetching more blocks retains more parameter buffers. With
``reshard_after_forward=True``, each block releases its full parameters after
forward and gathers them again for backward. Retaining parameters can reduce
communication at the cost of memory. Measure both choices for your model.

DTensor: partition tensors and layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DeviceMesh`` defines the worker topology; ``DTensor`` placements describe which
tensor dimensions are sharded or replicated. Tensor-parallel styles apply
those layouts to modules:

.. code-block:: python

   from torch.distributed.device_mesh import init_device_mesh
   from torch.distributed.tensor.parallel import (
       ColwiseParallel, RowwiseParallel, parallelize_module,
   )

   tp_mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("tp",))
   model = torch.nn.Sequential(
       torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16),
   ).to("neuron")
   model = parallelize_module(
       model, tp_mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()},
   )
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

Choose dimensions divisible by the mesh size. Create meshes and apply
parallelism before compilation. When combining TP and FSDP, use a two-dimensional
mesh named ``("dp", "tp")``, apply TP on ``mesh["tp"]`` first, then FSDP on
``mesh["dp"]``.

TP communication often lies on the next layer's dependency path. A sharded
layout or asynchronous redistribution alone does not establish overlap.
FSDP prefetching can provide independent communication to overlap with TP
compute. Do not assume CUDA/Inductor asynchronous-TP settings enable Neuron
overlap.

.. _torchneuron-distributed-pipeline:

Pipeline parallelism: split layers across stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``torch.distributed.pipelining`` when you want each worker to own a
different section of the model. A schedule divides the batch into
microbatches and exchanges activations and gradients between stages.
Configure EFA/RDMA, including for a single-node pipeline.

For a two-worker job, insert this after process-group initialization:

.. code-block:: python

   from torch.distributed.pipelining import PipelineStage, Schedule1F1B

   assert world_size == 2
   if rank == 0:
       stage_model = torch.nn.Sequential(
           torch.nn.Linear(16, 16), torch.nn.ReLU(),
       ).to("neuron")
   else:
       stage_model = torch.nn.Linear(16, 16).to("neuron")

   stage_device = next(stage_model.parameters()).device
   example_input = torch.empty(
       2, 16, device=stage_device, requires_grad=(rank > 0),
   )
   example_output = torch.empty(2, 16, device=stage_device, requires_grad=True)
   stage = PipelineStage(
       stage_model, stage_index=rank, num_stages=2, device=stage_device,
       input_args=example_input, output_args=example_output, group=dist.group.WORLD,
   )
   schedule = Schedule1F1B(stage, n_microbatches=4, loss_fn=torch.nn.MSELoss())
   optimizer = torch.optim.SGD(stage_model.parameters(), lr=0.01)
   optimizer.zero_grad()
   if rank == 0:
       schedule.step(torch.randn(8, 16, device=stage_device))
   else:
       schedule.step(target=torch.randn(8, 16, device=stage_device))
   optimizer.step()

Use the same two-worker ``torchrun`` launch as the basic collective example.
Each worker owns one stage; the example batch of eight is split into four
microbatches of two. ``schedule.step`` runs forward and backward, so do not
call ``loss.backward()`` separately. Keep stage boundary shapes and dtypes
consistent with the example metadata.
The input metadata on later stages must require gradients so backward can
propagate to the preceding stage.

Start with ``ScheduleGPipe`` for a fill-drain schedule or ``Schedule1F1B`` for
alternating forward and backward in steady state. The compatibility reference
describes other schedules and their stage-layout requirements. See
`API compatibility guide <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`_.

Pipeline schedules overlap work across stages. HostCC configures tensor
collectives; it does not choose the pipeline schedule or its point-to-point
transport. When combining PP with DDP, FSDP, or TP, use separate mesh dimensions
and groups for each parallelism axis, and wrap each worker's local stage.
Confirm overlap and remaining pipeline bubbles in a device profile.


.. _torchneuron-distributed-overlap:

Overlap communication with compute
----------------------------------

Streams are enabled by default. Enable HostCC before Python starts, on every
node. If you previously set ``TORCH_NEURONX_ENABLE_STREAMS=0``, remove that
override before using HostCC.

Before running the command below, add the communication-stream example in
this section inside the ``try`` block of ``test_distributed.py``, before
``destroy_process_group``. The original basic collective uses the default
stream, so enabling HostCC alone does not exercise the HostCC path:

.. code-block:: bash

   export TORCH_NEURONX_ENABLE_HOST_CC=1
   torchrun --standalone --nproc-per-node=2 test_distributed.py

HostCC allows supported collectives on non-default streams to overlap with
independent computation. The host submits these collectives directly to the
Neuron Runtime, and the collective itself executes on device. Some
non-contiguous inputs may require a CPU round trip during input preparation.

The backend selects the collective path using the current stream when the
operation is submitted. Collectives on the default stream retain the compiled
collective path, including during eager execution. With HostCC disabled,
compiled operations containing collectives are routed to the default stream
while preserving ordering with the submitting stream.

For application-managed overlap, use a communication stream, wait for the
input producer, run independent compute, and join before consuming the
collective result. Add this example inside the script's ``try`` block after
process-group initialization:

.. code-block:: python

   compute_stream = torch.neuron.current_stream()
   comm_stream = torch.neuron.Stream()
   values = torch.full((1024,), float(rank + 1), device=torch.device("neuron"))
   activation = torch.randn(512, 512, device=torch.device("neuron"))

   comm_stream.wait_stream(compute_stream)
   with torch.neuron.stream(comm_stream):
       work = dist.all_reduce(values, group=dist.group.WORLD, async_op=True)

   independent = activation @ activation
   work.wait()  # Order the current compute stream after the collective.
   combined = independent + values.sum()
   result_on_cpu = combined.cpu()

``work.wait()`` orders the current stream; its default form need not block
the CPU until the device finishes. Keep the tensors alive until their work
completes. Avoid device-wide synchronization, barriers, and CPU copies between
launch and the independent compute. The final CPU copy above is for inspection.

The HostCC path supports all-reduce, all-gather, and reduce-scatter. Equal-split
all-to-all raises an error on that path rather than automatically falling
back to the default stream. On Trn2 and Trn3, torch-neuronx automatically
enables the required ``NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true`` safety setting.
This limits kernels to seven PSUM banks and can reduce compute performance.
Explicitly setting it to ``false`` while HostCC is enabled is rejected.

Warm up compilation, then compare steady-state step time with HostCC disabled
and enabled in separate processes. Use a Neuron profile to confirm that
collectives and compute overlap on the device timeline. Different stream IDs
or successful results alone do not prove a speedup.


Common issues
-------------

.. rubric:: The runtime initialized before the process group

Initialize the Neuron process group before you create Neuron tensors or use
device APIs. Run a single-device installation check in a separate process.

.. rubric:: Ranks hang in a collective

Check that every member calls the same collective in the same order. Check
shapes, dtypes, roots, and split sizes. Use a matching number of training steps
on all ranks. For two-way point-to-point communication, use a batched exchange.

.. rubric:: A subgroup barrier fails

Neuron's barrier requires a group the size of the default world. Use a
default-group barrier only at a point reached by every default-group member.

.. rubric:: A point-to-point operation fails during transport setup

Check the EFA installation and fabric configuration on the participating nodes.
Check that the tensor is contiguous and the peer is explicit.

.. rubric:: Updating the timeout has no effect

Set ``timeout`` in ``init_process_group`` or ``new_group``. The Neuron backend
does not implement runtime timeout updates through ``set_timeout``.

.. rubric:: An object collective creates a graph break

Object collectives serialize Python objects on the host. Neuron explicitly
places several of these wrappers outside Dynamo capture. Keep them outside
compiled regions and use tensor collectives for bulk tensor data.

Related information
-------------------

- :doc:`/torch/tutorials/tutorial-torch-distributed` -- End-to-end distributed
  training tutorial.
- `API compatibility guide <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`_ -- Compatibility for
  FSDP, DTensor, DDP, dtypes, and compilation.
- :doc:`/torch/troubleshooting` -- TorchNeuron troubleshooting.

Upstream PyTorch references:

- `torch.distributed <https://docs.pytorch.org/docs/2.14/distributed.html>`_
- `DDP <https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html>`_
- `FSDP2 <https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html>`_
- `DTensor and tensor parallelism <https://docs.pytorch.org/docs/2.14/distributed.tensor.parallel.html>`_
- `Pipeline parallelism <https://docs.pytorch.org/docs/2.14/distributed.pipelining.html>`_
