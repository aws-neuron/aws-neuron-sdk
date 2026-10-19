.. meta::
   :description: Guidelines for memory management with TorchNeuron on AWS Trainium. Covers the Neuron caching allocator, NUMA CPU affinity, 64-bit dtype behavior, and cache placement.
   :keywords: TorchNeuron, torch-neuronx, memory management, allocator, reserved, fragmentation, NUMA, OOM, int64, AWS Neuron, Trainium
   :date-modified: 2026-10-08

.. _torchneuron-guidelines-memory-management:

====================================
Guidelines for memory management
====================================

This page explains how ``torch-neuronx`` manages host and device memory and
recommends practices for managing it. Use it to understand what the memory
counters mean, tune the caching allocator, and keep host memory local to each
Neuron device.

Applies to
----------

- ``torch-neuronx`` (native TorchNeuron), eager mode and ``torch.compile``
- Trn2 and Trn3 Trainium instances

How torch-neuronx manages device memory
----------------------------------------

Device memory is managed by a **caching segment allocator**. It allocates large
**segments** of device memory from the Neuron runtime (NRT), then places
tensors as **blocks** inside those segments. Each block is an NRT slice of its
segment, so it does not need a new device memory allocation. When a tensor is
freed, its block stays in the segment for reuse and is not returned to NRT.

The allocator serves memory allocations from two pools:

- Requests smaller than 1 MB go to the small pool.
- Requests of 1 MB or larger go to the large pool.

This keeps small, short-lived allocations from fragmenting the large segments
that hold weights and activations.

Requests are also rounded differently by size:

- Requests smaller than 16 MB are rounded up to a size class, with a minimum
  of 256 bytes.
- Requests of 16 MB or larger are not rounded.

The memory APIs (see :doc:`/torch/guides/memory-profiling-guide`) report three
quantities, which always satisfy ``reserved >= allocated >= active``:

- **Reserved** — device memory the allocator holds from NRT, including cached
  free blocks inside segments. ``reserved - allocated`` is cached free memory,
  available for immediate reuse.
- **Allocated** — the total of the rounded block sizes of live tensors.
- **Active** — the total bytes that live tensors requested, without rounding.
  ``allocated - active`` is internal fragmentation from rounding.

The allocator design also helps in the following behaviors:

- **Asynchronous allocations/deallocations.** Allocations/Deallocations run in stream order on a
  runtime worker thread. When a tensor is freed, a deallocation is queued on its
  stream and runs in the order it was submitted, after the work already queued
  on that stream. The block then returns to the shared pool, where any stream
  can reuse it. If other streams also used the tensor (for example through
  ``record_stream``), a deallocation is queued on each of them, and the block
  returns to the pool only after the last one runs.
  Because the pool is shared, memory freed on one stream can serve allocations
  on any other stream. This keeps the reserved memory lower, and reusing a
  cached block is faster than allocating a new segment.
  Call ``torch.neuron.synchronize()`` before you read a memory statistic.

- **OOM triggers automatic recovery in two steps.** First, if an allocation
  fails and work is pending on the same stream, the allocator waits for that
  work to finish and free memory, then retries. Second, if it still fails, it
  releases cached free segments and retries once more before it raises an
  error. Both steps are on by default. Set ``NEURON_OOM_RECOVERY_ENABLED=0`` to
  disable the second step.

Call ``torch_neuronx.empty_cache()`` to release the cached free segments that
the allocator holds, for example between phases with very different memory
footprints. It synchronizes the devices first.

Tune the Neuron caching allocator
---------------------------------

``torch-neuronx`` also provides ways to tune the caching allocator behavior.
The defaults are tuned from training-workload experiments. Override them only
if you know the effects of these knobs. To check fragmentation, use
``segment_pool_analysis_summary()`` (see
:doc:`/torch/guides/memory-profiling-guide`).

All tuning knobs are set through one environment variable. The allocator reads
it once, the first time it is used:

.. code-block:: bash

   TORCH_NEURONX_SEGMENT_ALLOCATOR_CONFIG="key1=value1,key2=value2"

Example adjustments:

- Do not split free blocks larger than 512 MB, and let such a block serve a
  request up to 64 MB smaller than the block:
  ``kMaxSplitSizeBytes=536870912,kMaxNonSplitRoundingBytes=67108864``. By
  default, every free block can be split.
- Round requests smaller than 16 MB in finer steps to reduce internal
  fragmentation: ``kRoundupPower2Divisions=8``. The default is 4 steps between
  each power of 2.

See :doc:`/torch/reference/env-vars` for the full key table, value format,
and defaults.

Understand 64-bit dtype memory behavior
---------------------------------------

``int64`` and ``float64`` tensors keep their full 8 bytes per element in device
memory.

How ``int64`` ops run depends on ``TORCH_NEURONX_INT64_MODE``:

- ``native`` (default): each op is handled separately. Ops that support
  ``int64`` natively run on the device as 64-bit. Other ops use the packed view
  (below). Matrix multiplication runs on the CPU.
- ``packed``: every ``int64`` op uses the packed view. The device stores each
  ``int64`` element as two ``int32`` words, and arithmetic uses only the low 32
  bits. Set ``TORCH_NEURONX_PACKED_INT64_CHECK_OVERFLOW=1`` to raise an error on
  values outside the ``int32`` range.
- ``native_strict``: ops that support ``int64`` natively run on the device;
  all other ``int64`` ops run on the CPU.
- ``cpu_bounce``: ``int64`` values are cast to ``int32`` through a CPU
  round trip, and the op runs on the device.
- ``cpu_fallback``: every ``int64`` op runs on the CPU.

Neuron lacks native support for ``float64``. Conversions between ``float64``
and ``float32`` run through the CPU.

Pin host memory to the local NUMA node
--------------------------------------

On multi-rank Trainium workloads (for example, 64 ranks on a
``trn2.48xlarge``), a Neuron device's CPU threads may run on a different NUMA
node without pinning, causing host memory accesses to cross the
interconnect.

``torch_neuronx.numa`` provides NUMA CPU affinity: it pins all CPU threads in
each rank's process to the same NUMA node as its assigned Neuron device. The
API follows the same patterns as ``torch.numa.binding`` (PyTorch 2.11+),
adapted for Neuron's ``PrivateUse1`` device model. Topology is discovered
dynamically via ``neuron-ls`` — there are no hardcoded per-instance-type
tables.

NUMA CPU affinity is on by default and needs no code changes. It activates
when ``torch.distributed.init_process_group`` creates the Neuron process group.
After pinning, new memory allocations land on the local node via Linux's
first-touch policy. However, memory allocated before the pin (Python
interpreter startup, imported modules, early torch allocations) may already
reside on the wrong node. A single process that does not create a process
group is not pinned.

To also cover the allocations made before the pin, wrap each rank with
``neuron-numa-launch`` (installed with ``torch-neuronx``; requires
``numactl``). This changes only the launch command:

.. code-block:: bash

   # Before:
   torchrun --nproc_per_node 64 ... ./train.py --args

   # After:
   torchrun --no-python --nproc_per_node 64 ... neuron-numa-launch ./train.py --args

``--no-python`` tells ``torchrun`` to launch ``neuron-numa-launch`` as a
binary. The wrapper then runs Python under
``numactl --membind --cpunodebind``, so all allocations — including Python
startup — land on the local NUMA node. If ``numactl`` is not installed, the
wrapper falls back to plain ``python``, and the automatic pinning still applies.

Set ``NEURON_NUMA_PIN=0`` to disable NUMA CPU affinity.

To check pinning status at runtime:

.. code-block:: python

   from torch_neuronx.numa import get_numa_status
   print(get_numa_status())
   # {'numa_pin_enabled': True, 'numa_pinned': True,
   #  'current_affinity': [...], 'topology_available': True, 'num_devices': 16}

Place caches deliberately
-------------------------

Compilation caches trade disk for recompilation time and affect shared
filesystems:

- ``TORCH_NEURONX_NEFF_CACHE_DIR`` (default ``/tmp/neff_cache``) — point at a
  shared mount (for example, FSx) to reuse compiled NEFFs across runs and
  instances.
- ``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`` (default ``/tmp/local_cache``) — keep it
  on a **local filesystem**. It holds the cache's lock files, and file locks are
  not reliable on network filesystems such as NFS or Lustre.

See :doc:`/torch/reference/env-vars` for the complete cache variable list.

.. include:: /includes/torch/trn3-sizing-note.txt

Related reference
-----------------

- :doc:`/torch/guides/memory-profiling-guide` — Profile memory usage in torch-neuronx.
- :doc:`/torch/reference/env-vars` — TorchNeuron environment variables.
- :doc:`/torch/reference/best-practices-pytorch-neuron` — Best practices for TorchNeuron.
