.. meta::
   :description: Learn how to understand, monitor, and optimize memory usage on AWS Neuron devices such as Trainium and Inferentia ML chips.
   :date-modified: 2026-10-15

.. _neuron-device-memory-deep-dive:

Neuron device memory
====================

Neuron Runtime allocates device memory (HBM) for several distinct purposes:
your input and output tensors, compiled model constants and code, scratchpad
space for intermediate results, DMA rings, profiling buffers, and
collective-communication overhead. This page describes each category, the
tools that report memory usage (``neuron-top`` and sysfs), how to read an
out-of-memory (OOM) report, and how to reduce consumption — with the most
detail on scratchpad, which is often the largest adjustable category.

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

The Neuron Runtime's memory usage falls into the following categories:

- ``tensors``: input and output tensors allocated by your application
- ``model constants``: compiled constants used by a NEFF program
- ``model code``: the executable instructions for the Neuron Core. This also includes a micro-code overhead of 96MB per physical Neuron Core (this overhead is subject to future improvements)
- ``profile buffers``: buffers used to store profiling events
- ``scratchpad`` and ``shared scratchpad``: additional space used to store intermediary SBUF and other computations. See :ref:`nd-scratchpad` for details.
- ``dma rings``: data transfer instructions describing data movements during NEFF execution.
- ``collectives``: memory overhead used to orchestrate collective communication

You can adjust these categories as follows:

1. You control ``model constants`` and ``tensors`` entirely. Adjust them as you would on other XLA devices, through matrix dimensions, batch sizes, and so on.
2. ``scratchpad`` and ``shared scratchpad`` depend on model size, model type, and tiling strategy. See :ref:`nd-scratchpad`.
3. ``dma rings`` usage is not easily actionable. You can reduce it by using DGE where possible, or by changing the model to reduce data movements such as transfers between HBM and SBUF.
4. Neuron Runtime allocates ``profile buffers`` when you enable profiling. You can influence these allocations by disabling profiling or adjusting the buffers manually. See :ref:`nd-profile-buffers`.
5. ``model code`` usage is not actionable. If you observe significant usage, contact AWS Neuron support.



Logical Neuron Cores
~~~~~~~~~~~~~~~~~~~~~

Starting with ``trn2``, multiple physical Neuron Cores are grouped into a
single Logical Neuron Core. For details, see the :doc:`Logical NeuronCore
configuration guide </about-neuron/arch/neuron-features/logical-neuroncore-config>`.

.. note::
   On ``trn2``, the default configuration is LNC2. With LNC1 (``NEURON_LOGICAL_NC_CONFIG=1``), two neighboring Neuron Cores share an HBM, as shown in the following diagram (two vertically neighboring NeuronCore-V3s share an HBM).

   .. image:: /images/architecture/Trainium2/trainium2.png

   As a result, you can hit noisy-neighbor problems and see out-of-memory (OOM) errors earlier than expected, depending on what is loaded on the neighboring core.

Debugging tools
~~~~~~~~~~~~~~~

neuron-top
^^^^^^^^^^

``neuron-top`` shows current memory usage per core. For details, see the
:doc:`neuron-top user guide </tools/neuron-sys-tools/neuron-top-user-guide>`.


sysfs
^^^^^

You can also find the same information through sysfs. For details, see the
:doc:`Neuron sysfs user guide </tools/neuron-sys-tools/neuron-sysfs-user-guide>`.


Out-of-memory (OOM) errors
^^^^^^^^^^^^^^^^^^^^^^^^^^

When an OOM occurs, Neuron Runtime dumps a breakdown of memory usage by type
for each NEFF. For example:

.. code-block:: text

   2025-May-15 20:58:33.895937 224822:224822 ERROR  TDRV:print_lnc_hbm_details                   LNC size is 1. Neuron Cores using this HBM: NC 4 and NC 5
   2025-May-15 20:58:33.897479 224822:224822 ERROR  TDRV:log_dev_mem                             Failed to allocate 4.000GB (alignment: none, usage: tensors) on ND 0:NC 4
   2025-May-15 20:58:33.899416 224822:224822 ERROR  TDRV:log_dev_mem_usage_table                 Displaying Current Memory Utilization:
   (NOTE: the lines are LONG, and NEFF id to name mapping is printed after)

                 |          |  Model   |  Model   |          |  Shared  |          |          |DMA Rings |DMA Rings | DMA Rings |DMA Rings |           |          | Profiler |
                 |  TOTAL   |   Code   |Constants | Tensors  |Scratchpad|Scratchpad| Runtime  |    IO    |  Spill   |Collectives| Runtime  |Collectives|  XT CC   | Buffers  |
   ND 0 Overall  | 20.188GB |192.102MB | 82.344KB | 20.000GB |  0.000B  |  0.000B  |350.125KB |179.000KB | 64.000KB |  0.000B   | 68.000KB |  0.000B   |  0.000B  |  0.000B  |
   \_NC 4        | 20.094GB | 96.065MB | 58.344KB | 20.000GB |  0.000B  |  0.000B  |229.062KB |118.000KB | 48.000KB |  0.000B   | 36.000KB |  0.000B   |  0.000B  |  0.000B  |
     \_NEFF 1001 |263.906KB | 28.562KB | 34.344KB |   n/a    |   n/a    |  0.000B  |108.000KB | 57.000KB | 32.000KB |  0.000B   | 4.000KB  |  0.000B   |   n/a    |   n/a    |
     \_NEFF 1002 |244.875KB | 31.875KB | 24.000KB |   n/a    |   n/a    |  0.000B  |108.000KB | 61.000KB | 16.000KB |  0.000B   | 4.000KB  |  0.000B   |   n/a    |   n/a    |
   \_NC 5        | 96.285MB | 96.037MB | 24.000KB |  0.000B  |  0.000B  |  0.000B  |121.062KB | 61.000KB | 16.000KB |  0.000B   | 32.000KB |  0.000B   |  0.000B  |  0.000B  |
     \_NEFF 1003 |244.875KB | 31.875KB | 24.000KB |   n/a    |   n/a    |  0.000B  |108.000KB | 61.000KB | 16.000KB |  0.000B   | 4.000KB  |  0.000B   |   n/a    |   n/a    |

   NEFF id to name mapping:
   1001: "1.0.41235.0+df4a714bb-/local/out-test0_meta_dense"
   1002: "1.0.41235.0+df4a714bb-/local/out-test0_meta_concat3"
   1003: "1.0.41235.0+df4a714bb-/local/out-test0_meta_concat3"

If this OOM message is truncated, the same information is available at
``/tmp/neuron_mem_table_device_<device_id>_hbm_<hbm_idx>.log``.

Per-NEFF INFO logs
^^^^^^^^^^^^^^^^^^

A NEFF's memory usage is also available as ``INFO``-level logs during model
load. Set ``NEURON_RT_LOG_LEVEL_TDRV=info`` to see a log like the following:

.. code-block:: text

   2025-May-15 07:41:15.014997 2198754:2198754  INFO  TDRV:dml_log_dev_neff_mem
   [ND 0:NC 0] Current Usage Total: 96.543MB
           shared scratchpad: 0.000B
   Per NEFF memory usage breakdown for [out-test0_meta_concat3]:
           Total: 230.562KB
           * model code: 30.562KB
           * model constants: 24.000KB
           * scratchpad: 0.000B
           * runtime: 95.000KB
           * dma rings io: 61.000KB
           * dma rings spill: 16.000KB
           * dma rings collectives: 0.000B
           * dma rings runtime: 4.000KB
           * collectives: 0.000B


.. _nd-profile-buffers:

Profile buffers
---------------

When you use NRT's profiling APIs and ``neuron-profiler capture``, Runtime
allocates buffers to store the profiling events. These buffers default to about
64 or 128 MB each, so expect around 2 GB of overhead (*subject to future
changes*).

You can adjust these buffer sizes with the
``NEURON_RT_PROFILE_BUF_<buffer type>_MB`` flags — for example,
``NEURON_RT_PROFILE_BUF_DMA_MB=512``. The adjustable buffers are ``EVENT``,
``DMA``, ``H2D_DMA``, ``THROTTLE``, ``CC_CORE_INSTRUCTION``, and
``CC_CORE_EVENT``. ``H2D_DMA`` records host-device DMA events; setting
``NEURON_RT_PROFILE_BUF_H2D_DMA_MB=0`` turns off host-device DMA capture.

.. note::
   Adjusting the buffer sizes manually is not recommended: buffers that are too small cause the profiler to lose events. Profile one NEFF at a time, and adjust buffer sizes only when profiling a single NEFF still runs out of memory.

To reduce memory usage further while profiling, use ``--single-io``. This
option creates one IO tensor the size of the largest IO tensor in the model;
the other IO tensors point to slices of it during execution. The output is no
longer correct, but the profile still captures performance realistically.
``--single-io`` is only available in ``neuron-profile``.

.. code-block:: bash

   neuron-profile capture -n file.neff --single-io

.. note::
   Only device profiles require extra device memory; system profiles do not. If you only need a high-level view of kernel execution latency and time spent in Neuron Runtime APIs, capture a system profile with the ``nrt_sys_trace_fetch_events`` or ``NEURON_RT_INSPECT_ENABLE`` APIs. System profiles use a host-side, per-NeuronCore event buffer, which can overflow and drop events on long or high-throughput runs. See :ref:`Handling dropped events and partial profiles <neuron-explorer-dropped-events>` to detect and control this.

.. _nd-scratchpad:

Scratchpad
----------

Aside from inputs and outputs, a NEFF execution needs additional HBM space for
temporary spills out of the state buffer (the cache), because a program's
working set can be arbitrarily large and may not fit in the state buffer. This
space is the **scratchpad**.

The compiler specifies a NEFF's scratchpad size entirely. It depends on kernel
size, kernel type, and tiling strategy. For a training workload, scratchpad
usage is usually determined by the size of the activation between the forward
and backward layer; for an inference kernel, by the size of the hidden states.
Optimal tiling and fusion of collective and compute operations can reduce
scratchpad usage substantially.

A NEFF's ``def.json`` records how much scratchpad space the NEFF requires.
Runtime allocates scratchpad memory on the HBM, per NeuronCore, and uses it only
while a NEFF is running. Because of this, Runtime shares the memory among all
loaded NEFFs to reduce the overall footprint: it allocates a **shared
scratchpad** used by all NEFFs loaded on a given NeuronCore. The shared
scratchpad's size equals the size of the largest scratchpad among the loaded
NEFFs. In some cases a variable cannot go in the shared scratchpad and is placed
in a **non-shared scratchpad** specific to a NEFF (see `Scratchpad variables`_).

Scratchpad variables
~~~~~~~~~~~~~~~~~~~~

The compiler fully manages the scratchpad space. A NEFF defines scratchpad
variables and their **size** and **offset** within that space. Runtime maps
every variable to the scratchpad space it allocates on the HBM. Some variables
overlap, because not all variables are "live" at the same time during NEFF
execution.

Runtime iterates through all scratchpad variables in ``def.json`` and computes
the ``MAX`` of ``offset + size`` across them. That value is the shared
scratchpad space the NEFF requires.

Shared scratchpad
~~~~~~~~~~~~~~~~~

The **shared scratchpad** is shared among all NEFFs loaded on a given
NeuronCore. This works because only one NEFF executes at a time on a NeuronCore,
and data cannot pass from one NEFF to another through the scratchpad. The
scratchpad grows and shrinks as NEFFs load and unload. To support this, Runtime
allocates the shared scratchpad in chunks called **scratchpad pages**.

Once a variable is placed in a scratchpad page, its physical location cannot
change: the variable cannot move to another page, and the page itself cannot
move. This is because Runtime generates DMA descriptors that point to the
variables' physical addresses, and it generates them only once, during NEFF
load. The number of pages grows and shrinks as NEFFs load and unload, but the
variables of loaded NEFFs keep their physical locations. When a new NEFF
requires more scratchpad space than any currently loaded NEFF, Runtime allocates
new pages.

Runtime reports shared scratchpad pages in the OOM output under the category
**shared scratchpad**, and in sysfs under:

.. code-block:: text

   /sys/devices/virtual/neuron_device/neuron<device_number>/neuron_core<nc_number>/stats/memory_usage/device_mem/model_shared_scratchpad/

.. _contiguous-scratchpad:

Contiguous shared scratchpad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Runtime 2.33 and later can allocate the scratchpad pages contiguously —
adjacent to each other in the HBM. This is supported on Inf2, Trn1, and
Trn2/Trn3 (LNC size 2 only); it is not supported for LNC size 1 on Trn2/Trn3.
On supported platforms it also requires Neuron Driver 2.24 or later. Where
supported, it is enabled by default; disable it by setting
``NEURON_RT_CONTIGUOUS_SCRATCHPAD=0``.

Because the pages are contiguous, they form a single scratchpad region that
Runtime grows or shrinks, and a scratchpad variable can span the entire region.

Here the page size is 64 MB, so Runtime grows or shrinks the region in multiples
of 64 MB. This size keeps wasted scratchpad small while still growing the region
in a reasonable number of allocations. It is not configurable:
``NEURON_SCRATCHPAD_PAGE_SIZE`` has no effect when contiguous scratchpad is
enabled.

Because the pages must be adjacent, in rare cases of high HBM fragmentation
Runtime cannot grow the region further even though the HBM has free space. Model
load then fails with OOM. In that case, disable contiguous scratchpad by setting
``NEURON_RT_CONTIGUOUS_SCRATCHPAD=0``, along with the page-size environment
variables described later in this document.

Non-contiguous page allocations (``NEURON_RT_CONTIGUOUS_SCRATCHPAD=0``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This mode is the default for LNC size 1 on Trn2/Trn3, and whenever contiguous
scratchpad cannot be used for another reason (for example, when Neuron Driver is
older than 2.24).

Because the pages are not necessarily contiguous in HBM, a scratchpad variable
must fit entirely within one page to be placed in the shared scratchpad
(``(var_offset % NEURON_SCRATCHPAD_PAGE_SIZE) + var_size <= NEURON_SCRATCHPAD_PAGE_SIZE``).
The default scratchpad page size in Runtime is 512 MB; using the environment
variables described later, you can set it to any multiple of 512 MB, up to
3.5 GB.

Non-shared/Private scratchpad allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a variable cannot fit into a shared scratchpad page, Runtime makes a
completely separate allocation for it. This arises only with non-contiguous page
allocations (``NEURON_RT_CONTIGUOUS_SCRATCHPAD=0``).

For example, suppose the scratchpad page size is 512 MB and you load the
following two NEFFs:

NEFF A has the following scratchpad variables (using a different format from ``def.json`` for brevity here):

``a_var1: {offset: 0, size: 536870912 [512 MB]}, a_var2: {offset: 536870912 [512 MB], size: 1073741824 [1 GB]}``

NEFF B has the following scratchpad variables:
``b_var1: {offset: 0, size: 104857600 [100 MB]}, b_var2: {offset: 104857600 [100 MB], size: 1610612736 [1.5 GB]}``

``a_var1`` and ``b_var1`` both satisfy the condition to fit within the 512 MB
shared scratchpad page. Because they are at the same offset, they share the same
shared scratchpad page.

``a_var2`` and ``b_var2`` are both larger than 512 MB, so Runtime makes separate
allocations for them: 1 GB of private allocation for NEFF A and 1.5 GB for
NEFF B.

This example has 2.5 GB of non-shared scratchpad allocations on the HBM. They
appear under the category **scratchpad** in the OOM output, and in sysfs under:

.. code-block:: text

   /sys/devices/virtual/neuron_device/neuron<device_number>/neuron_core<nc_number>/stats/memory_usage/device_mem/model_shared_scratchpad/

Runtime still calculates and allocates the required shared scratchpad. That
comes to 1.5 GB for NEFF A and 1.6 GB for NEFF B, so the maximum is 1.6 GB;
rounded up to the page size, it comes to 2 GB. Runtime therefore allocates 2 GB
of shared scratchpad (4 pages) and 2.5 GB of non-shared scratchpad, even though
it uses only 1 page of the shared scratchpad.

If you set the page size to 2 GB (``NEURON_SCRATCHPAD_PAGE_SIZE=2048`` — see the
environment variables described later), all variables fit within the shared
scratchpad page. After loading both NEFFs, Runtime allocates a single shared
2 GB page and no HBM for non-shared scratchpad. Choosing the right scratchpad
page size can therefore reduce HBM allocations substantially.

How to avoid high non-shared scratchpad usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the OOM report shows high non-shared scratchpad usage — high ``scratchpad``
category, but not ``shared scratchpad`` — it typically means Runtime is not
using contiguous scratchpad and the scratchpad variables are larger than the
default Runtime page size.

Examples of non-shared scratchpad usage in an OOM report:

.. code-block:: text

   Overall HBM usage
       * total: 23.577GB
       * ...
       * shared scratchpad: 9.000GB
       * scratchpad: 8.149GB   <--- non-shared scratchpad allocations
       * ...

Or, with recent changes to OOM reporting:

.. code-block:: text

                                                                   non-shared scratchpad allocations
                                                                             |
                                                                             v
                 |          |  Model   |  Model   |          |  Shared  |          |          |DMA Rings | ...
                 |  TOTAL   |   Code   |Constants | Tensors  |Scratchpad|Scratchpad| Runtime  |    IO    | ...
   ND 0 HBM 0    | 23.577GB |932.370MB | 1.438MB  | 5.359GB  |  9.000GB |  8.149GB |203.062KB |118.000KB | ...
   ...

You can experiment with larger scratchpad page sizes through the following
environment variables, for the compiler and Runtime respectively:

.. code-block:: bash

   export NEURON_CC_FLAGS=' <other flags if required> --hbm-scratchpad-page-size=<size in MB> ' # Env var for Neuron Compiler
   export NEURON_SCRATCHPAD_PAGE_SIZE=<size in MB>  # Env var for Neuron Runtime

Both environment variables specify the scratchpad page size in MB. They do not
apply when contiguous shared scratchpad is in use.

As an example, setting the scratchpad page size to 2 GB:

.. code-block:: bash

   export NEURON_CC_FLAGS=' --hbm-scratchpad-page-size=2048 '
   export NEURON_SCRATCHPAD_PAGE_SIZE=2048

Set the Neuron Compiler variable as well; otherwise the compiler may set the
variable offsets inefficiently.

The size must be a multiple of 512 and less than 4096 (4 GB). Setting it too low
leads to non-shared allocations; setting it too high can waste memory, because
the last page allocated may be only partly used. On OOM, try values such as
2048 (2 GB), 1536 (1.5 GB), and 1024 (1 GB).

Appendix: NEFF format for scratchpad variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you unpack a NEFF (using ``neuron-packager``) and inspect ``sg00/def.json``
(and ``sg01/def.json`` for NEFFs generated for the Trn2 LNC size 2
configuration), you see variable entries like these:

.. code-block:: text

   "var": {
           "some_variable_name": {
               "backing_variable_off": 17108992,
               "ops": [],
               "size": 131072,
               "type": "virtual",
               "var_id": 2349
           },
           ...
    }

A ``type`` of "virtual" indicates a scratchpad variable. The
``backing_variable_off`` field is the offset inside the shared scratchpad space
Runtime allocates, and the ``size`` field is the size of the variable.

DMA rings
---------

**DMA rings** are buffers that store DMA **descriptors**; each descriptor
describes a data movement that the DMA engines can execute.

DGE generates descriptors dynamically during NEFF execution, so a NEFF that uses
DGE for a given DMA needs no HBM allocation for those descriptors.

For DMAs that do not use DGE, Runtime allocates the DMA rings on HBM and builds
the descriptors before execution. The NEFF encodes the details for building
these descriptors in ``def.json`` and ``<engine>.json``, where ``<engine>`` is
the TPB engine that triggers the DMA operation.

Reducing DMA ring usage requires changes in the NEFF itself; the most effective
change is using DGE for DMAs where supported.

In OOM reports, DMA rings are further categorized as:

1. IO — descriptors whose source or destination is an I/O tensor.
2. Spill — descriptors that move data between NEFF variables/tensors, excluding I/O tensors.
3. Collectives — descriptors that move data for collective operations between ranks on the same node.
4. Runtime — descriptors that do not correspond to explicit DMAs in the NEFF but are needed to support NEFF execution. Examples: loading DVE and activation tables, and instruction-fetch DMAs for TPB engines.
