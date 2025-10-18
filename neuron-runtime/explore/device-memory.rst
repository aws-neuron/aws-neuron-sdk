.. meta::
   :description: Learn how to understand, monitor, and optimize memory usage on AWS Neuron devices such as Trainium and Inferentia ML chips. 
   :date-modified: 10/16/2025

.. _neuron-device-memory-deep-dive:

Neuron Device Memory
====================

Learn how to understand, monitor, and optimize memory usage on AWS Neuron devices. This topic covers memory categories including tensors, model constants, scratchpad allocations, DMA rings, and profiling buffers. Discover debugging tools like neuron-top and neuron-monitor, troubleshoot out-of-memory (OOM) errors, and implement strategies to reduce memory consumption for efficient ML workload execution on Inferentia and Trainium instances.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The Neuron Runtime's memory usage falls into the following categories:

- ``tensors``: input and output tensors allocated by application
- ``model constants``: compiled constants used by a NEFF program
- ``model code``: the executable instructions for the Neuron Core. This also includes a micro-code overhead of 96MB per physical Neuron Core (this overhead is subject to future improvements)
- ``profile buffers``: buffers used to store profling events
- ``scratchpad`` and ``shared scratchpad``: additional space used to store intermediary SBUF and other computations. Read :ref:`nd-scratchpad` for details.
- ``dma rings``: Data transfer instructions describing data movements during NEFF execution, used during NEFF execution.
- ``collectives``: Memory overhead used to orchestrate collective communication

Here's what users can do to adjust these forms of memory usage:

1. ``model constants`` and ``tensors`` are entirely controlled by the user. Adjust similar to other XLA devices with matrix dimensions, batch sizes, etc.
2. ``scratchpad`` and ``shared scratchpad`` depend on model size, model type and tiling strategy. Read the :ref:`nd-scratchpad`.
3. ``dma rings`` usage is not easily actionable. It can be reduced by using DGE where possible, or changing the model to reduce data movements (like transfers between HBM and SBUF).
4. ``profile buffers`` are allocated when the user enables profiling. Users can influence these allocations by either disabling profiling or manually adjusting. Read the :ref:`nd-profile-buffers` section.
5. ``model code`` usages are not actionable. If users observe significant usage, contact your AWS Neuron support.



Logical Neuron Cores
~~~~~~~~~~~~~~~~~~~~~

Starting with ``trn2``, we introduced the concept of Logical Neuron Cores, where multiple physical Neuron Cores are grouped into the same "Neuron Core". Read :doc:`this article </about-neuron/arch/neuron-features/logical-neuroncore-config>` for more details.

.. note::
   On ``trn2``, the default configuration is LNC2, but when using LNC1 (``NEURON_LOGICAL_NC_CONFIG=1``), two neighboring Neuron Cores will end up **SHARING a HBM**. See the following diagram, where two vertically neighboring NeuronCore-V3s share a HBM.

   .. image:: /images/architecture/Trainium2/trainium2.png

   As a result, there will be **noisy neighbor problems**, and you may see out-of-memory (OOM) errors earlier than expected depending on what is loaded on the neighboring core.

Debugging Tools
~~~~~~~~~~~~~~~

neuron-top
^^^^^^^^^^

Running ``neuron-top`` will give you a view of the current memory usages on a core level. Read :doc:`this article </tools/neuron-sys-tools/neuron-top-user-guide>` for more details.


sysfs
^^^^^

As an alternative, you can find the same information from the sysfs. Read :doc:`this article </tools/neuron-sys-tools/neuron-sysfs-user-guide>` for more details.

Out-of-memory (OOM) Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^

When an OOM occurs, the Neuron Runtime dumps a detailed breakdown of the various memory usage types for each NEFF. For example:

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

In case this OOM message is truncated, this information is also available under ``/tmp/neuron_mem_table_device_<device_id>_hbm_<hbm_idx>.log``.

Per-NEFF INFO logs
^^^^^^^^^^^^^^^^^^

The memory usage of a NEFF is also available as ``INFO`` level logs during model load. By using ``NEURON_RT_LOG_LEVEL_TDRV=info``, you'll see a log like:

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

Profile Buffers
---------------

When used with NRT's profiling APIs and ``neuron-profiler capture``, Runtime allocates buffers in order to store the profiling events. These profiling buffers by default are about 64 or 128 MB each, so expect around 2 GB overhead. (*subject to future changes*)

These profiler buffer sizes can be manually adjusted by setting flags ``NEURON_RT_PROFILE_BUF_<buffer type>_MB``. For example, ``NEURON_RT_PROFILE_BUF_DMA_MB=512``. Here's a list of the different buffers one can attempt adjusting: ``EVENT``, ``DMA``, ``THROTTLE``, ``CC_CORE_INSTRUCTION``, ``CC_CORE_EVENT``.

.. note::
   Adjusting the buffer sizes manually is NOT recommended, since buffers too small will cause profiler to lose events. **Prioritize profiling one NEFF at a time, and only consider when profiling a single NEFF still OOMs.**

Another option for reducing memory usage further when profiling is to use the ``--single-io``. This option will reduce the memory used by IO tensors by creating an IO tensor the size of the largest IO tensor in the model. Other IO tensors will point to slices of this tensor during execution. The output will no longer be correct but the profile will still realistically capture performance. Note that the ``--single-io`` option is only available to ``neuron-profile``.

.. code-block:: bash

   neuron-profile capture -n file.neff --single-io

**NOTE**: only device profiles require extra device memory. System profiles do not. If you are only interested in a high-level view of performance kernel execution latency and time spent in Neuron runtime APIs, consider capturing a system profile with the ``nrt_sys_trace_fetch_events`` or ``NEURON_RT_INSPECT_ENABLE`` APIs.

.. _nd-scratchpad:

Scratchpad
----------

Aside from inputs and outputs, a NEFF execution requires additional space on HBM for temporary spills out of the state buffer (the cache). This is necessary because the working set of a program can be arbitrarily large, and may not fit in the state buffer. We call this space **scratchpad**.

Scratchpad size requirement for a NEFF is specified entirely by the compiler. Scratchpad size depends on kernel size, kernel type and tiling strategy. For example, for a training workload, scratchpad usage is usually determined by the size of activation between forward and backward layer. For an inference kernel, scratchpad usage is usually determined by the size of hidden states. Additionally, optimal tiling and fusion of collective and/or compute operations can reduce scratchpad usage significantly.

``def.json`` within a NEFF contains information about how much scratchpad space is required for the NEFF. Scratchpad memory is allocated on the HBM, per NeuronCore. The memory is only used while a NEFF execution is running. Thus it makes sense to share this memory among all loaded NEFFs to reduce the overall memory footprint. Runtime allocates a **shared scratchpad** - that is shared by all NEFFs loaded on a particular NeuronCore. The size of the **shared scratchpad** size is equivalent of the size of the largest **scratchpad** among all the loaded NEFFs. In some cases a variable cannot be placed in **shared scratchpad** and is placed in a **non-shared scratchpad** specific to a NEFF (see `Scratchpad variables`_ below).

Scratchpad variables
~~~~~~~~~~~~~~~~~~~~

The scratchpad space is fully managed by the Compiler. A NEFF defines scratchpad variables and their **size** and **offset** within the scratchpad space. Runtime maps all these variables to the scratchpad space it allocates on the HBM. Some of the variables may overlap with others since not all variables are "live" at the same time during NEFF execution.

Runtime iterates through all scratchpad variables in ``def.json`` and computes ``MAX`` of ``offset + size`` over all of them. That is the size of the shared scratchpad space required by the NEFF.

Shared scratchpad
~~~~~~~~~~~~~~~~~

As the name implies, **shared scratchpad** is shared among all programs/NEFFs loaded on a particular NeuronCore. This is possible because only one NEFF executes at a time on a NeuronCore, and data cannot be passed from one NEFF to other through the scratchpad. That means the scratchpad dynamically grows/shrinks with NEFF loads/unloads. To achieve that, the **runtime allocates the shared scratchpad in chunks**, referred to as **scratchpad pages**.

Once a variable is placed in a scratchpad page the variable's physical location cannot be changed, i.e. the variable cannot be moved to another page and the page itself cannot be moved. That is because during NEFF load the Runtime generates DMA descriptors that point to the variables' physical addresses and the descriptors are generated only once during NEFF load. The number of pages can grow and shrink as NEFFs are loaded and unloaded but the variables for the loaded NEFFs retain their physical locations. When a new NEFF is loaded, it might require larger **scratchpad** space than any of the currently loaded NEFFs. In that case new pages are allocated, but the pages are not necessarily contiguous with the previously allocated pages.

Because the pages are not contiguous in HBM, a scratchpad variable must fit entirely within a page in order to be placed in the shared scratchpad (``(var_offset % NEURON_SCRATCHPAD_PAGE_SIZE) + var_size <= NEURON_SCRATCHPAD_PAGE_SIZE``). The default scratchpad page size in Runtime is 512 MB and through environment variables described later in this document, it can be set to any multiple of 512 MB, up to a maximum of 3.5 GB.

Shared scratchpad pages are shown in the OOM reporting in Runtime as category **"shared scratchpad"** and in sysfs under:

.. code-block:: text

   /sys/devices/virtual/neuron_device/neuron<device_number>/neuron_core<nc_number>/stats/memory_usage/device_mem/model_shared_scratchpad/

Non-shared/Private scratchpad allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a variable cannot fit into a shared scratchpad page, Runtime makes a completely separate allocation for it.

As an example, let's say scratchpad page size is 512 MB, and we load the following two NEFFs:

NEFF A has the following scratchpad variables (using a different format from ``def.json`` for brevity here):

``a_var1: {offset: 0, size: 536870912 [512 MB]}, a_var2: {offset: 536870912 [512 MB], size: 1073741824 [1 GB]}``

NEFF B has the following scratchpad variables:
``b_var1: {offset: 0, size: 104857600 [100 MB]}, b_var2: {offset: 104857600 [100 MB], size: 1610612736 [1.5 GB]}``

``a_var1`` and ``b_var1`` both satisfy the condition to fit within the 512 MB shared scratchpad page. Since they are both at same offset, they will end up sharing the same shared scratchpad page.

But ``a_var2`` and ``b_var2`` are both bigger than 512 MB, Runtime will make separate allocations for them. So there will be 1 GB of private allocation for NEFF A and another 1.5 GB of private allocation for NEFF B.

In this example we would have 2.5 GB of non-shared scratchpad allocations on the HBM. These would show up as category **"scratchpad"** in the OOM reporting in Runtime, and in sysfs under: 

.. code-block:: text

   /sys/devices/virtual/neuron_device/neuron<device_number>/neuron_core<nc_number>/stats/memory_usage/device_mem/model_shared_scratchpad/

One thing to note in this case is that Runtime will still calculate the required amount of shared scratchpad and allocate it. It comes to 1.5 GB for NEFF A and 1.6 GB for NEFF B - so the maximum among the NEFFs is 1.6 GB; and rounded up to scratchpad page size, it comes to 2 GB. Thus, Runtime will allocate 2 GB of shared scratchpad (or 4 pages), and 2.5 GB of non-shared scratchpad allocations in this case, even though it only ends up using 1 page of the shared scratchpad.

If the page size is set to 2GB (by setting ``NEURON_SCRATCHPAD_PAGE_SIZE=2048`` - see environment variables described later in this doc), all variables would fit within the shared scratchpad page. After loading both NEFFs only a single shared 2 GB page will be allocated, with zero HBM consumed by the non-shared scratchpad. Thus, choosing the right scratchpad page size can reduce HBM allocations by a significant amount.

How to avoid high non-shared scratchpad usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the OOM report has a high amount of non-shared scratchpad usage (i.e. high ``scratchpad`` category usage, but not ``shared scratchpad`` category), it typically means that the scratchpad variables are larger than the default Runtime scratchpad page size.

Examples of non-shared scratchpad usage in OOM report:

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

You can try experimenting with larger scratchpad page sizes through the following environment variables for Compiler and Runtime respectively:

.. code-block:: bash

   export NEURON_CC_FLAGS=' <other flags if required> --hbm-scratchpad-page-size=<size in MB> ' # Env var for Neuron Compiler
   export NEURON_SCRATCHPAD_PAGE_SIZE=<size in MB>  # Env var for Neuron Runtime

Both these environment variables specify the scratchpad page size in MBs (megabytes)

As an example, setting scratchpad page size to 2 GB:

.. code-block:: bash

   export NEURON_CC_FLAGS=' --hbm-scratchpad-page-size=2048 '
   export NEURON_SCRATCHPAD_PAGE_SIZE=2048

Note that the env variable for Neuron Compiler needs to be set as well, otherwise it may set the offsets for the variables in an inefficient manner.

**The size should be a multiple of 512 and less than 4096 (4 GB)**. Setting the scratchpad page size too low would lead to non-shared allocations, and setting it too high could also lead to memory wastage (as the last scratchpad page allocated may only be partially utilized). It is recommended to try values like 2048 (2 GB), 1536 (1.5 GB) and 1024 (1 GB) in case of OOM.

Appendix: NEFF format for scratchpad variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we unpack a NEFF (using ``neuron-packager``), and inspect ``sg00/def.json`` (and ``sg01/def.json`` in case of NEFFs generated for Trn2 LNC size 2 configuration), we will see variables entries like these:

.. code-block:: json

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

``type`` being "virtual" for a variable indicates that it is a scratchpad variable. The ``backing_variable_off`` field is the offset inside the shared scratchpad space allocated by Runtime, and the ``size`` field is the size of the variable.

DMA Rings
---------

**DMA rings** are buffers used to store DMA **descriptors** (each descriptor describes a data movement that the DMA engines can execute).

DGE generates the descriptors dynamically during NEFF execution, so, if a NEFF is using DGE for some DMA, then no allocation is needed on the HBM for those descriptors.

For any DMAs not using DGE, Runtime must allocate the DMA rings on HBM and build the DMA descriptors before execution. The details for building the descriptors for these DMAs in the NEFF is encoded in ``def.json`` and ``<engine>.json`` where ``<engine>`` is the TPB engine that will trigger the DMA operation.

Overall, reducing DMA rings usage requires changes in the NEFF itself, with the most effective change being using DGE for DMAs where supported.

In OOM reports, DMA rings are further categorized as:

1. IO - These descriptors have an I/O tensor as their source or destination
2. Spill - These descriptors move data between any NEFF variables/tensors, excluding any I/O tensors
3. Collectives - These descriptors move data for collectives operations between ranks on the same node
4. Runtime - These descriptors do not correspond to any explicit DMAs in the NEFF but are needed to perform DMAs to support NEFF execution. Examples: loading DVE and activation tables, instruction fetch DMAs for TPB engines