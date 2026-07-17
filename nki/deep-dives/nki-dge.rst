.. meta::
   :description: Deep dive into the Descriptor Generation Engine (DGE) modes for DMA operations in NKI on AWS Neuron hardware.
   :keywords: NKI, DGE, DMA, descriptor, swdge, hwdge, gather, scatter, AWS Neuron, Trainium
   :date-modified: 03/31/2026

.. _dge-documentation:

=============================================
Descriptor Generation Engine (DGE) Reference
=============================================

Every DMA operation (``nisa.dma_copy``, ``nisa.dma_transpose``) needs a
*descriptor* that tells the hardware the source address, destination address,
transfer shape, and stride pattern. We can specify *when* and *where* those
descriptors are produced---on the host before execution, on the GpSimd engine
at runtime, or on a dedicated hardware block. Each choice has different
performance characteristics and capability constraints. DGE (Descriptor
Generation Engine) is the umbrella term for the strategies that control this.

In the NKI API, there are three concrete strategies---plus an ``unknown`` mode
that lets the compiler choose---exposed through the ``nki.isa.dge_mode`` enum.
The rest of this document describes each mode, its constraints, and when to use
each one.

.. contents:: On this page
   :local:
   :depth: 2


DGE Modes
----------

``unknown`` --- let the compiler decide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.unknown)

The default. The compiler selects the best mode based on the target hardware,
tensor shapes, and surrounding instruction schedule. Use this unless you have a
specific reason to force a specific mode.

``none`` --- pre-computed descriptors in HBM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.none)

DMA descriptors are pre-computed on the Trainium host **before** NEFF
execution. The pre-computed descriptors are stored in HBM. At runtime the
DMA engine reads the pre-built descriptor directly---no on-device generation is
needed.

**When to use:**

- Fully static transfer patterns where source/destination addresses are known
  at compile time.
- When you want to avoid any on-device descriptor generation overhead.

**Trade-offs:**

- Descriptors consume HBM capacity (one per DMA instruction instance).
- Cannot handle dynamic (runtime-computed) addresses or indices.

``swdge`` --- software DGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.swdge)

The **GpSimd Engine** generates DMA descriptors during NEFF execution. This is
the only mode that supports indirect (gather/scatter) operations with dynamic
indices from SBUF.

**When to use:**

- Dynamic addresses that depend on runtime values.
- Gather or scatter operations using ``vector_offset`` (indirect indexing).
- Indirect transpose (``dma_transpose`` with indirect ``src``).

**Trade-offs:**

- Consumes GpSimd Engine cycles for descriptor generation.
- May compete with other GpSimd workloads.

Importantly, ``swdge`` has additional constraints for indirect transpose:

- ``src.shape[-1] <= 128``
- ``src.dtype`` must be 2 bytes (``float16`` / ``bfloat16``)
- ``src`` must be on HBM
- ``src.shape[0]`` must be divisible by 16
- When ``src`` is 4D: ``src.shape[1]`` or ``src.shape[2]`` must be 1
- Index tensor must be 2-D, on SBUF, with dtype ``uint32``
- ``indices.shape[0]`` must be in ``[16, 128]`` and divisible by 16
- When ``indices.shape[1] > 1``: ``indices.shape[0]`` must be exactly 128
- Only available on NeuronCore-v3 (Trainium2) or newer only

``hwdge`` --- hardware DGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.hwdge)

A dedicated **hardware block** on the NeuronCore generates descriptors on
demand, triggered by the Scalar Engine or Sync Engine sequencer. Each TRN2
NeuronCore has **two DGE instances**.

**When to use:**

- Dynamic or semi-dynamic transfer patterns on NeuronCore-v3+.
- When GpSimd Engine is busy with other work (avoids ``swdge`` contention).
- Overlapping descriptor generation with compute via Scalar Engine pipelining.

**Trade-offs:**

- Each hardware-DGE DMA instruction takes approximately **600 ns** to execute.
- Does **not** support indirect (gather/scatter) operations.

Note, for ``dma_copy`` with ``hwdge``, the ``engine`` parameter can optionally
select which sequencer triggers the DGE block:

.. code-block:: python

   # Let Scalar Engine trigger DGE (can overlap with earlier compute)
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.hwdge,
                 engine=nisa.engine.scalar)

   # Let Sync Engine trigger DGE
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.hwdge,
                 engine=nisa.engine.sync)

Only ``nisa.engine.scalar`` and ``nisa.engine.sync`` are valid when
``dge_mode=hwdge``.

Hardware DGE constraints for ``dma_transpose``:

- ``src.shape[0] == 16``
- ``src.shape[-1] % 128 == 0``
- ``src.dtype`` must be 2 bytes (``float16`` / ``bfloat16``)


Mode Selection Summary
------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 20 35

   * - Mode
     - Descriptor Source
     - Min HW
     - Indirect Support
     - Best For
   * - ``none``
     - Host (pre-computed in HBM)
     - Any
     - No
     - Fully static patterns, zero on-device overhead
   * - ``swdge``
     - GpSimd Engine
     - Any (indirect: v3+)
     - Yes
     - Gather/scatter, dynamic indices
   * - ``hwdge``
     - Hardware DGE block
     - NeuronCore-v3+
     - No
     - Dynamic patterns without GpSimd contention
   * - ``unknown``
     - Compiler decides
     - Any
     - Depends
     - Default---recommended unless tuning


How ``.ap()`` Affects DGE Mode
-------------------------------

When you use ``.ap()`` with ``vector_offset`` for indirect (gather/scatter)
access, the DGE mode is constrained to ``swdge``:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - Access Pattern
     - ``dma_copy``
     - ``dma_transpose``
   * - Static (no ``.ap()``, or ``.ap()`` without offsets)
     - Any mode
     - ``none``, ``hwdge``, or compiler-selected
   * - ``.ap()`` with ``scalar_offset``
     - Any mode
     - Any mode
   * - ``.ap()`` with ``vector_offset``
     - ``unknown`` or ``swdge``
     - ``unknown`` or ``swdge``

If you specify ``dge_mode=unknown`` (the default) with ``vector_offset``, the
compiler will automatically select ``swdge``.

The ``name`` Parameter
-----------------------

Both ``dma_copy`` and ``dma_transpose`` accept an optional ``name`` string:

.. code-block:: python

   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor, name="load_weights")

This label appears in profiling traces and compiler debug output. It does not
affect execution. Assigning meaningful names makes it significantly easier to
identify specific DMA operations when analyzing performance with Neuron
profiling tools.


Performance Implications
-------------------------

In essence, the choice comes down to where you want to spend your overhead
budget:

- **``none``** --- Lowest per-transfer latency (descriptor already in HBM), but
  each descriptor consumes HBM bandwidth on first fetch and HBM capacity
  permanently.
- **``swdge``** --- Flexible but uses GpSimd cycles. In GpSimd-bound kernels
  this can become a bottleneck.
- **``hwdge``** --- ~600 ns per instruction. When triggered from Scalar Engine,
  descriptor generation overlaps with earlier compute instructions in the
  pipeline, effectively hiding the cost. Frees GpSimd for other work.
- **``unknown``** --- The compiler applies heuristics to pick the best mode for
  the target and workload. Start here and only override after profiling.

In summary, use ``unknown`` until profiling tells you otherwise, then
switch to the specific mode that addresses the bottleneck you observe.


Code Examples
--------------

Static copy (no DGE)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa

   # Pre-computed descriptors — addresses fully known at compile time
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.none,
                 name="static_load")

Software DGE copy with dynamic address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa

   # GpSimd generates the descriptor at runtime
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.swdge,
                 name="dynamic_load")

Hardware DGE copy
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa

   # Hardware DGE block generates the descriptor (NeuronCore-v3+)
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 dge_mode=nisa.dge_mode.hwdge,
                 name="hwdge_load")

Hardware DGE transpose
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa

   # src must be [16, ...] with last dim divisible by 128, 2-byte dtype
   nisa.dma_transpose(dst=sbuf_tile, src=hbm_tensor,
                      dge_mode=nisa.dge_mode.hwdge,
                      name="hwdge_transpose")

Software DGE indirect transpose (gather + transpose)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa
   import nki.language as nl

   # indices is a 2-D uint32 SBUF tensor; src is on HBM
   # Effectively: dst = src[indices.T.flatten()[:src.shape[0]], :].T
   P, F = 128, 128
   src_ap = hbm_tensor.ap(
       pattern=[[P, F], [1, P]],
       vector_offset=indices,
       indirect_dim=0,
   )
   nisa.dma_transpose(dst=sbuf_tile, src=src_ap,
                      dge_mode=nisa.dge_mode.swdge,
                      name="gather_transpose")

Compiler-selected mode (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.isa as nisa

   # Let the compiler pick the best DGE mode
   nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor,
                 name="auto_load")

   nisa.dma_transpose(dst=sbuf_tile, src=hbm_tensor,
                      name="auto_transpose")
