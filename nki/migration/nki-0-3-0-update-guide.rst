.. meta::
   :description: NKI 0.3.0 Update Guide — update NKI kernels from NKI 0.2.0 to NKI 0.3.0
   :keywords: NKI, Neuron Kernel Interface, update guide, 0.3.0, Trainium, Inferentia

.. _nki-0-3-0-update-guide:

NKI 0.3.0 Update Guide
=======================

For developers with existing NKI 0.2.0 kernels, this document provides guidance on updating to NKI 0.3.0.

NKI 0.3.0 is a significant update to the Neuron Kernel Interface, available in AWS Neuron SDK 2.29.0.
This release moves NKI to General Availability with a new open-source NKI Standard Library (nki-stdlib),
a built-in CPU Simulator, ``nki.language`` APIs, and several API improvements for correctness
and consistency.

This guide is intended for NKI developers updating existing kernels from NKI 0.2.0 to NKI 0.3.0. It covers
new features, deprecated and removed APIs, and breaking changes with before-and-after code examples.

.. note::

   If you are migrating from NKI 0.1.0 (``neuronxcc.nki.*``), first complete the
   :doc:`NKI 0.2.0 Migration Guide <nki-beta2-migration-guide>` before following this guide.

.. contents:: Table of contents
   :local:
   :depth: 2


What's New in NKI 0.3.0
------------------------


NKI Standard Library (nki-stdlib)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 ships with the NKI Standard Library (nki-stdlib), which provides developer-visible code for all
NKI APIs and native language objects (e.g., ``NkiTensor``).


NKI CPU Simulator
~~~~~~~~~~~~~~~~~

NKI 0.3.0 introduces ``nki.simulate(kernel)``, which executes NKI kernels entirely on CPU without requiring
NeuronDevice hardware. The simulator interprets NKI operations using NumPy, producing numerically equivalent
results to on-device execution (with minor floating-point differences due to CPU vs NeuronCore arithmetic).
This enables local development, debugging, and functional correctness testing on any machine — including
laptops and CI environments.

.. note::

   The NKI CPU Simulator is experimental in NKI 0.3.0.

The simulator can be invoked in two ways:

1. **Set the environment variable** ``NKI_SIMULATOR=1`` to run existing kernels without code changes:

.. code-block:: bash

   NKI_SIMULATOR=1 python my_script.py

2. **Wrap the kernel call** with ``nki.simulate``:

.. code-block:: python

   import nki
   import numpy as np

   @nki.jit
   def my_kernel(X, Y):
       ...

   # Run on CPU — no Neuron device needed
   X = np.random.randn(128, 512).astype(np.float16)
   Y = np.zeros((128, 512), dtype=np.float16)
   nki.simulate(my_kernel)(X, Y)


``nki.typing`` Module
~~~~~~~~~~~~~~~~~~~~~

A new module for type-annotating kernel tensor parameters. Use ``nt.tensor[shape]`` to declare expected
tensor shapes:

.. code-block:: python

   import nki.typing as nt

   @nki.jit
   def my_kernel(
       X: nt.tensor[128, 512],
       Y: nt.tensor[128, 512]
   ):
       ...


New ``nki.isa`` APIs
~~~~~~~~~~~~~~~~~~~~

* ``nki.isa.exponential`` — Dedicated exponential instruction with max subtraction, faster than ``nisa.activation(op=nl.exp)`` and useful for Softmax calculation. Trn3 (NeuronCore-v4) only.


New ``nki.collectives`` APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``nki.collectives.all_to_all_v`` — Variable-length all-to-all collective. Unlike ``all_to_all``, uses a metadata tensor to specify per-rank send/recv counts.


Matmul Accumulation
~~~~~~~~~~~~~~~~~~~

``nc_matmul`` and ``nc_matmul_mx`` now have an ``accumulate`` parameter that controls whether the operation
overwrites or accumulates on the destination PSUM tile. The default (``accumulate=None``) auto-detects:
the first write to a PSUM location overwrites, and subsequent writes accumulate. This matches NKI 0.2.0
behavior.

.. code-block:: python

   nisa.nc_matmul(dst, stationary, moving, accumulate=True)
   nisa.nc_matmul_mx(dst, stationary, moving, stat_scale, mov_scale, accumulate=True)


Address Placement
~~~~~~~~~~~~~~~~~

The ``address`` parameter was added to ``nki.language.ndarray`` as an optional parameter for explicit
memory placement.

.. code-block:: python

   buf = nl.ndarray((128, 512), dtype=nl.float16, address=(p_off, f_off))  # explicit placement


``nki.language`` APIs
~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 introduces ``nki.language`` APIs as convenience wrappers around ``nki.isa`` APIs. These
include operations such as ``nl.load``, ``nl.store``, ``nl.copy``, ``nl.matmul``, ``nl.transpose``,
``nl.softmax``, and other high-level operations that map to one or more ``nki.isa`` calls.

.. note::

   The ``nki.language`` convenience APIs are experimental in NKI 0.3.0.


Deprecated and Removed APIs
----------------------------


``nki.isa.tensor_copy_dynamic_src`` / ``nki.isa.tensor_copy_dynamic_dst``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deprecated and scheduled for removal. Use ``nisa.tensor_copy()`` with ``.ap()`` and ``scalar_offset`` instead.


``nki.jit(platform_target=...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``platform_target`` parameter is deprecated. Set the target platform via the
``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable instead.

.. important::

   This is a breaking change. Passing ``platform_target`` to ``@nki.jit`` raises an error in NKI 0.3.0.


``nki.jit(mode=...)``
~~~~~~~~~~~~~~~~~~~~~

The ``mode`` parameter is deprecated and ignored. The NKI Compiler now inspects the kernel arguments to
detect the appropriate machine learning framework automatically:

1. **Torch tensors**: uses TorchXLA integration.
2. **JAX arrays**: uses JAX integration.
3. **NumPy arrays**: runs the kernel in standalone mode without a machine learning framework.

To run the kernel in the CPU simulator, set the environment variable ``NKI_SIMULATOR=1``, or wrap the
kernel call in ``nki.simulate``.

.. important::

   This is a breaking change. Code that passes ``mode=`` to ``@nki.jit`` should remove the parameter.


API Breaking Changes
--------------------

This section describes each breaking change with before-and-after code examples.


``nisa.dma_copy`` — Reading from PSUM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nisa.dma_copy`` no longer supports reading directly from PSUM. Copy the PSUM tensor to SBUF first
using ``nisa.tensor_copy``.

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_copy(dst=hbm_tensor, src=psum_tensor[0:TILE, 0:N])

   # NKI 0.3.0
   sbuf_temp = nl.ndarray((TILE, PSUM_SIZE), dtype=nl.float32, buffer=nl.sbuf)
   nisa.tensor_copy(dst=sbuf_temp[0:TILE, 0:N], src=psum_tensor[0:TILE, 0:N])
   nisa.dma_copy(dst=hbm_tensor, src=sbuf_temp[0:TILE, 0:N])


``nisa.dma_copy`` — ``dge_mode`` Type Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 enforces that source and destination element types must match when using
``dge_mode=dge_mode.hwdge``. NKI 0.2.0 did not validate this, allowing mismatched types to pass silently.

The DMA hardware moves raw bytes — HWDGE generates descriptors without interpreting data content, so no
type casting occurs. To reinterpret data as a different type, use ``.view()`` to match types before the copy.

.. code-block:: python

   # NKI 0.2.0 (no validation, undefined behavior)
   nisa.dma_copy(dst=dst_f4, src=src_ui16, dge_mode=nisa.dge_mode.hwdge)

   # NKI 0.3.0 — use .view() to reinterpret
   nisa.dma_copy(dst=dst_f4, src=src_ui16.view(nl.float4_e2m1fn_x4), dge_mode=nisa.dge_mode.hwdge)

Alternatively, use ``dge_mode.swdge`` or ``dge_mode.none`` if type casting is intended.


``nisa.dma_copy`` — ``dst_rmw_op`` and ``unique_indices`` Removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nisa.dma_copy`` no longer supports read-modify-write operations. The ``dst_rmw_op`` and ``unique_indices``
parameters have been removed. Use ``nisa.dma_compute`` instead.

.. code-block:: python

   # NKI 0.2.0 — simple read-modify-write
   nisa.dma_copy(dst, src, dst_rmw_op=nl.add)

   # NKI 0.3.0 — use dma_compute
   nisa.dma_compute(dst, [src], reduce_op=nl.add)

For accumulation loops with indirect indexing:

.. code-block:: python

   # NKI 0.2.0
   for k_idx in range(K):
       dst_rmw_op = None if k_idx == 0 else nl.add
       nisa.dma_copy(
           src=input.ap(...),
           dst=reduced_sb[:, :],
           dst_rmw_op=dst_rmw_op,
           unique_indices=True,
       )

   # NKI 0.3.0 — split into dma_copy + dma_compute
   for k_idx in range(K):
       src_access = input.ap(...)
       if k_idx == 0:
           nisa.dma_copy(dst=reduced_sb[:, :], src=src_access)
       else:
           nisa.dma_compute(
               dst=reduced_sb[:, :],
               srcs=[src_access, reduced_sb[:, :]],
               reduce_op=nl.add,
               unique_indices=True,
           )


``nisa.memset`` — Strict Type Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 enforces that the ``value`` argument must match the destination tensor's dtype. NKI 0.2.0 silently
cast float values to the destination type. For integer-typed tensors, pass an integer literal.

.. code-block:: python

   # NKI 0.2.0
   buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=2.0)

   # NKI 0.3.0
   buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=2)


``nisa.tensor_reduce`` — Axis Handling Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 fixes incorrect axis handling that existed in NKI 0.2.0. NKI 0.2.0 incorrectly allowed ``axis=1`` to
refer to the last free dimension even for 3D/4D tensors. NKI 0.3.0 corrects this so that axis values
correspond to the actual tensor dimensions.

Kernels that relied on the NKI 0.2.0 behavior (e.g., using ``axis=1`` to mean the last dimension of a 3D/4D
tensor) will produce errors in NKI 0.3.0.


``nisa.dma_compute`` — Parameter Reorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scales`` and ``reduce_op`` parameters swapped positions. ``scales`` is now optional, and
``unique_indices`` was added (moved from ``dma_copy``).

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_compute(dst, srcs, scales, reduce_op)

   # NKI 0.3.0
   nisa.dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True)


``nisa.sendrecv`` — ``dma_engine`` Enum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boolean ``use_gpsimd_dma`` parameter is replaced by the ``dma_engine`` enum.

.. code-block:: python

   # NKI 0.2.0
   nisa.sendrecv(..., use_gpsimd_dma=True)

   # NKI 0.3.0
   from nki.isa import dma_engine
   nisa.sendrecv(..., dma_engine=dma_engine.gpsimd_dma)
   nisa.sendrecv(..., dma_engine=dma_engine.dma)      # was use_gpsimd_dma=False


``nisa.affine_select`` — ``offset`` Parameter Moved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``offset`` parameter moved from the 3rd positional argument to a keyword argument with default ``0``.
Existing positional call sites will break.

.. code-block:: python

   # NKI 0.2.0
   nisa.affine_select(dst, pattern, offset, channel_multiplier, on_true, on_false)

   # NKI 0.3.0
   nisa.affine_select(dst, pattern, channel_multiplier, on_true, on_false, offset=offset)


``nisa.register_move`` — ``imm`` Renamed to ``src``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``imm`` parameter has been renamed to ``src`` and now accepts a ``VirtualRegister`` instead of a
compile-time constant. To move a compile-time constant into a register, first allocate a register with
the constant value.

.. code-block:: python

   # NKI 0.2.0
   nisa.register_move(dst, imm=42)

   # NKI 0.3.0
   src = nisa.register_alloc(x=42)
   nisa.register_move(dst, src=src)


Collectives — ``num_channels`` Removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_channels`` removed from ``collective_permute_implicit_current_processing_rank_id``. The high-level
``collective_permute_implicit()`` now accepts a ``channel_ids`` list directly.

.. code-block:: python

   # NKI 0.2.0
   rank_id = ncc.collective_permute_implicit_current_processing_rank_id(
       iteration_id=0, channel_id=ch, num_channels=N, replica_group=rg
   )

   # NKI 0.3.0
   rank_id = ncc.collective_permute_implicit_current_processing_rank_id(
       iteration_id=0, channel_id=ch, replica_group=rg
   )

   ncc.collective_permute_implicit(
       srcs_by_channel=[[src0], [src1]],
       dsts_by_channel=[[dst0], [dst1]],
       replica_group=rg,
       channel_ids=[0, 1],  # replaces num_channels=2
   )


Output Tensors Must Use ``nl.shared_hbm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All kernel output (return) tensors must be allocated with ``buffer=nl.shared_hbm``. Using ``nl.hbm``
for output tensors will cause compilation failures.

.. code-block:: python

   # NKI 0.2.0
   output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.hbm)

   # NKI 0.3.0
   output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.shared_hbm)


Integer Enum Constants No Longer Supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw integer values (e.g., ``dge_mode=2``) are no longer accepted for enum parameters. Use the named enum
members instead: ``nki.isa.engine``, ``nki.isa.dge_mode``, ``nki.isa.oob_mode``, ``nki.isa.reduce_cmd``,
and ``nki.isa.nc_version``.

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=2)

   # NKI 0.3.0
   nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=nisa.dge_mode.hwdge)


String Buffer Names No Longer Supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nl.ndarray``, ``nl.zeros``, and other creation ops no longer accept strings for the ``buffer`` parameter.
Use buffer objects from ``nki.language`` instead.

.. code-block:: python

   # NKI 0.2.0
   buf = nl.ndarray((128, 512), dtype=nl.float16, buffer='sbuf')

   # NKI 0.3.0
   buf = nl.ndarray((128, 512), dtype=nl.float16)  # buffer defaults to sbuf
   buf = nl.ndarray((128, 512), dtype=nl.float16, buffer=nl.sbuf)

.. list-table:: Buffer type mapping
   :header-rows: 1
   :widths: 50 50

   * - NKI 0.2.0 (string)
     - NKI 0.3.0 (object)
   * - ``"sbuf"``
     - ``nl.sbuf``
   * - ``"psum"``
     - ``nl.psum``
   * - ``"hbm"``
     - ``nl.hbm``
   * - ``"private_hbm"``
     - ``nl.private_hbm``
   * - ``"shared_hbm"``
     - ``nl.shared_hbm``


``nki.isa.dma_engine`` Alias Repurposed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.2.0 ``nki.isa.dma_engine`` module-level alias was unused and did not map correctly to a valid engine.
In NKI 0.3.0, it has been replaced with the ``nki.isa.dma_engine`` enum, which provides explicit control
over DMA transfer engines (``dma_engine.dma`` for shared DMA, ``dma_engine.gpsimd_dma`` for GPSIMD's
internal DMA engine).


Language Restrictions
---------------------

The NKI 0.3.0 compiler has stricter validation. The following patterns require changes for NKI 0.3.0.


Remove Keyword-Only Argument Separator (``*``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.3.0 compiler does not support the ``*`` separator in kernel function signatures. Move all
parameters with defaults to the end of the signature.

.. code-block:: python

   # NKI 0.2.0
   @nki.jit
   def my_kernel(X: nl.ndarray, *, flag: bool = True, scale: float = 1.0):
       ...

   # NKI 0.3.0
   @nki.jit
   def my_kernel(X: nl.ndarray, flag: bool = True, scale: float = 1.0):
       ...


Replace ``is`` / ``is not`` with ``==`` / ``!=``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.3.0 compiler does not support Python's ``is`` / ``is not`` operators. These operators check
object identity, which is not meaningful during NKI compilation tracing. Use ``==`` / ``!=`` instead.

.. code-block:: python

   # NKI 0.2.0
   if some_flag is True:
       ...

   # NKI 0.3.0
   if some_flag == True:
       ...


Replace List Kernel Arguments with Tuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.3.0 compiler does not support ``list`` as a kernel argument type. Convert list arguments to
tuples at the call site.

Tuples are immutable and hashable, which more accurately reflects the semantics of compiled kernels and enables 
the compiler to cache compilations based on the kernel's arguments.

.. code-block:: python

   # NKI 0.2.0
   @nki.jit
   def my_kernel(img, in_perm, stride=[1, 1]):
       ...
   my_kernel(img, in_perm=[0, 3, 1, 2], stride=[1, 1])

   # NKI 0.3.0
   @nki.jit
   def my_kernel(img, in_perm, stride=(1, 1)):
       ...
   my_kernel(img, in_perm=(0, 3, 1, 2), stride=(1, 1))


API Improvements
----------------

These changes improve correctness or usability but are non-breaking for most kernels.


``nisa.memset`` — x4 Packed Type Restriction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x4 packed types (``float8_e4m3fn_x4``, ``float8_e5m2_x4``, ``float4_e2m1fn_x4``) now enforce ``value=0``.
The ISA memset instruction fills the destination with a single u32 value and has no notion of the
sub-elements packed inside, so only zero is valid. To initialize x4 packed tensors with non-zero values,
use ``nisa.dma_copy`` to load pre-computed x4 data from an HBM kernel argument.

.. code-block:: python

   # Zero-fill works directly
   buf = nl.ndarray((128, 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=0)

   # Non-zero: pass pre-computed x4 data as a kernel argument from HBM
   # and use nisa.dma_copy to load it into SBUF
   nisa.dma_copy(dst=buf, src=precomputed_x4_hbm_tensor)


``nisa.range_select`` — Parameter Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.2.0 silently overrode ``on_false_value`` to ``FP32_MIN`` and ``reduce_cmd`` to ``reset_reduce``,
regardless of user input. In NKI 0.3.0:

* ``reduce_cmd`` now works as expected (default ``reset_reduce``)
* ``on_false_value`` must be ``FP32_MIN`` due to hardware constraints, but is now documented as a
  constraint rather than silently ignored


Parameter Default Value Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following default values changed in NKI 0.3.0:

* ``nki.isa.iota`` — ``offset`` is now optional with a default of ``0``
* ``nki.isa.core_barrier`` — ``engine`` default changed from ``unknown`` to ``gpsimd`` (no behavioral change)
* ``nki.language.num_programs`` — ``axes`` default changed from ``None`` to ``0``
* ``nki.language.program_id`` — ``axis`` now has a default value of ``0``
* ``nki.language.ndarray`` — ``buffer`` default changed from ``None`` to ``nl.sbuf``
* ``nki.language.zeros`` — ``buffer`` default changed from ``None`` to ``nl.sbuf``
* ``nki.language.sequential_range`` — ``stop`` and ``step`` now have default values (``None`` and ``1``)
