.. _nki_migration_home:

.. meta::
    :description: NKI Migration Guides for upgrading NKI kernels between versions, newest release on top.
    :keywords: NKI, AWS Neuron, Migration, Upgrade, Update Guide, 0.5.0, 0.4.0, 0.3.0, 0.2.0

====================
NKI Migration Guides
====================

This page consolidates all NKI version-to-version migration guidance in one place, ordered
**newest release first**. Find the version you are upgrading *from*, then work upward through
each subsequent section until you reach your target version.

.. list-table:: NKI version to Neuron SDK mapping
   :header-rows: 1
   :widths: 25 25 50

   * - NKI version
     - Neuron SDK
     - Migration section
   * - 0.5.0
     - 2.31.0
     - :ref:`nki-migrate-0-4-to-0-5`
   * - 0.4.0
     - 2.30.0
     - :ref:`nki-migrate-0-3-to-0-4`
   * - 0.3.0
     - 2.29.0
     - :ref:`nki-0-3-0-update-guide`
   * - 0.2.0
     - 2.28.0 / 2.27.0
     - :ref:`nki-migration-guide`

.. note::

   Block dimension support (a partition dimension placed anywhere other than the left-most
   position) was removed. If your kernels still use block dimensions, see
   :ref:`nki_block_dimension_migration_guide`, which applies regardless of which version you
   are upgrading from.

.. contents:: On this page
   :local:
   :depth: 2


.. _nki-migrate-0-4-to-0-5:

Migrating from NKI 0.4.0 to NKI 0.5.0
=====================================

NKI 0.5.0 ships in AWS Neuron SDK 2.31.0. It adds the ``float8_e8m0fnu`` MX scale dtype,
tensor indirection (gather/scatter) for on-chip compute operations, and an expanded set of
:class:`~nki.language.NkiTensor` view and query methods. Most kernels require no changes; the
items below are the ones that can require action.

Breaking changes
-----------------

``nki.language.rsqrt`` — PSUM inputs no longer supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The experimental ``nl.rsqrt`` now runs on the GpSimd Engine for higher numerical precision and
no longer accepts PSUM inputs. If you need a PSUM input or Scalar Engine throughput, call
``nki.isa.activation`` directly with the reciprocal-square-root op instead.

``nki.language.NkiTensor.reshape`` — element count now validated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reshape`` now asserts that the total element count is preserved. Reshapes that previously
passed silently while changing the number of elements now raise a clear error at trace time.
Correct any reshape whose source and destination element counts differ.

Deprecation notice: SPMD launch grid for LNC2 kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launching an LNC2 kernel currently relies on an SPMD launch grid whose dimension matches the LNC
degree. **The behavior does not change in NKI 0.5.0** — this is advance notice only. In a future
release:

1. the SPMD launch grid will no longer be required to launch an LNC2 kernel;
2. passing an SPMD dimension that differs from the LNC degree in use will raise a compile-time
   error; and
3. the kernel will always be specialized across all physical NeuronCores (PNCs) in the LNC,
   whether or not an SPMD dimension is specified.

Kernels that continue to pass a matching SPMD grid keep working for backwards compatibility. See
:doc:`LNC Overview </nki/get-started/about/lnc>`.

For the full list of new features in NKI 0.5.0, see the :ref:`NKI 0.5.0 release notes <nki-2-31-0-rn>`.


.. _nki-migrate-0-3-to-0-4:

Migrating from NKI 0.3.0 to NKI 0.4.0
=====================================

NKI 0.4.0 ships in AWS Neuron SDK 2.30.0. It adds ``trn3`` Scalar Engine APIs
(:func:`~nki.isa.activate2`), new opcodes, and bytes-aware ``tile_size`` constants. The following
changes can require action.

Breaking changes
-----------------

``nisa.dma_transpose`` — ``dst`` shape and rank now enforced exactly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nisa.dma_transpose`` now enforces that ``dst.shape`` matches the transposed ``src.shape``
exactly, including rank. Previously a lower-rank ``dst.shape`` was silently padded to match a
higher-rank ``src.shape`` (for example, a 3D ``dst`` against a 4D ``src``). The compiler now raises
an assertion error if the ranks differ.

To migrate, either match the ``dst`` rank to the ``src`` rank, or use a ``src`` and ``axes`` of the
same rank as the intended ``dst``:

.. code-block:: python

   # Option A — match dst rank to a 4D src
   dst = nl.ndarray((128, 1, 1, 4096), dtype=src.dtype, buffer=nl.shared_hbm)
   nisa.dma_transpose(dst=dst, src=src_4d, axes=(3, 1, 2, 0))

   # Option B — use a 3D src + axes of the same rank as the intended dst
   nisa.dma_transpose(dst=dst_3d, src=src_3d, axes=(2, 1, 0))

``neuronxcc.nki.*`` inside kernels — now a compilation error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the deprecated ``neuronxcc.nki.*`` namespace inside NKI kernels now raises a **compilation
error** instead of a warning. To migrate, move to the ``nki.*`` namespace following
:ref:`nki-migration-guide`.

Removed APIs
------------

``nki.isa.tensor_copy_dynamic_src`` / ``nki.isa.tensor_copy_dynamic_dst`` — removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These APIs were **removed** in NKI 0.4.0 (they were deprecated in 0.3.0). Use
``nisa.tensor_copy()`` with ``.ap()`` and ``scalar_offset`` instead. See
:doc:`nki.isa.tensor_copy </nki/api/generated/nki.isa.tensor_copy>`.

Deprecated APIs
---------------

``nki.language.tile_size.total_available_sbuf_size`` — deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite the name, this attribute returns the usable SBUF free dimension *per partition*, not total
SBUF capacity. It continues to work and returns the same value, but is deprecated. Use
``tile_size.sbuf_size_bytes`` for total SBUF capacity across all partitions, or
``tile_size.sbuf_fmax_bytes`` for the per-partition size. See
:doc:`nki.language.tile_size </nki/api/generated/nki.language.tile_size>`.

Behavior change: CPU simulator precision default
------------------------------------------------

``NKI_PRECISE_FP=1`` is now the **default** for CPU simulation. Low-precision dtypes
(``bfloat16``, ``float8``) are modeled accurately instead of being approximated with ``float32``,
producing simulator results closer to hardware. If you depended on the previous approximate
behavior, set ``NKI_PRECISE_FP=0`` to restore it. See
:doc:`nki.simulate </nki/api/generated/nki.simulate>`.


.. _nki-0-3-0-update-guide:

Migrating from NKI 0.2.0 to NKI 0.3.0
=====================================

For developers with existing NKI 0.2.0 kernels, this section provides guidance on updating to
NKI 0.3.0 (AWS Neuron SDK 2.29.0). NKI 0.3.0 moves NKI to General Availability with a new
open-source NKI Standard Library (nki-stdlib), a built-in CPU Simulator, ``nki.language`` APIs, and
several API improvements for correctness and consistency.

.. note::

   If you are migrating from NKI 0.1.0 (``neuronxcc.nki.*``), first complete
   :ref:`nki-migration-guide` before following this section.

What's new in NKI 0.3.0
-----------------------

NKI Standard Library (nki-stdlib)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 ships with the NKI Standard Library (nki-stdlib), which provides developer-visible code
for all NKI APIs and native language objects (e.g., ``NkiTensor``).

NKI CPU Simulator
~~~~~~~~~~~~~~~~~~

NKI 0.3.0 introduces ``nki.simulate(kernel)``, which executes NKI kernels entirely on CPU without
requiring NeuronDevice hardware. The simulator interprets NKI operations using NumPy, producing
numerically equivalent results to on-device execution (with minor floating-point differences due to
CPU vs NeuronCore arithmetic). This enables local development, debugging, and functional
correctness testing on any machine — including laptops and CI environments.

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

``nki.typing`` module
~~~~~~~~~~~~~~~~~~~~~~

A new module for type-annotating kernel tensor parameters. Use ``nt.tensor[shape]`` to declare
expected tensor shapes:

.. code-block:: python

   import nki.typing as nt

   @nki.jit
   def my_kernel(
       X: nt.tensor[128, 512],
       Y: nt.tensor[128, 512]
   ):
       ...

New ``nki.isa`` APIs
~~~~~~~~~~~~~~~~~~~~~

* ``nki.isa.exponential`` — Dedicated exponential instruction with max subtraction, faster than
  ``nisa.activation(op=nl.exp)`` and useful for Softmax calculation. Trn3 (NeuronCore-v4) only.

New ``nki.collectives`` APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``nki.collectives.all_to_all_v`` — Variable-length all-to-all collective. Unlike ``all_to_all``,
  uses a metadata tensor to specify per-rank send/recv counts.

Matmul accumulation
~~~~~~~~~~~~~~~~~~~~~

``nc_matmul`` and ``nc_matmul_mx`` now have an ``accumulate`` parameter that controls whether the
operation overwrites or accumulates on the destination PSUM tile. The default (``accumulate=None``)
auto-detects: the first write to a PSUM location overwrites, and subsequent writes accumulate. This
matches NKI 0.2.0 behavior.

.. code-block:: python

   nisa.nc_matmul(dst, stationary, moving, accumulate=True)
   nisa.nc_matmul_mx(dst, stationary, moving, stat_scale, mov_scale, accumulate=True)

``nki.language`` APIs
~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 introduces ``nki.language`` APIs as convenience wrappers around ``nki.isa`` APIs. These
include operations such as ``nl.load``, ``nl.store``, ``nl.copy``, ``nl.matmul``, ``nl.transpose``,
``nl.softmax``, and other high-level operations that map to one or more ``nki.isa`` calls.

.. note::

   The ``nki.language`` convenience APIs are experimental in NKI 0.3.0.

Deprecated and removed APIs
---------------------------

``nki.isa.tensor_copy_dynamic_src`` / ``nki.isa.tensor_copy_dynamic_dst``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deprecated in NKI 0.3.0 and **removed in NKI 0.4.0**. Use ``nisa.tensor_copy()`` with ``.ap()``
and ``scalar_offset`` instead. If you are upgrading past 0.3.0, migrate off these now — see
:ref:`nki-migrate-0-3-to-0-4`.

``nki.jit(platform_target=...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``platform_target`` parameter is deprecated. Set the target platform via the
``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable instead.

.. important::

   This is a breaking change. Passing ``platform_target`` to ``@nki.jit`` raises an error in NKI 0.3.0.

``nki.jit(mode=...)``
~~~~~~~~~~~~~~~~~~~~~~

The ``mode`` parameter is deprecated and ignored. The NKI Compiler now inspects the kernel
arguments to detect the appropriate machine learning framework automatically:

1. **Torch tensors**: uses TorchXLA integration.
2. **JAX arrays**: uses JAX integration.
3. **NumPy arrays**: runs the kernel in standalone mode without a machine learning framework.

To run the kernel in the CPU simulator, set the environment variable ``NKI_SIMULATOR=1``, or wrap
the kernel call in ``nki.simulate``.

.. important::

   This is a breaking change. Code that passes ``mode=`` to ``@nki.jit`` should remove the parameter.

API breaking changes
--------------------

This section describes each breaking change with before-and-after code examples.

``nisa.dma_copy`` — reading from PSUM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nisa.dma_copy`` no longer supports reading directly from PSUM. Copy the PSUM tensor to SBUF first
using ``nisa.tensor_copy``.

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_copy(dst=hbm_tensor, src=psum_tensor[0:TILE, 0:N])

   # NKI 0.3.0
   sbuf_temp = nl.ndarray((TILE, PSUM_SIZE), dtype=nl.float32, buffer=nl.sbuf)
   nisa.tensor_copy(dst=sbuf_temp[0:TILE, 0:N], src=psum_tensor[0:TILE, 0:N])
   nisa.dma_copy(dst=hbm_tensor, src=sbuf_temp[0:TILE, 0:N])

``nisa.dma_copy`` — ``dge_mode`` type matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 enforces that source and destination element types must match when using
``dge_mode=dge_mode.hwdge``. NKI 0.2.0 did not validate this, allowing mismatched types to pass
silently.

The DMA hardware moves raw bytes — HWDGE generates descriptors without interpreting data content, so
no type casting occurs. To reinterpret data as a different type, use ``.view()`` to match types
before the copy.

.. code-block:: python

   # NKI 0.2.0 (no validation, undefined behavior)
   nisa.dma_copy(dst=dst_f4, src=src_ui16, dge_mode=nisa.dge_mode.hwdge)

   # NKI 0.3.0 — use .view() to reinterpret
   nisa.dma_copy(dst=dst_f4, src=src_ui16.view(nl.float4_e2m1fn_x4), dge_mode=nisa.dge_mode.hwdge)

Alternatively, use ``dge_mode.swdge`` or ``dge_mode.none`` if type casting is intended.

``nisa.dma_copy`` — ``dst_rmw_op`` and ``unique_indices`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nisa.dma_copy`` no longer supports read-modify-write operations. The ``dst_rmw_op`` and
``unique_indices`` parameters have been removed. Use ``nisa.dma_compute`` instead.

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

``nisa.memset`` — strict type matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 enforces that the ``value`` argument must match the destination tensor's dtype. NKI 0.2.0
silently cast float values to the destination type. For integer-typed tensors, pass an integer
literal.

.. code-block:: python

   # NKI 0.2.0
   buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=2.0)

   # NKI 0.3.0
   buf = nl.ndarray((128, 128), dtype=nl.int32, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=2)

``nisa.tensor_reduce`` — axis handling fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.3.0 fixes incorrect axis handling that existed in NKI 0.2.0. NKI 0.2.0 incorrectly allowed
``axis=1`` to refer to the last free dimension even for 3D/4D tensors. NKI 0.3.0 corrects this so
that axis values correspond to the actual tensor dimensions.

Kernels that relied on the NKI 0.2.0 behavior (e.g., using ``axis=1`` to mean the last dimension of
a 3D/4D tensor) will produce errors in NKI 0.3.0.

``nisa.dma_compute`` — parameter reorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``scales`` and ``reduce_op`` parameters swapped positions. ``scales`` is now optional, and
``unique_indices`` was added (moved from ``dma_copy``).

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_compute(dst, srcs, scales, reduce_op)

   # NKI 0.3.0
   nisa.dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True)

``nisa.sendrecv`` — ``dma_engine`` enum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boolean ``use_gpsimd_dma`` parameter is replaced by the ``dma_engine`` enum.

.. code-block:: python

   # NKI 0.2.0
   nisa.sendrecv(..., use_gpsimd_dma=True)

   # NKI 0.3.0
   from nki.isa import dma_engine
   nisa.sendrecv(..., dma_engine=dma_engine.gpsimd_dma)
   nisa.sendrecv(..., dma_engine=dma_engine.dma)      # was use_gpsimd_dma=False

``nisa.affine_select`` — ``offset`` parameter moved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``offset`` parameter moved from the 3rd positional argument to a keyword argument with default
``0``. Existing positional call sites will break.

.. code-block:: python

   # NKI 0.2.0
   nisa.affine_select(dst, pattern, offset, channel_multiplier, on_true, on_false)

   # NKI 0.3.0
   nisa.affine_select(dst, pattern, channel_multiplier, on_true, on_false, offset=offset)

``nisa.register_move`` — ``imm`` renamed to ``src``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``imm`` parameter has been renamed to ``src`` and now accepts a ``VirtualRegister`` instead of a
compile-time constant. To move a compile-time constant into a register, first allocate a register
with the constant value.

.. code-block:: python

   # NKI 0.2.0
   nisa.register_move(dst, imm=42)

   # NKI 0.3.0
   src = nisa.register_alloc(42)
   nisa.register_move(dst, src)

Collectives — ``num_channels`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``num_channels`` removed from ``collective_permute_implicit_current_processing_rank_id``. The
high-level ``collective_permute_implicit()`` now accepts a ``channel_ids`` list directly.

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

Output tensors must use ``nl.shared_hbm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All kernel output (return) tensors must be allocated with ``buffer=nl.shared_hbm``. Using
``nl.hbm`` for output tensors will cause compilation failures.

.. code-block:: python

   # NKI 0.2.0
   output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.hbm)

   # NKI 0.3.0
   output = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.shared_hbm)

Integer enum constants no longer supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw integer values (e.g., ``dge_mode=2``) are no longer accepted for enum parameters. Use the named
enum members instead: ``nki.isa.engine``, ``nki.isa.dge_mode``, ``nki.isa.oob_mode``,
``nki.isa.reduce_cmd``, and ``nki.isa.nc_version``.

.. code-block:: python

   # NKI 0.2.0
   nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=2)

   # NKI 0.3.0
   nisa.dma_copy(src=src_tensor, dst=dst_tensor, dge_mode=nisa.dge_mode.hwdge)

String buffer names no longer supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``nl.ndarray``, ``nl.zeros``, and other creation ops no longer accept strings for the ``buffer``
parameter. Use buffer objects from ``nki.language`` instead.

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

``nki.isa.dma_engine`` alias repurposed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.2.0 ``nki.isa.dma_engine`` module-level alias was unused and did not map correctly to a
valid engine. In NKI 0.3.0, it has been replaced with the ``nki.isa.dma_engine`` enum, which
provides explicit control over DMA transfer engines (``dma_engine.dma`` for shared DMA,
``dma_engine.gpsimd_dma`` for GPSIMD's internal DMA engine).

Language restrictions
---------------------

The NKI 0.3.0 compiler has stricter validation. The following patterns require changes for NKI
0.3.0.

Remove keyword-only argument separator (``*``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.3.0 compiler does not support Python's ``is`` / ``is not`` operators. These operators
check object identity, which is not meaningful during NKI compilation tracing. Use ``==`` / ``!=``
instead.

.. code-block:: python

   # NKI 0.2.0
   if some_flag is True:
       ...

   # NKI 0.3.0
   if some_flag == True:
       ...

Replace list kernel arguments with tuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NKI 0.3.0 compiler does not support ``list`` as a kernel argument type. Convert list arguments
to tuples at the call site.

Tuples are immutable and hashable, which more accurately reflects the semantics of compiled kernels
and enables the compiler to cache compilations based on the kernel's arguments.

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

API improvements
----------------

These changes improve correctness or usability but are non-breaking for most kernels.

``nisa.memset`` — x4 packed type restriction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x4 packed types (``float8_e4m3fn_x4``, ``float8_e5m2_x4``, ``float4_e2m1fn_x4``) now enforce
``value=0``. The ISA memset instruction fills the destination with a single u32 value and has no
notion of the sub-elements packed inside, so only zero is valid. To initialize x4 packed tensors
with non-zero values, use ``nisa.dma_copy`` to load pre-computed x4 data from an HBM kernel
argument.

.. code-block:: python

   # Zero-fill works directly
   buf = nl.ndarray((128, 128), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
   nisa.memset(dst=buf, value=0)

   # Non-zero: pass pre-computed x4 data as a kernel argument from HBM
   # and use nisa.dma_copy to load it into SBUF
   nisa.dma_copy(dst=buf, src=precomputed_x4_hbm_tensor)

``nisa.range_select`` — parameter fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.2.0 silently overrode ``on_false_value`` to ``FP32_MIN`` and ``reduce_cmd`` to
``reset_reduce``, regardless of user input. In NKI 0.3.0:

* ``reduce_cmd`` now works as expected (default ``reset_reduce``)
* ``on_false_value`` must be ``FP32_MIN`` due to hardware constraints, but is now documented as a
  constraint rather than silently ignored

Parameter default value updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following default values changed in NKI 0.3.0:

* ``nki.isa.iota`` — ``offset`` is now optional with a default of ``0``
* ``nki.isa.core_barrier`` — ``engine`` default changed from ``unknown`` to ``gpsimd`` (no behavioral change)
* ``nki.language.num_programs`` — ``axes`` default changed from ``None`` to ``0``
* ``nki.language.program_id`` — ``axis`` now has a default value of ``0``
* ``nki.language.ndarray`` — ``buffer`` default changed from ``None`` to ``nl.sbuf``
* ``nki.language.zeros`` — ``buffer`` default changed from ``None`` to ``nl.sbuf``
* ``nki.language.sequential_range`` — ``stop`` and ``step`` now have default values (``None`` and ``1``)


.. _nki-migration-guide:

Migrating from NKI 0.1.0 to NKI 0.2.0
=====================================

This section covers best practices for migrating NKI kernels from the legacy ``neuronxcc.nki.*``
namespace to the new ``nki.*`` namespace which uses the new NKI Compiler. See
:ref:`nki_compiler_about` for more in-depth information.

.. note::

   Usage of the ``neuronxcc.nki.*`` namespace inside NKI kernels raises a **compilation error** as
   of NKI 0.4.0 (previously a warning). Completing this migration is required to use recent NKI
   versions.

Background: NKI has a compiler!
-------------------------------

As of Release 2.27, NKI now has a new standalone compiler. The syntax of NKI remains a subset of
Python. This means you can largely use Python syntax when writing NKI kernels. However, it is
important to remember that your NKI functions are compiled by the NKI Compiler and not evaluated by
the Python interpreter. The goal is to offer a better programming experience with more precise error
messages.

With the NKI Compiler, we have chosen to define the NKI meta-programming language as a subset of
Python. This means that all NKI programs are valid Python programs, but not all Python programs are
valid NKI programs. The delineation is the ``nki.jit`` decorator. Just as before, you mark your NKI
kernels with the ``nki.jit`` decorator. However, unlike before, the functions under this decorator
will be passed to the NKI Compiler and not be evaluated by the Python interpreter.

.. code-block:: python

   def a_function(x, y, z):
     # this is Python code
     ...

   @nki.jit
   def kernel(x, y, z):
     # this is NKI code
     ...

If you use Python features within a NKI kernel that are not supported, the NKI Compiler will give an
error. The goal is that programming in NKI is intuitive and convenient and all of the features you
need are available and behave as expected. However, if you find some curious errors or confusing
behavior, reach out to us on the NKI Samples repository on AWS Neuron GitHub.

This section is intended for experienced NKI developers who are looking to migrate their existing
kernels to the NKI 0.2.0 compiler. Most code snippets below are assumed to be executed within a
valid NKI kernel.

Key migration items
--------------------

These are the key items to migrate an existing kernel to the NKI 0.2.0 Compiler.

What new features are available in NKI 0.2.0?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* A new namespace for NKI 0.2.0, ``nki.*``
* ``device_print`` is available to inspect tensor values
* The behavior of loops and branching is consistent with regular Python
* Lists and dictionaries are available and their behavior in loops is consistent with regular Python
* Direct allocation APIs have been reworked

What features in ``neuronxcc.nki.*`` are not available in ``nki.*``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``arange`` has been removed, use slicing or :ref:`nki-aps`
* The ``mask`` parameter is no longer supported
* Block dimensions of tensors have been removed
* Explicit ``dst`` parameter is now required for ``nki.isa`` instructions and is always the first argument
* Nested slicing is not available
* Dynamic Access syntax has changed
* Decorators on sub-kernels need to be removed
* Dictionaries support only string keys

.. note::

   In NKI 0.2.0, ``nl.load`` and ``nl.store`` were removed in favor of ``nisa.dma_copy``. They were
   later **reintroduced in NKI 0.3.0** as convenience APIs in ``nki.language`` and remain available
   in current versions. If you are migrating all the way to a current NKI version, you can continue
   to use ``nl.load`` / ``nl.store``.

New features in NKI 0.2.0
-------------------------

New namespace, new APIs
~~~~~~~~~~~~~~~~~~~~~~~~~

NKI 0.2.0 introduces a number of changes to the language and to the compilation process. While we
are deprecating NKI 0.1.0, the NKI 0.2.0 release supports both versions of the language via
namespaces. The NKI 0.1.0 APIs can be used via the ``neuronxcc.nki.*`` namespace, while NKI 0.2.0
has moved to the ``nki.*`` namespace.

.. code-block:: python

   # Legacy NKI 0.1.0 APIs
   import neuronxcc.nki as nki
   import neuronxcc.nki.isa as nisa

   # New NKI 0.2.0 APIs
   import nki
   import nki.isa as nisa

We have made improvements to the APIs, like consistent naming, order of arguments, and matching more
closely the hardware ISA so that what developers write in NKI and what they see in the profiler are
the same. There is one change that developers should be aware of: all ISA functions now require a
destination parameter.

All ISA functions require a destination parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In NKI 0.2.0, all of the ISA functions now require a ``dst`` parameter instead of returning a
result. So, instead of writing:

.. code-block:: python

   result[...] = nisa.reciprocal(src)

Developers must write:

.. code-block:: python

   nisa.reciprocal(result[...], src)

This change makes the behavior of the APIs more consistent and matches cases where APIs may perform
accumulation or return multiple results. It also helps avoid scenarios where developers might
inadvertently write to the wrong buffer or inadvertently introduce additional copy operations.

Dynamic control flow
~~~~~~~~~~~~~~~~~~~~~~

NKI 0.2.0 includes support for dynamic (on-chip) control flow. All of the dynamic control flow uses
on-chip registers to hold the conditional values. See :ref:`trainium_inferentia2_arch` for more
information. If a control flow construct uses a register as a conditional, then the loop will be an
on-chip, dynamic (or runtime) loop. This is very common in scenarios like Mixture of Experts (MoE),
where the index space for the expert is known at runtime, but not at compile time. Dynamic control
flow with the new NKI APIs unlocks this use case.

To support dynamic control flow, NKI has a set of ``nki.isa`` APIs for reading and writing to
hardware registers. See :doc:`/nki/api/index` for more information. Their current signatures are:

.. code-block:: python

   # Allocate a register (optionally with an immediate value)
   nisa.register_alloc(x=None)

   # Move a value (constant or VirtualRegister) into dst
   nisa.register_move(dst, src)

   # Load an SRAM tensor element into the dst register
   nisa.register_load(dst, src)

   # Store the value of a register into SRAM
   nisa.register_store(dst, src)

The most basic dynamic loop is a ``for`` loop that uses a register value for the iteration value and
another register for the upper bound. Developers can write this kind of loop using
``nl.dynamic_range``:

.. code-block:: python

   # dynamic loop with dynamically computed upper bounds
   # upper_bound is a hardware register
   upper_bound = nisa.register_alloc()
   nisa.register_load(upper_bound, tensor)
   for i in nl.dynamic_range(5, upper_bound, 2):
     ...

Developers can also write dynamic while loops. When using a dynamic while loop, the developer should
update the register within the body of the loop.

.. code-block:: python

   # initialize a conditional tensor which will be updated in the loop
   cond = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)

   # create register with initial value
   reg = nisa.register_alloc(5)

   while reg:  # loop will terminate when the value reaches 0
     ...
     # store the register value into SBUF for computation
     nisa.register_store(cond, reg)
     # decrement the condition variable by 1
     nisa.tensor_scalar(cond, cond, nl.add, -1)
     # load (updated) value from cond tensor into register
     nisa.register_load(reg, cond)

Update indexing syntax for ``mgrid`` and ``arange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If using ``nl.mgrid/arange`` to access continuous elements in an existing NKI kernel, this should be
replaced with integer slicing. Take a look at the following example.

.. code-block:: python

   # Example 1
   t = nl.ndarray(shape=(128, 16, 64), ...)
   # Old approach: use mgrid to access continuous elements
   i_p, if0, if1 = nl.mgrid[0:128, 0:8, 0:64]
   t[i_p, if0, if1]
   # Updated: should just use integers to create the slice
   t[0:128, 0:8, 0:64]

   # Example 2
   t = nl.ndarray(shape=(128, 16 * 64))
   # Old approach: using mgrid
   i_p, if0, if1 = nl.mgrid[0:128, 0:8, 0:64]
   t[i_p, if0 * 64 + if1]
   # should just use integer slicing
   t[0:128, 0:8 * 64]

If your use case cannot be represented with the slicing syntax above, see :ref:`nki-aps`.

Changes in NKI 0.2.0 from NKI 0.1.0
-----------------------------------

Consistent control flow behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In NKI 0.1.0, range iterators were converted into special objects that allowed the eDSL to capture
the loop body. Because of this, loops were only executed once by the Python evaluator, which could
lead to some surprising results. For example, in the code below, the normal Python variable ``val``
ends up with a value of 1 rather than the expected value of 8. This has been solved in the new NKI
Compiler.

.. code-block:: python

   val = 0
   for i in range(8):
     val += 1
   print(val)  # will print 1 in NKI 0.1.0, prints 8 in NKI 0.2.0

For similar reasons, sometimes Python control flow constructs, such as ``if`` statements, could not
be handled properly when nested within a ``for`` loop. For example, in NKI 0.1.0 the code below
produces an undefined result. In NKI 0.2.0, this code produces the expected result.

.. code-block:: python

   val = 0
   for i in range(8):
     if i == 0:
       val = 1
     else:
       val = 2
   print(val)  # undefined behaviour in NKI 0.1.0, prints 2 in NKI 0.2.0

Many other examples of troublesome control flow have been fixed, which should make using NKI easier
and more intuitive.

.. _nki-mask:

Deprecation of masking
~~~~~~~~~~~~~~~~~~~~~~~~

Follow this section if you are using the ``mask`` parameter in your kernel.

In NKI 0.1.0, the concept of masking was introduced to modify the behavior of tensor indexing
expressions. The use of masking was almost always used to avoid out-of-bounds access. For example,
suppose a developer is tiling a tensor of size 129 x 513, and you want to use tiles of size
128 x 512. A typical way to write a tiling loop in NKI 0.1.0 is shown below.

.. code-block:: python

   t = nl.ndarray(shape=(129, 513), ...)
   result = nl.ndarray(shape=(129, 513), ...)
   for i in range(2):
     for j in range(2):
       i_p, i_f = nl.mgrid[0:128, 0:512]
       result[i_p + 128 * i, i_f + 512 * j] = nisa.tensor_copy(t[i_p + 128 * i, i_f + 512 * j],
        mask=(i_p + 128 * i < 129) & (i_f + 512 * j < 513))

Note, when ``i`` (or ``j``) is equal to 1, then the index expression
``result[i_p+128*i, i_f+512*j]`` would overflow the tensor dimension. The mask expression modifies
the indexing so that the equations are true, and thus inbounds of the tensor. This mechanism has
many drawbacks, including being error-prone and non-intuitive for Python developers. Therefore, this
mechanism has been deprecated in NKI 0.2.0.

In NKI 0.2.0, developers can use standard constructs from Python such as ``min`` and ``slice`` to
build indexing expressions that are in bounds for the tensor. For example, the above code can now be
written as:

.. code-block:: python

   for i in range(2):
     p_start = i * 128
     p_end = min(129, p_start + 128)
     p = slice(p_start, p_end)  # a.k.a. (p_start:p_end)

     for j in range(2):
       f_start = j * 512
       f_end = min(513, f_start + 512)
       f = slice(f_start, f_end)  # a.k.a. (f_start:f_end)

       nisa.tensor_copy(result[p, f], t[p, f])

The developer may also choose to inline the slices, if that is more natural.

.. code-block:: python

   nisa.tensor_copy(result[p_start:p_end, f_start:f_end],
                    t[p_start:p_end, f_start:f_end])

Improved allocation API
~~~~~~~~~~~~~~~~~~~~~~~~~

The manual allocation API has been simplified. In NKI 0.2.0 there is an ``address`` argument to
``nl.ndarray`` that allows the offset of each tensor to be specified: ``(partition_offset,
free_offset)``. Similar to NKI 0.1.0, while the partition offset corresponds to a physical partition
lane on the hardware, the free dimension offset is the element offset within each partition. The
free dimension offset is translated into a physical SBUF address in the compiler.

.. code-block:: python

   # creates your buffer on partition 0, offset by 128 elements of your data type
   a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result",
     address=(0, 128), buffer=nl.sbuf)

The address space for PSUM is now also 2D to be consistent with the hardware. Recall that PSUM on
NeuronCore v2/v3/v4 is organized into 128 partitions, each consisting of 16KB of memory. Each
partition is further divided into 8 PSUM banks, with each bank holding up to 2KB worth of values.
The allocation for PSUM tensors must start at the beginning of each bank — the compiler will throw
an error otherwise.

For example, the following code will allocate a PSUM tensor on bank 3:

.. code-block:: python

   bank_id = 3
   PSUM_BANK_SIZE = 2048
   psum_t = nl.ndarray(dtype=nl.bfloat16, shape=(128, 1024),
     address=(0, bank_id * PSUM_BANK_SIZE))

Translate from the NKI 0.1.0 direct allocation API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To translate the direct-allocated kernel in NKI 0.1.0, all data structures must not use the block
dimension. This means reformatting tensors to place the partition dimension in the left-most
position, using either lists or multi-dimensional tensors for the rest of your dimensions. See
:ref:`nki_block_dimension_migration_guide` for more information.

After this, translate the address of each block. For example, given the following tensor in NKI
0.1.0 that uses modular allocation:

.. code-block:: python

   # NKI 0.1.0 - uses block dimension and mod allocator
   k_loaded = nl.ndarray((num_512_tiles_cur_section, nl.par_dim(p_k), n_k),
    dtype=nl.bfloat16,
    buffer=sb_mod(base_addr=sca, num_free_tiles=(num_512_tiles_cur_section,)))

Now with NKI 0.2.0, developers can translate the block dimension into a list and compute the address
for each block.

.. code-block:: python

   # NKI 0.2.0 - use lists of tensors and get lists of virtual byte addresses
   k_loaded_tensors = []
   for i in range(num_512_tiles_cur_section):
     k_loaded_tensors.append(nl.ndarray(shape=(p_k, n_k), dtype=nl.bfloat16,
       buffer=nl.sbuf, address=(0, sca + (i % num_512_tiles_cur_section) * n_k * 2)))

Remove ``nki.jit`` decorator on sub-kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For kernels that call other kernels, or call any other functions that are decorated with a
``nki.jit`` decorator, the ``nki.jit`` decorator will need to be removed from sub-kernels.

In NKI 0.1.0, all the sub-kernels called from a top-level kernel could be decorated with
``nki.jit(mode='trace')`` decorator. This decorator needs to be removed for the new NKI Compiler.
Otherwise, you will see an error about classes needing to inherit from ``nl.NKIObject`` thrown from
the callsite of the sub-kernels.

If a kernel is being called by another kernel and it is also called standalone, the decorator can be
applied on-the-fly at the call site to avoid this problem.

.. code-block:: python

   # Do not apply the decorator on the kernel definition
   def my_kernel(...):
     pass

   # When calling the kernel, apply the decorator
   a = torch.tensor(...)
   kernel_decorated = nki.jit(my_kernel)
   result = kernel_decorated(a)

Dynamic access pattern
~~~~~~~~~~~~~~~~~~~~~~~~

Follow this section for a kernel that uses dynamic access, i.e. using a runtime value to index
another tensor.

The syntax for representing dynamic access patterns has changed. In NKI 0.1.0, an access with a
dynamic scalar offset could be represented as shown below where ``batch_idx`` is a dynamic value in
the SBUF:

.. code-block:: python

   batch_idx = nl.multiply(nl.bitwise_and(nl.load(dynamic_idx), y=3), 128)
   result = nl.ndarray((128, 256), A.dtype, buffer=nl.shared_hbm)
   batch_idx[...] = 4  # set a constant, but batch_idx is a runtime SBUF value
   i_p, i_f = nl.mgrid[0:128, 0:256]
   nisa.dma_copy(src=A[batch_idx, i_p, i_f], dst=result[...])

Scalar dynamic access
^^^^^^^^^^^^^^^^^^^^^^

In NKI 0.2.0, we need to use a physical access pattern, specified with the ``.ap`` method, to
represent this.

.. code-block:: python

   def indirect_scalar_dynamic_dma(A):
     # Assume input A is of shape (4*128, 512). We want to copy from A[3*128:, 0:256]
     # The 3*128 offset comes from a dynamic variable in SBUF
     assert A.shape == [512, 512]
     batch_idx = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
     nisa.memset(batch_idx, value=3 * 128)

     result = nl.ndarray((128, 256), A.dtype, buffer=nl.shared_hbm)

     nisa.dma_copy(src=A.ap(
       pattern=[[512, 128], [1, 256]], offset=0,
       scalar_offset=batch_idx, indirect_dim=0
       ),
       dst=result[...])

     return result

The ``scalar_offset`` is an SBUF value that specifies the index on the ``indirect_dim`` of the
tensor. For example, the code block above accesses ``batch_idx`` on the 0-th dimension of the tensor
``A``. It is important to note that the dimension is relative to the **base tensor**, not relative
to the **pattern** specified.

This example will access the memory from ``A`` starting at the element offset below.

.. code-block:: python

   # prod(A.shape[indirect_dim+1:]) is the accumulated shape
   # to the right of indirect_dim
   offset + scalar_offset * prod(A.shape[indirect_dim+1:])

In the example above, the access would start from:

.. code-block:: python

   0 + batch_idx * 512

Again, we should notice that ``512`` is read from the shape of the **base tensor**, not from the
access pattern. The shape of the access pattern is ``(128, 256)``.

In conventional NumPy syntax, the above means that we are accessing
``A[batch_idx:batch_idx+128, 0:256]``. Writing this in the canonical loop form, the result of the
access is the following:

.. code-block:: python

   result = nl.ndarray(shape=(128, 256), dtype=A.dtype, buffer=nl.sbuf)
   for x in range(128):
     for y in range(256):
       result[x, y] = A.flatten()[0 + batch_idx * 512 + x * 512 + y * 1]

Vector dynamic access
^^^^^^^^^^^^^^^^^^^^^^

Vector dynamic access is similar to that of scalar, except that we need to specify the field
``vector_offset``. **Currently, only** ``indirect_dim=0`` **is supported.** The stride on the
leading dimension must be the total number of elements to the right of the leading dimension in the
**base tensor**, and the stride specified in the leading dimension of the pattern in the ``.ap()``
is currently ignored. We still recommend setting the stride properly so that code would still work
if this limitation is lifted in the future.

.. code-block:: python

   def indirect_vector_dynamic_dma(A):
     # shape of A is (128, 512)
     dynamic_idx_legal = nl.ndarray((64, 1), nl.int32, buffer=nl.sbuf)
     nisa.iota(dynamic_idx_legal, [[1, 1]], offset=0, channel_multiplier=2)

     result_sb = nl.ndarray((64, 512), nl.float32, buffer=nl.sbuf)
     result_hbm = nl.ndarray((64, 512), nl.float32, buffer=nl.shared_hbm)

     nisa.dma_copy(src=A.ap(
       [[512, 64], [1, 512]], offset=0, vector_offset=dynamic_idx_legal, indirect_dim=0
       ), dst=result_sb, name='inst0')

     nisa.dma_copy(result_hbm, result_sb, name="copy1")

     return result_hbm

For this particular case, the semantics of the access are the following. Note that the stride on the
dynamic dimension is directly read from the **base tensor**.

.. code-block:: python

   indirect_dimension = 0

   for w in range(64):
     for z in range(512):
       dynamic_idx = dynamic_idx_legal[w]
       A[
              # static offsets
              offset +
              # AP with the indirect dimension number replaced
              # Note that the 512 is read from the shape of the **base** tensor.
              1 * z + 512 * dynamic_idx
             ]

Further reading
---------------

- :doc:`/nki/deep-dives/nki-compiler`
- :doc:`/nki/api/index`


.. _nki_block_dimension_migration_guide:

Removing block dimensions from NKI kernels
==========================================

SBUF/PSUM tensors in NKI used to allow block dimensions in front of the partition dimension. Block
dimension support has been removed for the following reasons.

* Removing block dimensions does not hurt the expressivity of NKI.
* Block dimension is a pure software concept and does not have a direct hardware mapping.
* The block dimension is unintuitive and causes confusion.
* Using a block dimension has no inherent performance benefit; in particular, using a block
  dimension has no relationship with memory throughput whatsoever.
* Multi-buffering is implicit with block dimension. Removing block dimensions makes multi-buffering
  more natural.

This section first explains the semantics of block dimensions in detail, then provides information
on how to migrate existing code that uses block dimensions while maintaining functional correctness
and performance.

What are block dimensions?
--------------------------

Consider the following NKI tensor.

.. code-block:: python

  a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), buffer=nl.sbuf)

  # - (4, 8): (B) block dimensions
  # - 128: (P) partition dimension
  # - (2, 512): (F) free dimension

A NKI tensor has three types of dimensions: ``(B, P, F)``. The partition dimension maps to the
partition dimension of the physical memory, and the free dimensions describe how data is organized
in each SBUF/PSUM partition. The block dimensions describe how many physical ``(P, F)`` tiles the
tensor has.

The block dimension of tensors is a **logical** dimension and is a pure software concept. The
compiler analyzes the memory dependency and allocates a physical address to each tile. **This means
that the physical tiles may not be alive in the memory simultaneously**, and in most cases they do
not. Consider the following code snippet that accesses the tensor ``a``.

.. code-block:: python

  @nki.jit
  def exp_func(inp):
    output = nl.ndarray((4, 8, 128, 2, 512), dtype=nl.float32,
      buffer=nl.shared_hbm)
    a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(4):
      for j in range(8):
        a[i, j] = nl.load(inp[i, j])
        a[i, j] = nl.exp(a[i, j])
        nl.store(output[i, j], value=a[i, j])

At the very minimum, only 1 physical tile of ``a`` needs to be alive. Then the execution is
completely serialized. Essentially, all physical tiles would have the exact same memory address.

.. code-block::

  Physical Address Map

  a[0, 0] --> Partition 0 - 128, Free 0 - 2048B
  a[0, 1] --> Partition 0 - 128, Free 0 - 2048B
  ...

Instead, the compiler could choose to allocate 2 physical tiles to ``a``, then the DMA copy from HBM
to SBUF can overlap with the exponential operation. In other words, **the block dimension allows the
compiler to perform a space-time tradeoff at liberty.**

.. code-block::

  Physical Address Map

  a[0, 0] --> Partition 0 - 128, Free 0    - 2048B
  a[0, 1] --> Partition 0 - 128, Free 2048 - 4096B
  a[0, 2] --> Partition 0 - 128, Free 0    - 2048B
  a[0, 3] --> Partition 0 - 128, Free 2048 - 4096B
  ...

When performing the migration, it is important to understand the dependency relationship between
blocks and choose the correct migration method accordingly. There are two performance-equivalent
ways to translate block dimensions: use a Python-like list, or use a differently-shaped tensor.

Use a Python-like list
----------------------

A block dimension in NKI 0.1.0 was syntactic sugar for a list of tensors managed by the compiler.
You can code this pattern directly with a standard Python list, without any extra compiler support.

.. code-block:: python

   # Before migration (block dimension)
   t = nl.ndarray((8, nl.par_dim(128), 256), dtype=nl.float32, buffer=nl.sbuf)
   for i in range(8):
     ...  # use t[i]

   # After migration: an explicit list of tensors
   t_lst = []
   for i in range(8):
     t_lst.append(nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf))
   for i in range(8):
     ...  # use t_lst[i]

With this approach, the programs generated before and after migration are identical and should yield
the same performance.

Use a differently-shaped tensor
-------------------------------

Alternatively, fold the block dimension into the free dimension or hoist the tensor declaration
inside the loop, depending on whether the blocks must be alive simultaneously.

Migration for SBUF tensors
--------------------------

If blocks need to be alive at the same time, move the block dimension into the free dimension
****************************************************************************************************

.. code-block:: python

  a = nl.ndarray((8, 128, 512), buffer=nl.sbuf, dtype=nl.bfloat16)  # was (8, par_dim(128), 512)

As an example, all 8 blocks of ``add_buf`` need to be alive at the same time when the first ``for``
loop finishes. Therefore, the block dimension needs to be folded into the free dimension.

.. code-block:: python

    @nki.jit
    def sb_blocks(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[i] = nl.load(inp[i])
        for i in range(8):
            nl.store(res[i], add_buf[i])
        return res

    # should migrate to
    @nki.jit
    def sb_blocks_migrated(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(128, 8, 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[0:128, i, 0:512] = nl.load(inp[i])
        for i in range(8):
            nl.store(res[i], add_buf[0:128, i, 0:512])
        return res

If blocks do not need to be alive at the same time, remove the block dimension and hoist it down
****************************************************************************************************

.. code-block:: python

  # before
  a = nl.ndarray((8, 128, 256), buffer=nl.sbuf)  # was (8, par_dim(128), 256)
  for i in nl.affine_range(8):
    ...  # do something with a[i]

  # should be transformed to ....
  for i in nl.affine_range(8):
    a = nl.ndarray((128, 256), buffer=nl.sbuf)
    ...  # do something with a

As an example, all 8 blocks of ``add_buf`` do not need to be alive at the same time. We can remove
the block dimension and hoist the tensor down inside the loop.

.. code-block:: python

    @nki.jit
    def sb_blocks(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[i] = nl.load(inp[i])
            nl.store(res[i], add_buf[i])
        return res

    # should migrate to
    @nki.jit
    def sb_blocks_migrated(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        for i in range(8):
            add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
            add_buf[0:128, 0:512] = nl.load(inp[i])
            nl.store(res[i], add_buf[0:128, 0:512])
        return res

.. warning::
    To preserve performance, it is important to hoist the tensor down inside the loop.

It is important to note that the dependency relationship between loop iterations is different in
``sb_blocks_migrated`` and the following ``sb_blocks_migrated_incorrect``.

.. code-block:: python

    @nki.jit
    def sb_blocks_migrated_incorrect(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[0:128, 0:512] = nl.load(inp[i])
            nl.store(res[i], add_buf[0:128, 0:512])
        return res

In ``sb_blocks_migrated``, the compiler could unroll the loop and materialize multiple copies of the
tensor ``add_buf``. However, in ``sb_blocks_migrated_incorrect``, the execution will be serialized
because the loop carries a dependency on ``add_buf``.
