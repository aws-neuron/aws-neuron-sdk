.. meta::
   :description: Scalar values in NKI, covering nki.isa.scalar_load and scalar_store, arithmetic, dtype width, dynamic addressing with scalar_offset and loop bounds.
   :keywords: NKI, scalar, NkiScalar, scalar_load, scalar_store, scalar_offset, int64, AWS Neuron, Trainium
   :date-modified: 09/11/2026

.. _nki-scalar:

=========================
NKI Scalar Values
=========================

Most of NKI moves *tiles*: rectangles of data in SBUF, PSUM and HBM. Alongside
that, a kernel needs a few plain integers --- an index to gather from, a trip
count read at run time, an offset computed on device. An
:class:`nki.isa.NkiScalar` holds one such value.

This page covers where scalars come from, what arithmetic they support, how
their dtype decides both range and available operations, and the two places a
kernel puts them to work: dynamic addressing and for-loop bounds.

.. contents:: On this page
   :local:
   :depth: 2

.. note::
   Scalars require the **tracer** frontend; the parser frontend does not
   support them. Every example below is run with :func:`nki.simulate`, and each
   one produces the same value on Trainium.


Loading and Storing
===================

:func:`nki.isa.scalar_load` reads a 1x1 tile into a scalar, and
:func:`nki.isa.scalar_store` writes one back to a 1x1 tile.

Arithmetic on a scalar is *traced*, not evaluated. ``i * 4 + 1`` computes
nothing at compile time --- it records the operations and yields a new scalar.

.. code-block:: python

   import nki
   import nki.isa as nisa
   import nki.language as nl
   import numpy as np

   @nki.jit
   def scalar_arithmetic(index: nl.NkiTensor):
       index_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.dma_copy(dst=index_tile, src=index)

       i = nisa.scalar_load(index_tile)   # an NkiScalar
       i = i * 4 + 1                      # traced, not evaluated in Python

       result = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.scalar_store(result, i)

       out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=result)
       return out

   nki.simulate(scalar_arithmetic)(np.array([[10]], dtype=np.int32))  # [[41]]

The source tile must be ``nl.int32`` or ``nl.int64``. Unsigned dtypes are not
accepted; use the signed dtype of the same width. ``scalar_store`` takes a
scalar, not a constant.


Width and Range
===============

A scalar's dtype is fixed when it is loaded and never changes. Nothing widens
automatically, so arithmetic that outgrows ``nl.int32`` **wraps**:

.. code-block:: python

   @nki.jit
   def scalar_wraps(value: nl.NkiTensor):
       tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.dma_copy(dst=tile, src=value)

       v = nisa.scalar_load(tile)
       squared = v * v                    # 10**10 does not fit 32 bits

       result = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.scalar_store(result, squared)
       out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=result)
       return out

   nki.simulate(scalar_wraps)(np.array([[100000]], dtype=np.int32))  # [[1410065408]]

.. warning::
   Wrapping is silent, and identical on device and under :func:`nki.simulate`. A
   scalar keeps the low bits of the true result and nothing else: overflow past
   the largest value of its dtype continues from the smallest, and underflow the
   other way. ``100000 * 100000`` is ``10**10``, whose low 32 bits are
   ``1410065408``. A value that needs more than 32 bits has to be loaded as
   ``nl.int64`` from the start.

For a wider value, load from an ``nl.int64`` tile. Arithmetic then keeps all 64
bits, and a literal too large for 32 bits is added as written:

.. code-block:: python

   @nki.jit
   def wide_scalar(value: nl.NkiTensor):
       tile = nl.ndarray((1, 1), dtype=nl.int64, buffer=nl.sbuf)
       nisa.dma_copy(dst=tile, src=value)

       v = nisa.scalar_load(tile)         # nl.int64 in, so a 64-bit scalar
       v = (v + 12345) * 3

       result = nl.ndarray((1, 1), dtype=nl.int64, buffer=nl.sbuf)
       nisa.scalar_store(result, v)
       out = nl.ndarray((1, 1), dtype=nl.int64, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=result)
       return out

   nki.simulate(wide_scalar)(np.array([[1 << 40]], dtype=np.int64))  # [[3298534920363]]

The two widths cannot be mixed. Combining an ``nl.int32`` and an ``nl.int64``
scalar in one operation is an error at compile time, and so is storing a scalar
into a tile of the other width. A Python ``int`` has no width of its own: it
takes the width of the scalar it is used with, and a literal that does not fit
that width will be rejected rather than truncated.


Supported Operations by Scalar Width
------------------------------------

``nl.int64`` supports fewer operations than ``nl.int32``:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Operation
     - ``nl.int32``
     - ``nl.int64``
   * - ``+``, ``-`` (binary and unary), ``*``
     - Supported
     - Supported
   * - ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``
     - Traced, but the result cannot be used
     - Rejected at compile time
   * - Bound of :func:`nki.language.fori_loop`
     - Supported, including an arithmetic expression
     - Not supported
   * - ``scalar_offset`` on an HBM access pattern
     - Supported
     - Supported, including offsets beyond 4GB
   * - ``scalar_offset`` on an SBUF or PSUM access pattern
     - Supported
     - Not supported

The operation limits come from the hardware, not from a NKI check that could be
relaxed: on Trainium 1, 2 and 3 the register ALU implements only addition,
subtraction and multiplication at 64 bits, so a comparison has no 64-bit form,
and neither does the compare-and-branch that a loop bound compiles into. The
restriction on an SBUF or PSUM offset has a different cause, given under
`Choosing a Width for an Offset`_.

Every unsupported case above is reported at compile time. A comparison traces,
but its result cannot be stored, used in further arithmetic, or used as an
offset, and each of those is refused where it is written.


Dynamic Addressing
==================

An access pattern's ``scalar_offset`` takes a scalar, which is how a kernel
reads a location it cannot know until run time. The offset counts elements of
the dimension named by ``indirect_dim``, so with ``indirect_dim=0`` a scalar
selects a row:

.. code-block:: python

   ROWS, COLS = 4, 8

   @nki.jit
   def gather_at_scalar_offset(data: nl.NkiTensor, index: nl.NkiTensor):
       index_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.dma_copy(dst=index_tile, src=index)
       row = nisa.scalar_load(index_tile)

       tile = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.sbuf)
       nisa.dma_copy(
           dst=tile,
           src=data.ap(
               pattern=[[COLS, 1], [1, COLS]], offset=0,
               scalar_offset=row, indirect_dim=0,
           ),
       )
       out = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=tile)
       return out

   data = np.arange(32, dtype=np.float32).reshape(4, 8)
   nki.simulate(gather_at_scalar_offset)(data, np.array([[2]], dtype=np.int32))
   # [[16. 17. 18. 19. 20. 21. 22. 23.]]

.. warning::
   A ``scalar_offset`` is not bounds-checked. An index computed on device that
   falls outside the tensor still forms an address, and the DMA reads that
   address: the kernel returns another tensor's data, or uninitialized HBM,
   without an error from the compiler or from :func:`nki.simulate`. Check the
   arithmetic that produces an index, not just the kernel that consumes it.


Choosing a Width for an Offset
------------------------------

The offset counts elements of the indirect dimension rather than bytes, and is
scaled by that dimension's stride when the address is formed, so a 32-bit index
reaches well past 4GB. Rows of 512 ``nl.float32`` values are 2KB apart, so
``2**31`` of them span roughly 4TB. An ``nl.int64`` offset is needed when the
index value itself passes ``2**31`` --- element-indexing a tensor of more than
``2**31`` elements, for instance.

An offset into SBUF or PSUM must be ``nl.int32``. A 64-bit one is rejected:
SBUF and PSUM today are too small for an offset that needs 64 bits to encode.


Loop Bounds and the Loop Variable
=================================

:func:`nki.language.fori_loop` accepts a scalar as a bound, which is how a
kernel loops a number of times it reads at run time. The loop variable handed
to the body is itself a scalar, usable in arithmetic and as an offset:

.. code-block:: python

   @nki.jit
   def sum_rows_to_bound(data: nl.NkiTensor, count: nl.NkiTensor):
       count_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.dma_copy(dst=count_tile, src=count)
       trip_count = nisa.scalar_load(count_tile)

       total = nl.zeros((1, COLS), dtype=nl.float32, buffer=nl.sbuf)

       def body(i):                       # i is an NkiScalar
           row = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.sbuf)
           nisa.dma_copy(
               dst=row,
               src=data.ap(
                   pattern=[[COLS, 1], [1, COLS]], offset=0,
                   scalar_offset=i, indirect_dim=0,
               ),
           )
           nisa.tensor_tensor(dst=total, data1=total, data2=row, op=nl.add)

       nl.fori_loop(0, trip_count, body)

       out = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=total)
       return out

   nki.simulate(sum_rows_to_bound)(data, np.array([[3]], dtype=np.int32))
   # [[24. 27. 30. 33. 36. 39. 42. 45.]]

The loop variable is always 32-bit. A 64-bit scalar can be loaded inside the
body and used as an offset there, but it cannot be combined with the loop
variable: mixing widths is refused, so a wide offset stays the same on every
iteration unless the loop carries it.

:func:`nki.language.while_loop` does not take a scalar as its condition.
