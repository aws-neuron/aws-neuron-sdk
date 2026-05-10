.. meta::
   :description: Deep dive into Access Patterns (AP) to directly specify how tensors are accessed on Trainium hardware
   :keywords: NKI kernels, Neuron Kernel Interface, AWS Neuron SDK, kernel compilation, Trainium, Inferentia, machine learning acceleration
   :date-modified: 12/19/2025

.. _nki-aps:

===================
NKI Access Patterns
===================

NKI supports hardware-native access patterns (APs) on ``NkiTensor`` via
the :meth:`NkiTensor.ap` method, letting developers specify precisely
what they want their instructions to read from on the hardware.

For most code, the higher-level NkiTensor view primitives — ``slice``,
``reshape``, ``permute``, ``broadcast``, ``select``, ``view(dtype)``, etc.
— should be sufficient. See the :ref:`Tensor Values <tensor-values>`
section of the NKI Language Guide for an overview of those primitives.
When none of the NkiTensor wrapper APIs expresses the access pattern you
need, drop down to the lower-level ``.ap()`` API — it can express any
access pattern the hardware supports.

.. note::

   The NkiTensor view primitives are implemented on top of ``.ap()``:
   each view method composes a hardware access pattern at specialization
   time, without any runtime cost. ``.ap()`` is the raw building block.

Hardware Capability
===================

Instructions can read and write tensors from/to the SBUF or PSUM, which are 
both two-dimensional memories with 128 partitions on NeuronCore v2/v3/v4. 
Within each SBUF/PSUM partition, the tensor read/write logic on the NeuronCore 
supports accessing elements from up to four-dimensional arrays, though most 
instructions only support 1D/2D/3D in the free dimension due to instruction 
length limitations.

The multi-dimensional access patterns are typically described using two pieces 
of information: 1) the element stepping (i.e., ``stride``) and 2) number of 
elements (i.e., ``size``) in each dimension. A tensor access pattern of an 
instruction is expected to be the same across all partitions.

In addition to the free dimension pattern, additional information is required 
to locate the number of elements to access: 1) the offset from the beginning 
of the tensor and 2) the number of partitions. The next section will describe 
how the NKI API abstracts this information.

NKI API for the Access Pattern
===============================

The NKI API for access pattern is a direct reflection of the hardware capability. 
The ``nl.ndarray`` has an ``ap`` method.

.. code-block:: python

   def ap(self, pattern: list[list[int]],
      offset: int | VirtualRegister | None = None,
      scalar_offset: NkiTensor | VirtualRegister | None = None,
      vector_offset: NkiTensor | None = None,
      indirect_dim: int = 0,
      dtype: DType | None = None):
      pass

The parameters have the following definitions:

* ``pattern``: A list of two-element tuples, each tuple describes the access on one dimension. The first element represents the element stepping and the second element represents the number of elements in each dimension. This tuple is referred to as ``[step, num]`` going forward.

  * The shape of a pattern is the collection of num. For example, given pattern ``[[w_step, w_num], [z_step, z_num], [y_step, y_num], [x_step, x_num]]``, the shape is ``[w_num, z_num, y_num, x_num]``.
  * **Note**: The order of the pattern specified here is in the opposite order to what is actually accepted by the hardware. Therefore, the order of the tuples shown on the profiler will be in the opposite order of what is specified here.

* ``offset``: The offset to start the access in terms of number of elements from the beginning of the tensor. Can be a compile-time integer or a :doc:`VirtualRegister <../api/generated/nki.isa.VirtualRegister>`. When ``None`` (the default), the new view inherits the current view's storage offset.
* ``scalar_offset``: An SBUF tensor of shape ``(1, 1)`` or a :doc:`VirtualRegister <../api/generated/nki.isa.VirtualRegister>` that specifies the location to start the access in terms of number of elements on the ``indirect_dim`` of the access pattern. At most one of the ``scalar_offset`` and ``vector_offset`` can be specified.
* ``vector_offset``: An SBUF tensor that specifies the location to start the access in terms of number of elements from the beginning of the indirect dimension specified by ``indirect_dim``. At most one of the ``scalar_offset`` and ``vector_offset`` can be specified.
* ``indirect_dim``: The indirect dimension on which to apply ``scalar_offset`` and ``vector_offset``.
* ``dtype``: The data type of the access pattern. The default value is the ``dtype`` of the tensor being accessed.

.. _ap-meta-programming:

View Primitives Build APs at Specialization Time
=================================================

The higher-level NkiTensor view primitives (``slice``, ``reshape``,
``permute``, ``broadcast``, ``select``, ``view(dtype)``, ``rearrange``,
etc.) are implemented as NKI meta-programming on top of ``.ap()``.
Every view operation runs at specialization time: it reads the current
view's shape, strides, and offset, applies its transform, and produces
a new ``NkiTensor`` whose ``get_pattern()`` returns a hardware access
pattern — the same pattern you would have written by hand with
``.ap()``.

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
   u = t.slice(dim=1, start=8, end=24)
   assert u.get_pattern() == [[64, 128], [1, 16]]
   assert u.offset == 8

   # Equivalent to constructing the same AP by hand with .ap():
   v = t.ap(pattern=[[64, 128], [1, 16]], offset=8)
   assert v.get_pattern() == u.get_pattern()
   assert v.offset == u.offset

Because the view composition happens at specialization time, a chain of
view operations has zero runtime cost: only the final pattern is handed
to the ISA instruction that consumes the tensor.

Semantics of the Access Pattern
================================

Access patterns can be thought of as compact representations of a loop. The 
offset is an integer indicating the start offset in terms of elements with 
respect to the beginning of the tensor. Each two-element list ``[step, num]`` 
represents the stride in terms of elements and the number of iterations of 
each level of the loop. The semantics are explored through the following 
example.

Given a tensor, the Access Pattern conceptually flattens the tensor to 1D,
and then uses a loop to fetch elements from the tensor to construct a view.
Consider the following NKI code:

.. code-block:: python

   t = nl.ndarray((p_count, N), dtype=nl.float32, buffer=nl.sbuf)
   access = t.ap(
     pattern=[[N, p_size], [z_step, z_num], [
     y_step, y_num], [x_step, x_num]], 
     offset)

The above represents the following access on the tensor ``t``, written below in pseudo-code.

.. code-block:: python

   access = nl.ndarray((p_size, z_num, y_num, x_num), dtype=nl.float32, buffer=nl.sbuf)
   for w in range(p_size):
     for z in range(z_num):
       for y in range(y_num):
         for x in range(x_num):
           t_flatten = t.flatten() # first flatten the tensor to 1d
           access[w, z, y, x] = t_flatten[offset + (w * N) + (z * z_step)
                     + (y * y_step) + (x * x_step)]

The access pattern has the following properties:

1. Recall from the hardware capability, the access pattern in each partition 
must be identical. Therefore, the step of the first tuple in the AP must be 
equal to the number of elements in the free dimension of the tensor.
2. The shape of the result view is always the same as the shape of the pattern.

Note that calling ``.ap`` on a tensor does not do any computation directly. 
It describes how to get data. The engines will consume data when the AP 
is passed into a ``nki.isa`` instruction.

.. code-block:: python

   src = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)
   dst = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)
   src_access = src.ap([32, 16], [1, 32]) # no computation happens
   dst_access = dst.ap([32, 16], [1, 32]) # no computation happens

   # Engine reads both src_access and dst_access and performs the copy
   nisa.dma_copy(dst_access, src_access)

A Concrete Example
==================

Given a tensor ``t`` of size (16P, 16F), to iterate all the elements in 
``t[0:16, 8:16]`` the access pattern can be written as:

.. code-block:: python

   t = nl.ndarray((16, 16), dtype=nl.float32, buffer=nl.sbuf)
   access = t.ap(pattern=[[16, 16], [1, 8]], offset=8)


   # Semantics, the following is pseudo-code
   access = nl.ndarray((16, 8), dtype=nl.float32, buffer=nl.sbuf)
   # in loop form
   for w in range(16):
     for z in range(8):
       idx = 8 + (w * 16) + (1 * z)
       t_flatten = t.flatten()
       access[w, z] = t_flatten[idx]

.. image:: /nki/img/deep-dives/memory-access-visualization-1.png
   :width: 80%
   :align: center

Restriction on SBUF/PSUM Tensors
=================================

For SBUF/PSUM tensors, the first tuple must always be the access for the 
partition dimension. On NeuronCore v2/v3/v4, the access on the partition 
dimension must be contiguous, meaning that the step of the leading dimension 
must be the element count of the entire free dimension of the tensor. 
Therefore, given a tensor of shape ``(p_dim, f_dim0, f_dim1)``, the step of 
the leading dimension must be ``f_dim0 * f_dim1``.

The following example is not allowed because it reads every other partition.

.. code-block:: python

   t = nl.ndarray((16, 32), dtype=nl.float32, buffer=nl.sbuf)

   # The following is illegal, because the first stride is 32*2 and reads every other partition
   t.ap(pattern=[[64, 8], [1, 32]], offset=0)

.. image:: /nki/img/deep-dives/memory-access-visualization-2.png
   :width: 80%
   :align: center

Composition and ``.ap`` on a view
=================================

``.ap`` addresses the tensor's underlying storage directly: the pattern
and offset are interpreted against ``storage``, ignoring any shape,
strides, or offset carried by the view ``.ap`` is called on.  In
particular, ``.ap`` is **not composable** — chaining ``.ap`` is allowed
but the inner ``.ap`` is fully overridden by the outer one:

.. code-block:: python

   t = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)
   inner = t.ap(pattern=[[256, 128], [2, 128]])
   outer = inner.ap(pattern=[[256, 128], [1, 64]])
   # outer is equivalent to t.ap(pattern=[[256, 128], [1, 64]])
   # — `inner`'s pattern has no effect on `outer`.

For composable, view-on-view semantics that stay inside the NkiTensor
model, use the higher-level view primitives exposed on ``NkiTensor`` —
``slice``, ``reshape``, ``permute``, ``broadcast``, ``select``,
``view(dtype)``, etc. They return new ``NkiTensor`` views of the same
storage and chain freely:

.. code-block:: python

   t = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.sbuf)

   # Compose slicing on both dims, then slice again.
   t_a = t.slice(dim=0, start=0, end=128).slice(dim=1, start=0, end=256, step=2)
   t_b = t_a.slice(dim=0, start=0, end=64).slice(dim=1, start=0, end=64)
   # t_b has shape (64, 64), storage shared with t.

See the :ref:`Tensor Values <tensor-values>` section of the NKI Language
Guide for the full catalogue of view primitives.


Reinterpret Cast with ``ap``
============================

The ``dtype`` parameter can be used for reinterpret casting the tensor.
Since both the pattern and the offset are in terms of number of elements,
not bytes, the count must be computed accordingly. See the following example
of reinterpret cast from ``INT32`` to ``BF16``.

.. code-block:: python

   t = nl.ndarray((128, 256), dtype=nl.int32, buffer=nl.sbuf)
   cast_to_bf16 = t.ap(pattern=[
     [512, 128], [1, 512]
    ], # notice the number of elements is doubled due to dtype size change
   offset = 0, dtype=nl.bfloat16) # cast_to_bf16 has shape (128, 512)

For the common case of reinterpreting the bits of an existing view
without otherwise changing its layout, :meth:`NkiTensor.view` is the
idiomatic alternative — e.g. ``t.view(nl.bfloat16)`` — and takes care
of scaling the last-dim size automatically.

Dynamic Access with ``scalar_offset`` and ``vector_offset``
===========================================================

The ``scalar_offset`` and ``vector_offset`` are for dynamic tensor access, i.e. using a 
runtime value to index another tensor. 

Scalar Dynamic Access
---------------------

The ``scalar_offset`` is an SBUF value that specifies the index on the ``indirect_dim`` of the tensor. 

.. code-block:: python
   
   def scalar_dynamic_dma(A):
      # Assume input A is of shape (4*128, 512). We want to copy from A[3*128:, 0:256]
      # The 3*128 offset comes from a dynamic variable in SBUF
      assert A.shape == [512, 512]
      batch_idx = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
      nisa.memset(batch_idx, value=3*128)

      result = nl.ndarray((128, 256), A.dtype, buffer=nl.shared_hbm)

      nisa.dma_copy(src=A.ap(
         pattern=[[512, 128], [1, 256]], offset=0,
         scalar_offset=batch_idx, indirect_dim=0
         ),
         dst=result[...])

      return result

The code block above accesses ``batch_idx`` on the 0-th dimension 
of the tensor A. Note that the dimension is relative to 
the base tensor, not relative to the pattern specified.

This example will access the memory from A starting at the element offset below.

.. code-block:: python

   # prod(A.shape[indirect_dim+1:]) is the accumulated shape
   # to the right of indirect_dim
   offset + scalar_offset * prod(A.shape[indirect_dim+1:])

In the example above, the access starts from:

.. code-block:: python

   0 + batch_idx * 512

Again, we should notice that 512 is read from the shape of the base tensor, not from the access pattern. The shape of the access pattern is ``(128, 256)``.


Vector Dynamic Access
---------------------

Vector dynamic access is similar to that of scalar, except that the dynamic offsets are in a vector. 
We need to specify the field ``vector_offset``. **Currently, only ``indirect_dim=0`` is supported**. 
The stride on the leading dimension must be the total number of elements to the right of the 
leading dimension in the base tensor, and the stride specified in the 
leading dimension of the pattern in the .ap() is currently ignored. 
We still recommend setting the stride properly so that code would still work 
if this limitation is lifted in the future.

.. code-block:: python 

   def indirect_vector_dynamic_dma(A):
      # shape of A is (128, 512)
      dynamic_idx_legal = nl.ndarray((64, 1), nl.int32, nl.sbuf)
      nisa.iota(dynamic_idx_legal, [[1, 1]], 0, 2)

      result_sb = nl.ndarray((64, 512), nl.float32, buffer=nl.sbuf)
      result_hbm = nl.ndarray((64, 512), nl.float32, buffer=nl.shared_hbm)

      nisa.dma_copy(src=A.ap(
         [[512, 64], [1, 512]], 0, vector_offset=dynamic_idx_legal, indirect_dim=0
         ), dst=result_sb, name='inst0')

      nisa.dma_copy(result_hbm, result_sb, name="copy1")

      return result_hbm

For this particular case, the semantics of the access are the following. Note that the stride on the dynamic dimension is directly read from the base tensor.

.. code-block:: python

   indirect_dimension = 0

   for w in range(64):
     for z in range(512):
      dynamic_idx = dynamic_idx_legal[w]
         A[
            // static offsets
            offset +
            // AP with the indirect dimension number replaced
            // Note that the 512 is read from the shape of the **base** tensor.
            1 * z + 512 * dynamic_idx
         ]


Interaction with DGE
--------------------

The ``scalar_offset`` and ``vector_offset`` interact with the DGE mode selection. Refer to 
:doc:`Descriptor Generation Engine (DGE) Reference </nki/deep-dives/nki-dge>` for details.
