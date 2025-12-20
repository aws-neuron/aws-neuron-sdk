.. meta::
   :description: Overview of Indexing in NKI
   :date_updated: 12/02/2025

.. _nki-about-indexing:

=======================
Tensor Indexing on NKI
=======================

This topic covers basic tensor indexing and how it applies to developing with the AWS Neuron SDK. This overview describes basic indexing of tensors with several examples of how to use indexing in NKI kernels.

Basic Tensor Indexing
^^^^^^^^^^^^^^^^^^^^^

NKI supports basic indexing of tensors using integers as indexes. For example,
we can index a 3-dimensional tensor with a single integer to get get a *view*
of a portion of the original tensor.

.. code-block::

   x = nl.ndarray((2, 2, 2), dtype=nl.float32, buffer=nl.hbm)

   # `x[1]` return a view of x with shape of [2, 2]
   # [[x[1, 0, 0], x[1, 0 ,1]], [x[1, 1, 0], x[1, 1 ,1]]]
   assert x[1].shape == [2, 2]

NKI also supports creating views from sub-ranges of the original tensor
dimension. This is done with the standard Python **slicing** syntax. For
example:

.. code-block::

   x = nl.ndarray((2, 128, 1024), dtype=nl.float32, buffer=nl.hbm)

   # `x[1, :, :]` is the same as `x[1]`
   assert x[1, :, :].shape == [128, 1024]

   # Get a smaller view of the third dimension
   assert x[1, :, 0:512].shape == [128, 512]

   # `x[:, 1, 0:2]` returns a view of x with shape of [2, 2]
   # [[x[0, 1, 0], x[0, 1 ,1]], [x[1, 1, 0], x[1, 1 ,1]]]
   assert x[:, 1, 0:2].shape == [2, 2]

When indexing into tensors, NeuronCore offers much more flexible memory access
in its on-chip SRAMs along the free dimension. You can use this to efficiently
stride the SBUF/PSUM memories at high performance for all NKI APIs that access
on-chip memories. Note, however, this flexibility is not supported along the
partition dimension. That being said, device memory (HBM) is always more
performant when accessed sequentially.

.. _nki-advanced-tensor-indexing:

Tensor Indexing by Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we share several use cases that benefit from advanced
memory access patterns and demonstrate how to implement them in NKI.

Case #1 - Tensor split to even and odd columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we split an input tensor into two output tensors, where the first
output tensor gathers all the even columns from the input tensor,
and the second output tensor gathers all the odd columns from the
input tensor. We assume the rows of the input tensors are mapped to SBUF
partitions. Therefore, we are effectively gathering elements along
the free dimension of the input tensor. :numref:`Fig. %s <nki-fig-pm-index-1>`
below visualizes the input and output tensors.

.. figure:: /nki/img/pm-index-1.png
   :align: center
   :width: 60%

   Tensor split to even and odd columns

.. nki_example:: /nki/examples/index-case-1.py
   :language: python
   :linenos:
   :whole-file:

The main concept in this example is that we are using slices to access the even
and odd columns of the input tensor. For the partition dimension, we use the
slice expression `:`, which selects all of the rows of the input tensor. For
the free dimension, we use `0:sz_f:2` for the even columns. This slice says:
start at index `0`, take columns unto index `sz_f`, and increment by `2` at
each step. The odd columns are similar, except we start at index `1`.


Case #2 - Transpose tensor along the f axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we transpose a tensor along two of its axes. Note,
there are two main types of transposition in NKI:

1. Transpose between the partition-dimension axis and one of the free-dimension axes, which is achieved via the
   :doc:`nki.isa.nc_transpose </nki/api/generated/nki.isa.nc_transpose>` API.
2. Transpose between two free-dimension axes, which is achieved via a :doc:`nki.isa.dma_copy </nki/api/generated/nki.isa.dma_copy>` API,
   with indexing manipulation in the transposed axes to re-arrange the data.

In this example, we'll focus on the second case: consider a
three-dimensional input tensor ``[P, F1, F2]``, where the ``P`` axis is mapped
to the different SBUF partitions and the ``F1`` and ``F2`` axes are
flattened and placed in each partition, with ``F1`` being the major
dimension. Our goal in this example is to transpose the ``F1`` and
``F2`` axes with a parallel dimension ``P``,
which would re-arrange the data within each partition. :numref:`Fig. %s <nki-fig-index-2>`
below illustrates the input and output tensor layouts.

.. figure:: /nki/img/pm-index-2.png
   :align: center
   :width: 60%

   Tensor F1:F2 Transpose

.. nki_example:: /nki/examples/transpose2d/transpose2d_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_33

The main concept introduced in this example is a 2D memory access
pattern per partition, via additional indices. We copy ``in_tile`` into
``out_tile``, while traversing the memory in different access patterns
between the source and destination, thus achieving the desired
transposition.

You may download the full runnable script from :ref:`Transpose2d tutorial <tutorial_transpose2d_code>`.

Case #3 - 2D pooling operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lastly, we examine a case of
dimensionality reduction. We implement a 2D MaxPool operation, which
is used in many vision neural networks. This operation takes
``C x [H,W]`` matrices and reduces each matrix along the ``H`` and ``W``
axes. To leverage free-dimension flexible indexing, we can map the ``C``
(parallel) axis to the ``P`` dimension and ``H/W`` (contraction)
axes to the ``F`` dimension.
Performing such a 2D pooling operation requires a 4D memory access
pattern in the ``F`` dimension, with reduction along two axes.
:numref:`Fig. %s <nki-fig-index-3>`
below illustrates the input and output tensor layouts.

.. figure:: /nki/img/pm-index-3.png
   :align: center
   :width: 60%

   2D-Pooling Operation (reducing on axes F2 and F4)

.. nki_example:: /nki/examples/index-case-3.py
   :language: python
   :linenos:
   :whole-file:


