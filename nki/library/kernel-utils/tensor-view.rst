.. meta::
    :description: API reference for the TensorView utility in the NKI Library.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.core.utils.tensor_view

TensorView API Reference
========================

This topic provides the API reference for the ``TensorView`` utility. It provides zero-copy tensor view operations for NKI tensors.

When to Use
-----------

Use ``TensorView`` when you need to:

* **Reshape without copying**: Change tensor layout for different computation phases
* **Slice with strides**: Extract non-contiguous elements efficiently
* **Permute dimensions**: Transpose or reorder dimensions for matmul compatibility
* **Broadcast dimensions**: Expand size-1 dimensions without data duplication
* **Chain operations**: Combine multiple view transformations fluently

``TensorView`` is essential for kernels that need to interpret the same data in multiple layouts (e.g., attention kernels that reshape between ``[B, S, H]`` and ``[B, num_heads, S, head_dim]``).

API Reference
-------------

**Source code**: https://github.com/aws-neuron/nki-library

TensorView
^^^^^^^^^^

.. py:class:: TensorView(base_tensor)

   A view wrapper around NKI tensors supporting various operations without copying data.

   :param base_tensor: The underlying NKI tensor.
   :type base_tensor: nl.ndarray

   .. py:attribute:: shape
      :type: tuple[int, ...]

      Current shape of the view.

   .. py:attribute:: strides
      :type: tuple[int, ...]

      Stride of each dimension in elements.

   .. py:method:: get_view()

      Generates the actual NKI tensor view using array pattern.

      :return: NKI tensor with the view pattern applied.
      :rtype: nl.ndarray

   .. py:method:: slice(dim, start, end, step=1)

      Creates a sliced view along a dimension.

      :param dim: Dimension to slice.
      :type dim: int
      :param start: Start index (inclusive).
      :type start: int
      :param end: End index (exclusive).
      :type end: int
      :param step: Step size. Default 1.
      :type step: int
      :return: New TensorView with sliced dimension.
      :rtype: TensorView

   .. py:method:: permute(dims)

      Creates a permuted view by reordering dimensions.

      :param dims: New order of dimensions.
      :type dims: tuple[int, ...]
      :return: New TensorView with permuted dimensions.
      :rtype: TensorView

      **Note**: For SBUF tensors, partition dimension (dim 0) must remain at position 0.

   .. py:method:: broadcast(dim, size)

      Expands a size-1 dimension to a larger size without copying.

      :param dim: Dimension to broadcast (must have size 1).
      :type dim: int
      :param size: New size for the dimension.
      :type size: int
      :return: New TensorView with broadcasted dimension.
      :rtype: TensorView

   .. py:method:: reshape_dim(dim, shape)

      Reshapes a single dimension into multiple dimensions.

      :param dim: Dimension to reshape.
      :type dim: int
      :param shape: New sizes (can contain one -1 for inference).
      :type shape: tuple[int, ...]
      :return: New TensorView with reshaped dimension.
      :rtype: TensorView

   .. py:method:: flatten_dims(start_dim, end_dim)

      Flattens a range of contiguous dimensions into one.

      :param start_dim: First dimension to flatten (inclusive).
      :type start_dim: int
      :param end_dim: Last dimension to flatten (inclusive).
      :type end_dim: int
      :return: New TensorView with flattened dimensions.
      :rtype: TensorView

   .. py:method:: expand_dim(dim)

      Inserts a new dimension of size 1.

      :param dim: Position to insert the new dimension.
      :type dim: int
      :return: New TensorView with added dimension.
      :rtype: TensorView

   .. py:method:: squeeze_dim(dim)

      Removes a dimension of size 1.

      :param dim: Dimension to remove (must have size 1).
      :type dim: int
      :return: New TensorView with removed dimension.
      :rtype: TensorView

   .. py:method:: select(dim, index)

      Selects a single element along a dimension, reducing dimensionality.

      :param dim: Dimension to select from.
      :type dim: int
      :param index: Index to select (int for static, nl.ndarray for dynamic).
      :type index: int | nl.ndarray
      :return: New TensorView with one fewer dimension.
      :rtype: TensorView

   .. py:method:: rearrange(src_pattern, dst_pattern, fixed_sizes=None)

      Rearranges dimensions using einops-style patterns.

      :param src_pattern: Source dimension pattern with named dimensions.
      :type src_pattern: tuple[str | tuple[str, ...], ...]
      :param dst_pattern: Destination dimension pattern.
      :type dst_pattern: tuple[str | tuple[str, ...], ...]
      :param fixed_sizes: Dictionary mapping dimension names to sizes.
      :type fixed_sizes: dict[str, int], optional
      :return: New TensorView with rearranged dimensions.
      :rtype: TensorView

   .. py:method:: reshape(new_shape)

      Reshapes the tensor to new dimensions.

      :param new_shape: New dimension shape.
      :type new_shape: tuple[int, ...]
      :return: New TensorView with reshaped dimensions.
      :rtype: TensorView

      .. note:: General reshape is not yet implemented and will raise an error. Use ``reshape_dim`` for single-dimension reshaping.

   .. py:method:: has_dynamic_access()

      Checks if the tensor view uses dynamic indexing (via a prior ``select`` with an ``nl.ndarray`` index).

      :return: True if the view has dynamic access, False otherwise.
      :rtype: bool

Examples
--------

Reshape and Permute
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl
   from nkilib.core.utils.tensor_view import TensorView

   @nki.jit
   def kernel_reshape_permute(data_sb):
       view = TensorView(data_sb)  # Shape: (128, 24, 64)
       
       reshaped = view.reshape_dim(1, (4, 6))  # (128, 4, 6, 64)
       transposed = reshaped.permute((0, 2, 1, 3))  # (128, 6, 4, 64)
       
       result = transposed.get_view()

Slicing with Step
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tensor_view import TensorView

   @nki.jit
   def kernel_strided_slice(data_sb):
       view = TensorView(data_sb)  # Shape: (128, 256)
       
       # Take every other element: indices 0, 2, 4, ...
       strided = view.slice(dim=1, start=0, end=256, step=2)  # (128, 128)
       
       result = strided.get_view()

Broadcasting
^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tensor_view import TensorView

   @nki.jit
   def kernel_broadcast(scale_sb, data_sb):
       # scale_sb shape: (128, 1, 64)
       # data_sb shape: (128, 32, 64)
       
       scale_view = TensorView(scale_sb)
       
       # Broadcast dim 1 from size 1 to 32
       broadcasted = scale_view.broadcast(dim=1, size=32)  # (128, 32, 64)
       
       # Now can multiply element-wise
       result = data_sb * broadcasted.get_view()

Einops-Style Rearrange
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tensor_view import TensorView

   @nki.jit
   def kernel_rearrange(data_sb):
       view = TensorView(data_sb)  # Shape: (128, 512, 64)
       
       # Reshape and transpose: (p, h*w, c) -> (p, c, h, w)
       # where h=32 (must specify one dimension for -1 inference)
       rearranged = view.rearrange(
           src_pattern=('p', ('h', 'w'), 'c'),
           dst_pattern=('p', 'c', 'h', 'w'),
           fixed_sizes={'h': 32}
       )  # (128, 64, 32, 16)
       
       result = rearranged.get_view()

Chained Operations
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tensor_view import TensorView

   @nki.jit
   def attention_reshape(qkv_sb, num_heads, head_dim):
       # qkv_sb shape: (128, seq_len, 3 * num_heads * head_dim)
       view = TensorView(qkv_sb)
       
       # Chain: reshape -> slice Q -> reshape to heads
       q_view = (view
           .reshape_dim(2, (3, num_heads, head_dim))  # (128, S, 3, H, D)
           .select(dim=2, index=0)                     # (128, S, H, D) - select Q
           .permute((0, 2, 1, 3)))                     # (128, H, S, D)
       
       q = q_view.get_view()

See Also
--------

* :doc:`stream_shuffle_broadcast </nki/library/kernel-utils/stream-shuffle-broadcast>` - Hardware broadcast for partition dimension
* :doc:`SbufManager </nki/library/kernel-utils/allocator>` - Memory allocation with scope management
