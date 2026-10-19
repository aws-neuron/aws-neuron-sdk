.. meta::
    :description: API reference for the stream_shuffle_broadcast utility in the NKI Library.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.core.utils.stream_shuffle_broadcast

stream_shuffle_broadcast API Reference
======================================

This topic provides the API reference for the ``stream_shuffle_broadcast`` utility. It broadcasts the first partition of a tensor across all partitions using hardware shuffle instructions.

When to Use
-----------

Use ``stream_shuffle_broadcast`` when you need to:

* **Broadcast scalar or vector values**: Replicate a single partition's data across all partitions
* **Distribute bias terms**: Broadcast bias vectors to match activation tensor partition layout
* **Replicate position IDs**: Broadcast position encodings across batch partitions

This utility is commonly used after loading a 1-partition tensor (like a bias or scale) that needs to be applied element-wise across a multi-partition activation tensor.

API Reference
-------------

**Source code**: https://github.com/aws-neuron/nki-library

stream_shuffle_broadcast
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: stream_shuffle_broadcast(src, dst)

   Broadcasts the first partition of ``src`` onto all partitions of ``dst``.

   Both tensors must be in SBUF and 2D. The function uses ``nisa.nc_stream_shuffle`` with a zero shuffle mask to replicate partition 0 across all destination partitions.

   :param src: 2D source tensor in SBUF. Only partition 0 is read.
   :type src: nl.ndarray
   :param dst: 2D destination tensor in SBUF. All partitions are written.
   :type dst: nl.ndarray
   :rtype: None

   **Constraints**:

   * Both ``src`` and ``dst`` must be 2D tensors
   * ``src.shape[1]`` must equal ``dst.shape[1]`` (matching free dimension)
   * Both tensors must be in SBUF

   **Raises**:

   * **AssertionError** – If tensors are not 2D or free dimensions don't match.

Examples
--------

Without stream_shuffle_broadcast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl

   @nki.jit
   def kernel_without_broadcast(bias_hbm, activation_sb):
       # Load bias (1 partition)
       bias_sb = nl.ndarray((1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
       bias_sb[0, :] = nl.load(bias_hbm[0, :])
       
       # Manual broadcast - verbose and error-prone
       bias_broadcast = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
       for p in nl.affine_range(128):
           bias_broadcast[p, :] = bias_sb[0, :]
       
       # Now can add to activation
       result = activation_sb + bias_broadcast

With stream_shuffle_broadcast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl
   from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

   @nki.jit
   def kernel_with_broadcast(bias_hbm, activation_sb):
       # Load bias (1 partition)
       bias_sb = nl.ndarray((1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
       bias_sb[0, :] = nl.load(bias_hbm[0, :])
       
       # Efficient hardware broadcast
       bias_broadcast = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
       stream_shuffle_broadcast(src=bias_sb, dst=bias_broadcast)
       
       # Now can add to activation
       result = activation_sb + bias_broadcast

In-Place Broadcast
^^^^^^^^^^^^^^^^^^

The function reads only from partition 0 of ``src``, so ``src`` and ``dst`` can be the same tensor:

.. code-block:: python

   import nki.language as nl
   from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

   @nki.jit
   def kernel_inplace_broadcast(scale_hbm, num_partitions):
       # Allocate full-size buffer, load to partition 0
       scale_sb = nl.ndarray((num_partitions, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
       scale_sb[0, :] = nl.load(scale_hbm[0, :])
       
       # Broadcast in-place: reads src[0:1, :], writes to all dst partitions
       stream_shuffle_broadcast(src=scale_sb, dst=scale_sb)
