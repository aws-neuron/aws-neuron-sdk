.. meta::
    :description: Cumsum kernel computes cumulative sum along the last dimension.
    :date-modified: 01/21/2026

.. currentmodule:: nkilib.core.cumsum

Cumsum Kernel API Reference
============================

Computes cumulative sum along the last dimension of the input tensor. Optimized for batch sizes up to 2048 and hidden dimension sizes up to 8192. Supports 3D inputs with sequence length up to 10.

The kernel supports:

* Cumulative sum computation along the last dimension only
* 2D and 3D input tensors
* Float32 accumulation for numerical stability
* Efficient tiled processing for large tensors
* Sequential processing to maintain cumulative dependencies

Background
--------------

The ``cumsum`` kernel implements cumulative sum computation, where each element in the output is the sum of all preceding elements (including itself) along the specified dimension. This operation is commonly used in various machine learning applications including attention mechanisms and sequence processing.

The kernel applies the following transformation along the last dimension:

* ``out[..., i] = sum(x[..., 0:i+1])``

The implementation uses ``tensor_tensor_scan`` operations with float32 accumulation for numerical stability, processing data in tiles to handle large tensors efficiently.

API Reference
----------------

**Source code for this kernel API can be found at**: `cumsum.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/cumsum/cumsum.py>`_

cumsum
^^^^^^^^^^^^^^^

.. py:function:: cumsum(x, axis=-1)

   Compute cumulative sum along the last dimension.

   :param x: Input tensor of shape ``[B, H]`` for 2D or ``[B, S, H]`` for 3D in HBM
   :type x: ``nl.ndarray``
   :param axis: Axis along which to compute cumsum. Must be -1 or the last dimension index. Default is -1.
   :type axis: ``int``, optional
   :return: Output tensor with same shape and dtype as input, containing cumulative sums along the last dimension
   :rtype: ``nl.ndarray``

   **Constraints**:

   * Only supports cumsum along the last dimension (axis=-1)
   * Batch size (``B``) must be up to 2048
   * Hidden dimension size (``H``) must be up to 8192
   * Sequence length (``S``) for 3D inputs must be up to 10
   * Input tensor must be 2D or 3D
   * For very long hidden dimensions (>5K), expect ~1e-2 absolute error due to fp32 accumulation

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Tiled Processing**: Processes data in tiles to handle large tensors efficiently:
   
   * **Partition Tiles**: Up to 128 elements per partition tile
   * **Free Dimension Tiles**: Up to 2048 elements per free dimension tile
   * Sequential processing across free dimension tiles to maintain cumulative dependencies

2. **Numerical Stability**: Uses float32 accumulation internally regardless of input dtype to maintain numerical precision for long sequences.

3. **Tensor Scan Operations**: Leverages ``tensor_tensor_scan`` with multiply and add operations to compute cumulative sums efficiently:
   
   * ``result[i] = ones[i] * result[i-1] + data[i] = result[i-1] + data[i]``

4. **Carry Forward**: Maintains cumulative state across tiles by carrying forward the last column of each processed tile as the initial value for the next tile.

5. **Memory Management**: Efficiently manages SBUF allocations for intermediate buffers and uses DMA operations for HBM transfers.