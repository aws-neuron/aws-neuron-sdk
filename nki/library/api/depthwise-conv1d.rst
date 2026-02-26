.. meta::
    :description: Depthwise Conv1D kernel using implicit GEMM approach for TRN2.
    :date-modified: 02/06/2025

.. currentmodule:: nkilib.experimental.conv

Depthwise Conv1D Kernel API Reference
======================================

Implements depthwise 1D convolution using implicit GEMM without full im2col materialization.

The kernel supports:

* Depthwise 1D convolution with stride=1 and zero padding
* Implicit GEMM approach for memory efficiency
* LNC2 sharding on channel dimension
* Optimized for TRN2 platform

Background
-----------

The ``depthwise_conv1d_implicit_gemm`` kernel performs depthwise 1D convolution by loading input with shape [S_TILE, Q] where row k contains elements starting at index k (i.e., input[k:k+Q]), enabling implicit im2col via offset-based loading. This approach avoids materializing the full im2col matrix, saving W*S*C memory. The kernel tiles on S dimension for S > 128 and is optimized for TRN2 platform with LNC2 sharding on channel dimension.

API Reference
--------------

**Source code for this kernel API can be found at**: `depthwise_conv1d.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/depthwise_conv1d.py>`_

depthwise_conv1d_implicit_gemm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: depthwise_conv1d_implicit_gemm(img_ref: nl.ndarray, filter_ref: nl.ndarray, padding: tuple = ((0, 0), (0, 0)), stride: tuple = (1, 1), rhs_dilation: tuple = (1, 1), lhs_dilation: tuple = (1, 1), feature_group_count: int = 1, batch_group_count: int = 1, in_perm: tuple = None, kern_perm: tuple = None, out_perm: tuple = None) -> nl.ndarray

   Depthwise Conv1D using implicit GEMM without full im2col materialization.

   Performs depthwise 1D convolution by loading input with shape [S_TILE, Q] where
   row k contains elements starting at index k (i.e., input[k:k+Q]), enabling implicit
   im2col via offset-based loading. Tiles on S dimension for S > 128. Optimized for
   TRN2 platform with LNC2 sharding on channel dimension.

   :param img_ref: Input tensor on HBM with shape [N, C, 1, W].
   :type img_ref: ``nl.ndarray``
   :param filter_ref: Depthwise kernel weights on HBM with shape [C, 1, 1, S].
   :type filter_ref: ``nl.ndarray``
   :param padding: Padding as ((H_pad_l, H_pad_r), (W_pad_l, W_pad_r)). Default: ((0,0),(0,0)), only zeros supported.
   :type padding: ``tuple``
   :param stride: Stride values. Default: (1, 1), only (1, 1) supported.
   :type stride: ``tuple``
   :param rhs_dilation: RHS dilation. Default: (1, 1).
   :type rhs_dilation: ``tuple``
   :param lhs_dilation: LHS dilation. Default: (1, 1).
   :type lhs_dilation: ``tuple``
   :param feature_group_count: Number of feature groups. Default: 1.
   :type feature_group_count: ``int``
   :param batch_group_count: Number of batch groups. Default: 1.
   :type batch_group_count: ``int``
   :param in_perm: Input permutation. Default: None.
   :type in_perm: ``tuple``, optional
   :param kern_perm: Kernel permutation. Default: None.
   :type kern_perm: ``tuple``, optional
   :param out_perm: Output permutation. Default: None.
   :type out_perm: ``tuple``, optional
   :return: Convolution output on HBM with shape [N, C, 1, Q] where Q = W - S + 1.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Only supports stride=1 and zero padding
   * Requires C to be divisible by NUM_SHARDS (2)
   * Uses LNC2 sharding on channel dimension
   * For depthwise convolution, feature_group_count must equal C

   **Dimensions**:

   * N: Batch size
   * C: Number of channels
   * W: Input width (spatial dimension)
   * S: Kernel size
   * Q: Output width (W - S + 1)

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **Implicit GEMM Approach**: Avoids materializing full im2col matrix by using offset-based loading patterns, saving W*S*C memory.

2. **Tiling Strategy**: 
   - Input: [N, C, W] tiled as [N, C_TILES, C_TILE] x [S_TILES, S_TILE, Q]
   - Filter: [C, S] tiled as [C_TILES, C_TILE] x [S_TILES, S_TILE]
   - Output: [N, C, Q] accumulated in [Q_TILES, Q_TILE] chunks

3. **Tile Size Selection**:
   - S_TILE = min(S, 128): Matches partition dimension (P_MAX=128)
   - Q_TILE = min(Q, 512): Matches free dimension (F_MAX=512)
   - C_TILE = min(C_per_shard, 128): Balances parallelism and memory

4. **Filter Preloading**: Amortizes transpose cost across channels by preloading filter tiles in outer loop.

5. **Sequential S-tile Accumulation**: Enables pipelining and reduces PSUM pressure.

6. **LNC2 Sharding**: Distributes computation across channel dimension for parallel processing.