.. meta::
    :description: 1D Convolution operation using tensor engine with replication strategy.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.experimental.conv

Conv1D Kernel API Reference
===========================

Implements 1D convolution using tensor engine with a replication strategy for efficient computation.

The kernel supports:

* Arbitrary stride, padding, and dilation values
* Optional bias addition
* Activation function fusion
* LNC sharding on the output channel dimension

Intended usage range:

* Kernel size (K): 1 to 128
* Sequence length (L): 1 to 4096
* Input channels (C_in): 1 to 4096
* Output channels (C_out): 1 to 4096
* Batch size (B): Any positive integer

Background
-----------

The ``conv1d`` kernel applies 1D convolution filters across the input sequence dimension. It uses a replication strategy to efficiently utilize the tensor engine by stacking multiple filter positions along the partition dimension.

API Reference
--------------

**Source code for this kernel API can be found at**: `conv1d.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/conv/conv1d.py>`_

conv1d
^^^^^^

.. py:function:: conv1d(x_in: nl.ndarray, filters: nl.ndarray, bias: Optional[nl.ndarray] = None, stride: int = 1, padding: tuple[int, int] = (0, 0), dilation: int = 1, activation_fn: Optional[ActFnType] = None, lnc_shard: bool = False) -> nl.ndarray

   1D Convolution operation using tensor engine with replication strategy.

   :param x_in: [B, C_in, L], Input tensor on HBM.
   :type x_in: ``nl.ndarray``
   :param filters: [K, C_in, C_out], Convolution filter weights on HBM.
   :type filters: ``nl.ndarray``
   :param bias: [C_out], Optional bias tensor on HBM. Default None.
   :type bias: ``Optional[nl.ndarray]``
   :param stride: Stride for convolution. Must be >= 1. Default 1.
   :type stride: ``int``
   :param padding: Tuple of (left_pad, right_pad). Must be non-negative. Default (0, 0).
   :type padding: ``tuple[int, int]``
   :param dilation: Dilation factor for dilated convolution. Must be >= 1. Default 1.
   :type dilation: ``int``
   :param activation_fn: Optional activation function to fuse. Default None.
   :type activation_fn: ``Optional[ActFnType]``
   :param lnc_shard: If True, shard computation across LNC cores on C_out dimension. Default False.
   :type lnc_shard: ``bool``
   :return: [B, C_out, L_out], Output tensor on HBM where L_out = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1
   :rtype: ``nl.ndarray``

   **Notes**:

   * All input tensors (x_in, filters, bias) must have the same dtype
   * Input channels C_in must match filter channels
   * Uses replication strategy to stack K filter positions along partition dimension
   * Partition alignment rules limit K replication factor based on C_in tile size
   * Memory management uses SbufManager with multi-buffering for efficiency

   **Dimensions**:

   * B: Batch size
   * C_in: Number of input channels
   * C_out: Number of output channels
   * L: Input sequence length
   * L_out: Output sequence length = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1

