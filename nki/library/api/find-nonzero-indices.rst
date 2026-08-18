.. meta::
    :description: Find indices of nonzero elements along the T dimension.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.core.subkernels

Find Nonzero Indices Subkernel API Reference
=============================================

Finds indices of nonzero elements along the T dimension.

The kernel supports:

* Finding nonzero indices in an input tensor of shape [T, C]
* LNC2 sharding across columns
* GpSimd ``nonzero_with_count`` ISA for parallel processing
* Token counts up to 65536 and column counts up to 128
* Optional column subsetting via ``col_start_id`` and ``n_cols``

Background
-----------

The ``find_nonzero_indices`` subkernel computes the indices of nonzero elements along the T dimension for each column of an input tensor. It uses the GpSimd ``nonzero_with_count`` ISA instruction for parallel processing of 8 columns at a time, with LNC2 sharding across the column dimension.

API Reference
--------------

**Source code for this kernel API can be found at**: `find_nonzero_indices.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/core/subkernels/find_nonzero_indices.py>`_

find_nonzero_indices
^^^^^^^^^^^^^^^^^^^^

.. py:function:: find_nonzero_indices(input_tensor: nl.ndarray, col_start_id: nl.ndarray = None, n_cols: int = None, chunk_size: int = None, index_dtype: nki.dtype = nl.int32)

   Find indices of nonzero elements along the T dimension.

   :param input_tensor: [T, C], Input tensor on HBM. Nonzero elements are found along the T dimension for each column.
   :type input_tensor: ``nl.ndarray``
   :param col_start_id: [1], Optional HBM tensor containing the starting column index in the C dimension. If specified, only n_cols Columns starting from col_start_id are processed. If None, all C Columns are processed.
   :type col_start_id: ``nl.ndarray``
   :param n_cols: Number of columns (in C dimension) to process. Required when col_start_id is specified, ignored otherwise.
   :type n_cols: ``int``
   :param chunk_size: Size of chunks for processing T dimension. If None, defaults to T. Must divide T evenly. Smaller chunk sizes reduce memory usage.
   :type chunk_size: ``int``
   :param index_dtype: Data type for output indices tensor. Default is nl.int32.
   :type index_dtype: ``nki.dtype``
   :return: [C, T] or [n_cols, T], Tensor containing nonzero indices. For each column c, the first N values are the T-indices of nonzero elements, followed by -1 padding values.
   :rtype: ``nl.ndarray``
   :return: [C] or [n_cols], Count of nonzero elements per column.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Requires LNC2 configuration (2 NeuronCores)
   * C must be divisible by 2 (for LNC2 sharding)
   * chunk_size must be divisible by 128 (partition size)
   * Uses GpSimd nonzero_with_count ISA which only operates on partitions [0, 16, 32, ..., 112]

   **Dimensions**:

   * T: Sequence/token dimension (first dimension of input)
   * C: Column dimension that used to calculate the non zero indices (second dimension of input)
   * C_full: Full columns dimension from input tensor shape

