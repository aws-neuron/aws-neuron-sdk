.. meta::
    :description: Determine if packed scales should be stored for the current tile.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.quantize_mxfp8

Quantize Mxfp8 Kernel API Reference
===================================

Determine if packed scales should be stored for the current tile.

Background
-----------

The ``should_store_packed_scales`` kernel determines whether packed scales should be stored for the current tile during MXFP8 block-wise quantization.

API Reference
--------------

**Source code for this kernel API can be found at**: `quantize_mxfp8.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/quantize_mxfp8/quantize_mxfp8.py>`_

should_store_packed_scales
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: should_store_packed_scales(tile_k_idx: int, total_num_tiles: int) -> bool

   Determine if packed scales should be stored for the current tile.

   :param tile_k_idx: Current tile index in K dimension (across all tiles including remainder)
   :type tile_k_idx: ``int``
   :param total_num_tiles: Total number of tiles including remainder tiles
   :type total_num_tiles: ``int``
   :return: True if scales should be stored, False otherwise
   :rtype: ``nl.ndarray``

quantize_block_mxfp8_kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: quantize_block_mxfp8_kernel(src_tensor: nl.ndarray, return_fp8_dtype: str, run_with_lnc2: bool = False, enable_scale_packing: bool = True) -> tuple[nl.ndarray, nl.ndarray]

   Kernel for quantizing BF16 tensor to MXFP8 format with block-wise quantization.

   :param src_tensor: [F, K], Input tensor in BF16 format on HBM
   :type src_tensor: ``nl.ndarray``
   :param return_fp8_dtype: FP8 dtype string like "float8_e4m3fn" or "float8_e5m2"
   :type return_fp8_dtype: ``str``
   :param run_with_lnc2: Enable LNC2 parallelization along F dimension (default: False)
   :type run_with_lnc2: ``bool``
   :param enable_scale_packing: Enable scale packing optimization (default: True)
   :type enable_scale_packing: ``bool``
   :return: [K // 4, F], Scales in uint8 format on HBM
   :rtype: ``nl.ndarray``
   :return: [K // 4, F * INTERLEAVE_FACTOR], Quantized data in FP8 format on HBM
   :rtype: ``nl.ndarray``

   **Notes**:

   * K dimension must be divisible by 512 for mxfp8 quantization
   * F dimension must be divisible by 8 for quantization
   * LNC2 splits work along F dimension; supports uneven splits

   **Dimensions**:

   * F: Feature dimension (rows in input tensor)

