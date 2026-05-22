.. meta::
    :description: Performs matrix multiplication with MXFP8 quantization.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.matmul_mxfp8

Matmul MXFP8 Kernel API Reference
=================================

Performs matrix multiplication with MXFP8 quantization.

This kernel implements efficient matrix multiplication using MXFP8 quantization format, supporting both pre-quantized inputs and automatic quantization from BF16. The kernel uses hardware-optimized tiling and supports LNC2 parallelization for improved throughput.

Background
-----------

The ``matmul_mxfp8`` kernel implements efficient matrix multiplication using MXFP8 quantization format, supporting both pre-quantized inputs and automatic quantization from BF16. It uses hardware-optimized tiling and supports LNC2 parallelization for improved throughput.

API Reference
--------------

**Source code for this kernel API can be found at**: `matmul_mxfp8_generic_kernel.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_generic_kernel.py>`_

matmul_mxfp8
^^^^^^^^^^^^

.. py:function:: matmul_mxfp8(lhs, rhs, TILES_IN_BLOCK_M: int = None, TILES_IN_BLOCK_N: int = None, TILES_IN_BLOCK_K: int = None, TILES_IN_LOAD_M: int = None, TILES_IN_LOAD_N: int = None, lhs_matmul_tile_shape_logical: tuple = None, rhs_matmul_tile_shape_logical: tuple = None, block_loop_order: str = 'mnk', tile_loop_order: str = 'mnk', float8_dtype: str = 'float8_e5m2', output_dtype = nl.float32, run_with_lnc2: bool = True, lnc_2_shard_rhs: bool = True, lhs_scales = None, rhs_scales = None, use_scale_packing: bool = False, spill_reload: bool = False, lhs_is_swizzled: bool = True, rhs_is_swizzled: bool = True) -> nl.ndarray

   Performs matrix multiplication with MXFP8 quantization.

   :param lhs: Left-hand side matrix, either BF16 tensor or tuple (data, scales) for pre-quantized MXFP8
   :param rhs: Right-hand side matrix, either BF16 tensor or tuple (data, scales) for pre-quantized MXFP8
   :param TILES_IN_BLOCK_M: Number of matmul tiles per block in M dimension (auto-generated if None)
   :type TILES_IN_BLOCK_M: ``int``
   :param TILES_IN_BLOCK_N: Number of matmul tiles per block in N dimension (auto-generated if None)
   :type TILES_IN_BLOCK_N: ``int``
   :param TILES_IN_BLOCK_K: Number of matmul tiles per block in K dimension (auto-generated if None)
   :type TILES_IN_BLOCK_K: ``int``
   :param TILES_IN_LOAD_M: Number of tiles to load at once in M dimension (auto-generated if None)
   :type TILES_IN_LOAD_M: ``int``
   :param TILES_IN_LOAD_N: Number of tiles to load at once in N dimension (auto-generated if None)
   :type TILES_IN_LOAD_N: ``int``
   :param lhs_matmul_tile_shape_logical: LHS tile shape (TILE_K, TILE_M) in logical space (auto-generated if None)
   :type lhs_matmul_tile_shape_logical: ``tuple``
   :param rhs_matmul_tile_shape_logical: RHS tile shape (TILE_K, TILE_N) in logical space (auto-generated if None)
   :type rhs_matmul_tile_shape_logical: ``tuple``
   :param block_loop_order: Block processing order, default 'mnk'
   :type block_loop_order: ``str``
   :param tile_loop_order: Tile processing order within blocks, default 'mnk'
   :type tile_loop_order: ``str``
   :param float8_dtype: FP8 dtype for quantization, default "float8_e5m2"
   :type float8_dtype: ``str``
   :param output_dtype: Output data type, default nl.float32
   :param run_with_lnc2: Enable LNC2 parallelization across 2 cores, default True
   :type run_with_lnc2: ``bool``
   :param lnc_2_shard_rhs: When run_with_lnc2=True, shard on N dimension (RHS) if True, or shard on M dimension (LHS) if False. Default True.
   :type lnc_2_shard_rhs: ``bool``
   :param lhs_scales: Optional pre-computed scales for LHS
   :param rhs_scales: Optional pre-computed scales for RHS
   :param use_scale_packing: If True and inputs are pre-quantized, assert that scales are packed, default False
   :type use_scale_packing: ``bool``
   :param spill_reload: If True, each quantized block will be written to HBM and on every subsequent load, this spilled block will be reloaded.
   :type spill_reload: ``bool``
   :param lhs_is_swizzled: Whether LHS BF16 tensor is pre-swizzled [K/4, M*4], default True. If False, expects [M, K] layout.
   :type lhs_is_swizzled: ``bool``
   :param rhs_is_swizzled: Whether RHS BF16 tensor is pre-swizzled [K/4, N*4], default True. If False, expects [N, K] layout.
   :type rhs_is_swizzled: ``bool``

   **Notes**:

   * Supports non-divisible tensor shapes using dynamic slicing (nl.ds)
   * Auto-generates optimal tiling parameters when not specified
   * LNC2 mode requires at least 2 blocks in N dimension
   * Pre-quantized inputs must be in MXFP8 format (data, scales) tuple
   * When use_scale_packing=True, pre-quantized inputs must have packed scales
   * TODO: Specify intended usage range for optimal performance Physical vs Logical Dimensions:
   * Logical: Theoretical tensor dimensions [M, K] @ [K, N] for the matmul operation
   * Physical: Hardware storage format (depends on quantization and swizzling) * Pre-swizzled: [K//4, M*4] or [K//4, N*4] * Quantized: [K//4, M] or [K//4, N] Tiles (smallest processing unit):
   * Matmul Tile: Hardware matmul operation shape * LHS: [128, 128] physical, [512, 128] logical * RHS: [128, 512] physical, [512, 512] logical
   * Load Tile: Data loaded per matmul tile (varies by quantization state)
   * Quantize Tile: Input shape for quantization to produce one matmul tile Blocks (collection of tiles):
   * Group of tiles processed together
   * Must fit in SBUF (including load, quantize, and output buffers)
   * Accumulates results across K dimension before storing to HBM Non-Divisible Shape Handling:
   * Uses ceiling division for block counts
   * Applies nl.ds (dynamic slice) for boundary handling at load and store operations Example:: import nki.language as nl # Basic usage with BF16 inputs lhs = nl.ndarray((512, 1024), dtype=nl.bfloat16, buffer=nl.hbm) rhs = nl.ndarray((512, 2048), dtype=nl.bfloat16, buffer=nl.hbm) result = matmul_mxfp8( lhs=lhs, rhs=rhs, TILES_IN_BLOCK_M=2, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=1, TILES_IN_LOAD_M=1, TILES_IN_LOAD_N=1, lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, 512), ) # Usage with pre-quantized inputs (tuple of data and scales) lhs_quantized = (lhs_data, lhs_scales) result = matmul_mxfp8(lhs=lhs_quantized, rhs=rhs, ...)

   **Dimensions**:

   * M: Number of rows in left-hand side matrix (output rows)
   * K: Contraction dimension (columns in LHS, rows in RHS)

