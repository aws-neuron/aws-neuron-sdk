"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI implementation for matrix multiplication NKI tutorial.

"""

import nki as nki
import nki.isa as nisa
import nki.language as nl
import numpy as np


# NKI_EXAMPLE_16_BEGIN
@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [64,512]
  """
  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape

  # Check that the contraction dimension matches and all dimensions
  #are what were expected.
  assert K == K_, \
    f"Expected contraction dimension to match on both lhsT ({K}) and rhs ({K})"
  assert K == 128, f"Expected contraction dimension to be 128, but got {K}"
  assert M == 64, f"Expected lhsT matrix to have dimension M of 64, but got {M}"
  assert N == 512, f"Expected rhs matrix to have dimension N of 512, but got {N}"

  # Create a tensor to write the result into (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Creating a tensor in SBUF to load the inputs into (not initialized)
  lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)

  # Loading the inputs (HBM->SBUF)
  # Note: here we take Tile dtype definition into account,
  # which forces P-dim as the left most index
  nisa.dma_copy(dst=lhs_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)

  # Create a tensor in PSUM to accumulate the result in (uninitialized)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)

  # Perform the matrix-multiplication
  # Note: A NKI matmul instruction always writes to PSUM in float32 data-type
  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)

  # Create a tensor in SBUF and copy the result from PSUM back to SBUF, 
  # and cast to expected output data-type
  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=result.dtype)

  # The result of [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
  # This dictates which indices to use to address the result tile.
  nisa.dma_copy(dst=result, src=result_sbuf)

  return result
  # NKI_EXAMPLE_16_END


# NKI_EXAMPLE_18_BEGIN
@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
  # NKI_EXAMPLE_18_END


# NKI_EXAMPLE_19_BEGIN
@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner
     while hoisting the load of the lhsT and rhs to outer loops.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  # Verify that the lhsT and rhs are the expected sizes.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Verify that the input matrices are a multiple of the tile dimensions.
  assert M % TILE_M == 0, \
    f"Expected M, {M}, to be a multiple of stationary free-dimension max, {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected N, {N}, to be a multiple of moving free-dimension max, {TILE_N}"
  assert K % TILE_K == 0, \
    f"Expected K, {K}, to be a multiple of the partition dimension max, {TILE_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    # Load a whole column tiles from lhsT (with K * TILE_M numbers)
    # This corresponds to the whole row in the original lhs
    lhsT_tiles = []
    for k in nl.affine_range(K // TILE_K):
      # Allocate space in SBUF for the tile (uninitialized)
      lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
      # Copy the tile from HBM to SBUF
      nisa.dma_copy(dst=lhsT_tile, 
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                             m * TILE_M:(m + 1) * TILE_M])
      # Append the tile to the list of tiles.
      lhsT_tiles.append(lhsT_tile)

    for n in nl.affine_range(N // TILE_N):
      # Load a whole column tiles from rhs (with K * TILE_N numbers)
      rhs_tiles = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=rhs_tile,
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])
        # Append the tile to the list of tiles.
        rhs_tiles.append(rhs_tile)

      # Allocate a tile in PSUM for the result (uninitialized)
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        # Accumulate partial-sums into PSUM
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)

      # Copy the result from SBUF to HBM.
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                               n * TILE_N:(n + 1) * TILE_N],
                    src=res_sb)

  return result
  # NKI_EXAMPLE_19_END


# NKI_EXAMPLE_20_BEGIN
@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  """NKI kernel to compute a matrix multiplication operation while blocking the
     free dimensions of the LHS and RHS to improve memory access pattern.

  Args:
      lhsT: an input tensor of shape [K,M], where both K and M are multiples for
        128.  It is the left-hand-side argument of the matrix multiplication,
        delivered transposed for optimal performance.
      rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
        is a multiple of 512.  It is the right-hand-side argument of the matrix
        multiplication.
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Configuring the blocking size for the free dimensions
  TILES_IN_BLOCK_M = 2
  TILES_IN_BLOCK_N = 2

  BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

  # the size has to be multiple of block size
  assert M % BLOCK_M == 0
  assert N % BLOCK_N == 0

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Loop over blocks over the M dimension
  for m in nl.affine_range(M // BLOCK_M):
    # Load TILES_IN_BLOCK_M columns tiles by TILES_K rows from lhsT
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      # Inner tile array.
      lhsT_tiles_internal = []
      for k in nl.affine_range(K // TILE_K):
        # Allocate space in SBUF for the tile (uninitialized)
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M),
                               dtype=lhsT.dtype,
                               buffer=nl.sbuf)
        # Copy the tile from HBM to SBUF
        nisa.dma_copy(dst=lhsT_tile,
                      src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                               (m * TILES_IN_BLOCK_M + bm) *
                               TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                               TILE_M])
        # Append the tile to the inner list of tiles.
        lhsT_tiles_internal.append(lhsT_tile)
      # Append the inner list of tiles into the outer list of tiles.
      lhsT_tiles.append(lhsT_tiles_internal)

    for n in nl.affine_range(N // BLOCK_N):
      # Load TILES_IN_BLOCK_N columns from rhs by TILES_K rows from rhs
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        # Inner tile array.
        rhs_tiles_internal = []
        for k in nl.affine_range(K // TILE_K):
          # Allocate space in SBUF for the tile (uninitialized)
          rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N),
                                dtype=rhs.dtype,
                                buffer=nl.sbuf)
          # Copy the tile from HBM to SBUF
          nisa.dma_copy(dst=rhs_tile,
                        src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                (n * TILES_IN_BLOCK_N + bn) *
                                TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                TILE_N])
          # Append the tile to the inner list of tiles.
          rhs_tiles_internal.append(rhs_tile)
        # Append the inner list of tiles into the outer list of tiles.
        rhs_tiles.append(rhs_tiles_internal)

      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          # Allocate a tensor in PSUM
          result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                   dtype=nl.float32,
                                   buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            # Accumulate partial-sums into PSUM
            nisa.nc_matmul(dst=result_tile,
                           stationary=lhsT_tiles[bm][k],
                           moving=rhs_tiles[bn][k])
  
          # Copy the result from PSUM back to SBUF, and cast to expected
          # output data-type
          result_tmp = nl.ndarray(shape=result_tile.shape,
                                  dtype=result.dtype,
                                  buffer=nl.sbuf)
          nisa.tensor_copy(dst=result_tmp, src=result_tile)

          # Copy the result from SBUF to HBM.
          nisa.dma_copy(dst=result[(m * TILES_IN_BLOCK_M + bm) *
                                   TILE_M:((m * TILES_IN_BLOCK_M + bm) + 1) *
                                   TILE_M,
                                   (n * TILES_IN_BLOCK_N + bn) *
                                   TILE_N:((n * TILES_IN_BLOCK_N + bn) + 1) *
                                   TILE_N],
                        src=result_tmp)

  return result
  # NKI_EXAMPLE_20_END


# NKI_EXAMPLE_21_BEGIN
@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):
  """NKI kernel to compute a large matrix multiplication efficiently by
     blocking all dimensions and doing layout optimization.

  Args:
      lhsT: an input tensor of shape [K,M], where K is a multiple of 128 *
        TILES_IN_BLOCK_K and M is a multiple of 128 * TILES_IN_BLOCK_M.  It is the
        left-hand-side argument of the matrix multiplication, delivered transposed
        for optimal performance.
      rhs: an input tensor of shape [K,N],  where K is a multiple of 128 *
        TILES_IN_BLOCK_K and N is a multiple of 512 * TILES_IN_BLOCK_N.  It is
        the right-hand-side argument of the matrix multiplication.
      TILES_IN_BLOCK_*: meta parameters to control blocking dimensions
  Returns:
      result: the resulting output tensor of shape [M,N]
  """

  # Verify that the lhsT and rhs have the same contraction dimension.
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Lookup the device matrix multiply dimensions.
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  TILE_K = nl.tile_size.pmax  # 128
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Compute the block dimensions.
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K

  # Verify the size is a multiple of block size
  assert M % BLOCK_M == 0, \
    f"Expected M {M} to be divisble by {BLOCK_M} when there are {TILES_IN_BLOCK_M}"
  assert N % BLOCK_N == 0, \
    f"Expected N {N} to be divisble by {BLOCK_N} when there are {TILES_IN_BLOCK_N}"
  assert K % BLOCK_K == 0, \
    f"Expected K {K} to be divisble by {BLOCK_K} when there are {TILES_IN_BLOCK_K}"

  # Create a space for the result in HBM (not initialized)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Compute the number of blocks in each dimension
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  # Blocking N dimension (the RHS free dimension)
  for n in nl.affine_range(NUM_BLOCK_N):
    # Create the initial result tiles in SBUF and initialize each tile to
    # 0.0, since the final results will be accumulated here. Results in 3-d array.
    result_tmps = []
    for m_idx in range(NUM_BLOCK_M):
      block_m = []
      for bm_idx in range(TILES_IN_BLOCK_M):
        block_n = []
        for bn_idx in range(TILES_IN_BLOCK_N):
          # Create the result tile (uninitialized)
          tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
          # Initialize the tile 0.0
          nisa.memset(dst=tile, value=0.0)
          # Append the tile to block_n array.
          block_n.append(tile)
        # Append block_n array to block_m array.
        block_m.append(block_n)
      # Append block_m array into result_tmps.
      result_tmps.append(block_m)

    # Blocking K dimension (the contraction dimension)
    # Use `sequential_range` because we do not want the compiler
    # to change this loop by, for example, vectorizing it
    for k in nl.sequential_range(NUM_BLOCK_K):
      # Loading tiles from rhs
      # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
      rhs_tiles = []
      for bk_r in range(TILES_IN_BLOCK_K):
        # Allocate rhs_tile tensor, TILE_K x BLOCK_N
        rhs_tile = nl.ndarray(shape=(TILE_K, BLOCK_N),
                              dtype=rhs.dtype,
                              buffer=nl.sbuf)
        # Copy block tile from rhs, to rhs_tile.
        nisa.dma_copy(dst=rhs_tile[0:TILE_K, 0:BLOCK_N],
                      src=rhs[(TILES_IN_BLOCK_K * k + bk_r) *
                              TILE_K:(TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                              BLOCK_N * n:BLOCK_N * (n + 1)])
        # Append rhs_tile to rhs_tiles.
        rhs_tiles.append(rhs_tile)


      # Blocking M dimension (the LHS free dimension)
      for m in nl.affine_range(NUM_BLOCK_M):
        # Loading tiles from lhsT
        lhsT_tiles = []
        for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
          # Allocate lhsT_tile in SBUF (uninitialized)
          lhsT_tile = nl.ndarray(shape=(TILE_K, BLOCK_M),
                                 dtype=lhsT.dtype,
                                 buffer=nl.sbuf)
          # Copy block tile from lhsT to lhsT_tile
          nisa.dma_copy(dst=lhsT_tile[0:TILE_K, 0:BLOCK_M],
                        src=lhsT[(TILES_IN_BLOCK_K * k + bk_l) *
                                 TILE_K:(TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                 BLOCK_M * m:BLOCK_M * (m + 1)])
          # Append to list of lhsT tiles.
          lhsT_tiles.append(lhsT_tile)

        # Do matmul with all tiles in the blocks
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          for bm in nl.affine_range(TILES_IN_BLOCK_M):
            # Allocate result_tile in PSUM (uninitialized)
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N),
                                     dtype=nl.float32,
                                     buffer=nl.psum)
            for bk in nl.affine_range(TILES_IN_BLOCK_K):
              # Perform matrix multiply on a tile.
              nisa.nc_matmul(
                dst=result_tile,
                stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N]
              )
            # Accumulate the result into the result_tmps tile.
            nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                               data1=result_tmps[m][bm][bn],
                               data2=result_tile,
                               op=nl.add)

    # Copying the result from SBUF to HBM
    for m in nl.affine_range(NUM_BLOCK_M):
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        # coalesce result tiles for better DMA performance
        result_packed = nl.ndarray(shape=(TILE_M, BLOCK_N),
                                   dtype=nl.float32,
                                   buffer=nl.sbuf)
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          nisa.tensor_copy(
            dst=result_packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
            src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])

        # Copy packed result from SBUF to HBM.
        nisa.dma_copy(dst=result[(TILES_IN_BLOCK_M * m + bm) *
                                 TILE_M:(TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                 BLOCK_N * n:BLOCK_N * (n + 1)],
                      src=result_packed[0:TILE_M, 0:BLOCK_N])

  return result
# NKI_EXAMPLE_21_END
