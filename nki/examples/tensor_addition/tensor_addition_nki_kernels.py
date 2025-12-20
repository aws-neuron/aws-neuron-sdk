"""
Copyright (C) 2025, Amazon.com. All Rights Reserved

NKI implementation for tensor addition NKI tutorial.

"""
# NKI_EXAMPLE_27_BEGIN
import nki as nki
import nki.language as nl
import nki.isa as nisa

@nki.jit(platform_target="trn1")
def nki_tensor_add(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output sizes can be uniformly tiled to [128,512]

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor shared between all SPMD instances as 
  # result tensor (uninitialized)
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Extract the dimensions for the a_input shape.
  M, N = a_input.shape

  # Set the tile dimensions, while the TILE_N is not, strictly speaking, limited to 
  # 512 for the additiona operation, we stick with this size for simplicity.
  TILE_M = 128
  TILE_N = 512

  # Check the input sizes match and match the tilable constraint.
  assert a_input.shape == b_input.shape, \
    f"Expected shaps {a_input.shape} and {b_input.shape} to match"
  assert a_input.dtype == b_input.dtype, \
    f"Expected data types {a_input.dtype} and {b_input.dtype} to match"
  assert M % TILE_M == 0, \
    f"Expected partition dimention ({M}) to be divisble by {TILE_M}"
  assert N % TILE_N == 0, \
    f"Expected partition dimention ({N}) to be divisble by {TILE_N}"

  # Lop over each tile, load the tile, do the addition, and save it back to HBM.
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocte space for the a_tile and b_tile in sbuf (uninitialized)
      a_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=a_input.dtype, buffer=nl.sbuf)
      b_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=b_input.dtype, buffer=nl.sbuf)

      # Load the a_tile and b_tile from HBM into SBUF.
      nisa.dma_copy(dst=a_tile,
                    src=a_input[m * TILE_M:(m + 1) * TILE_M,
                                n * TILE_N:(n + 1) * TILE_N])
      nisa.dma_copy(dst=b_tile,
                    src=b_input[m * TILE_M:(m + 1) * TILE_M,
                                n * TILE_N:(n + 1) * TILE_N])

      # Allocate space for the c_tile in sbuf.
      c_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=a_input.dtype, buffer=nl.sbuf)

      # Perform the addition using the element-wise tensor_tensor instruction.
      nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

      # Copy the result to the output tensor.
      nisa.dma_copy(dst=c_output[m * TILE_M:(m + 1) * TILE_M,
                                 n * TILE_N:(n + 1) * TILE_N],
                    src=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output
  # NKI_EXAMPLE_27_END
