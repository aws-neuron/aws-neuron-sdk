import neuronxcc.nki.language as nl
from neuronxcc import nki


@nki.jit
def matmul_128x128x512_spmd(A, B):
  """NKI kernel to compute a 128x128x512 matrix multiplication operation.
     Use SPMD program IDs to index into the full A and B input tensor to get tiles
     for 128x128x512 matrix multiplication.

  Args:
      A: an input tensor of shape [M=512,K=128],
         a left hand side argument of the matrix multiplication,
      B: an input tensor of shape [K=128,N=1024],
         a right hand side argument of the matrix multiplication
      result: the resulting output tensor of shape [M=512,N=1024]
  """
  N, K = A.shape
  K_, M = B.shape
  assert K == K_
  # Create output tensor shared between all SPMD instances as result tensor
  result = nl.ndarray((N, M), dtype=A.dtype, buffer=nl.shared_hbm)

  # Defining starting indexes for input A and B
  i_A_row = nl.program_id(0) * 128
  i_B_col = nl.program_id(1) * 512

  # Loading the inputs (HBM->SBUF)
  A_tile = nl.load(A[i_A_row:i_A_row+128, 0:128])
  B_tile = nl.load(B[0:128, i_B_col:i_B_col+512])

  # Perform the matrix-multiplication
  # Note1: nl.matmul will invoke a transpose on A_tile before performing the actual matmul operation
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nl.matmul(A_tile, B_tile)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # The result of a [128,128] x [128,512] matrix multiplication has a shape of [128, 512].
  # This dictates which indices to use to address the result tile.
  nl.store(result[i_A_row:i_A_row+128, i_B_col:i_B_col+512], value=result_sbuf)

  return result

if __name__ == "__main__":
  from torch_xla.core import xla_model as xm
  import torch

  device = xm.xla_device()

  A = torch.ones((512, 128), dtype=torch.bfloat16).to(device=device)
  B = torch.ones((128, 1024), dtype=torch.bfloat16).to(device=device)

  # Launch kernel with a 2D grid
  result = matmul_128x128x512_spmd[4, 2](A, B)

  print(result) # an implicit XLA barrier/mark-step
