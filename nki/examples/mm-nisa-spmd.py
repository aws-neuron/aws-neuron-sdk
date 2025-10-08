import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc import nki

@nki.jit
def matmul_128x128x512_spmd(A_T, B):
  """NKI kernel to compute a 128x128x512 matrix multiplication operation
     Use SPMD program IDs to index into the full A and B input tensor to get tiles
     for 128x128x512 matrix multiplication.

  Args:
      A_T: an input tensor of shape [K=128,M=512],
         a left hand side argument of the matrix multiplication,
      B: an input tensor of shape [K=128,N=1024],
         a right hand side argument of the matrix multiplication
      result: the resulting output tensor of shape [M=512,N=1024]
  """
  K, N = A_T.shape
  K_, M = B.shape
  assert K == K_
  # Create output tensor shared between all SPMD instances as result tensor
  result = nl.ndarray((N, M), dtype=A_T.dtype, buffer=nl.shared_hbm)

  # Defining starting indexes for input A.T and B
  i_A_T_col = nl.program_id(0) * 128
  i_B_col = nl.program_id(1) * 512

  # Loading the inputs (HBM->SBUF)
  A_T_tile = nl.load(A_T[0:128, i_A_T_col:i_A_T_col+128])
  B_tile = nl.load(B[0:128, i_B_col:i_B_col+512])

  # Perform the matrix-multiplication
  # Note1: p-dim of both input tiles is mapped to the contraction dimension, aligned
  # with TensorE layout requirements (LayoutConstraint #1: For MatMult, contraction
  # axis must be mapped to P-dim)
  # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
  result_psum = nisa.nc_matmul(A_T_tile, B_tile)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  # Store back into result tile with the correct SPMD offsets.
  nl.store(result[i_A_T_col:i_A_T_col+128, i_B_col:i_B_col+512], value=result_sbuf)

  return result


if __name__ == "__main__":
  from torch_xla.core import xla_model as xm
  import torch

  device = xm.xla_device()

  # Pre-transpose A for matmul on TensorE
  A_T = torch.ones((128, 512), dtype=torch.bfloat16).to(device=device)
  B = torch.ones((128, 1024), dtype=torch.bfloat16).to(device=device)

  # Launch kernel with a 2D grid
  result = matmul_128x128x512_spmd[4, 2](A_T, B)

  print(result) # an implicit XLA barrier/mark-step
