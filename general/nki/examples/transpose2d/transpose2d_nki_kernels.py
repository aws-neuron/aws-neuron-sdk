"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI baremetal implementation for transpose2d NKI tutorial.
"""

import numpy as np
# NKI_EXAMPLE_33_BEGIN
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
  """
  NKI kernel to reorder the elements on axis[1] of the input tensor.

  Every row of the input tensor is a flattened row-major 2D matrix.
  The shape2D argument defines the dimensions of the flattened matrices (#rows,#cols).
  Our goal in this kernel is to transpose these flattened 2D matrices, i.e. make them (#cols,#rows).

  Example:
      in_tensor = [a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3]
      shape2D = (3,4)
  this means that in_tensor has 3 rows and 4 columns, i.e. can be represented as:
      [a0,a1,a2,a3]
      [b0,b1,b2,b3]
      [c0,c1,c2,c3]
  after transpose, we expect to get:
      [a0,b0,c0]
      [a1,b1,c1]
      [a2,b2,c2]
      [a3,b3,c3]
  Thus, out_tensor is expected to be [a0,b0,c0,a1,b1,c1,a2,b2,c2,a3,b3,c3]

  Args:
    in_tensor: an input tensor
    shape2D: tuple representing the dimensions to be transposed: (#rows, #cols)
    out_tensor: an output (transposed) tensor
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  # Gather input shapes
  sz_p, _ = in_tensor.shape

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Performing f1/f2 transpose
  # ==========================
  # The desired transpose pattern is provided as an input:
  sz_f1, sz_f2 = shape2D

  # We're going to need 3 indices to perform f1:f2 transpose.
  # - i_p0 is the parallel index
  # - i_f1 and i_f2 are both free-dim indices, and will be used to transpose between the f1/f2 axes
  i_p0 = nl.arange(sz_p)[:, None, None]
  i_f1 = nl.arange(sz_f1)[None, :, None]
  i_f2 = nl.arange(sz_f2)[None, None, :]

  # Perform the transposition via a SBUF-to-SBUF copy, with access-pattern manipulation
  # Note that we have 2D tensors and 3 indices, since we need to represent a 2D access pattern *per partition*
  # RHS traverses an F1 x F2 matrix in a row major manner
  # LHS traverses an F2 x F1 (new) matrix in a row major manner
  out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=out_tensor.dtype)
  out_tile[i_p0, i_f2*sz_f1+i_f1] = nl.copy(in_tile[i_p0, i_f1*sz_f2+i_f2])

  # Finally, we store out_tile to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor
  # NKI_EXAMPLE_33_END


if __name__ == "__main__":
  P, X, Y = 5, 3, 4
  a = np.arange(P*X*Y, dtype=np.int8).reshape((P, X*Y))

  a_t_nki = tensor_transpose2D_kernel_(a, (X, Y))

  a_t_np = np.transpose(a.reshape(P, X, Y), (0, 2, 1)).reshape(P, X * Y)

  print(a, a_t_nki, a_t_np)

  allclose = np.allclose(a_t_np, a_t_nki)
  if allclose:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert allclose