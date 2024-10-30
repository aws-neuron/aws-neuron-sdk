"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI implementation for average pool 2D NKI tutorial.

"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


def tensor_avgpool_kernel_(in_tensor, out_tensor, pool_size):
  """NKI kernel to compute a 2D avg-pool operation

  Args:
      in_tensor: an input tensor, of shape C x H x W
      pool_size: an integer representing a (square) pool-window size
      out_tensor: the resulting output tensor, of shape C x (H/pool_size) x (W/pool_size)
  """

  # Get input/output dimensions
  sz_cin, sz_hin, sz_win = in_tensor.shape
  sz_cout, sz_hout, sz_wout = out_tensor.shape
  assert sz_cin == sz_cout

  # Set relevant sizes
  sz_p = sz_cin
  sz_pool = pool_size

  # Generate tensor h/w index patterns
  # 3D indexing according to [C, H, W]
  i_p = nl.arange(sz_p)[:, None, None] # 3D for
  i_win = nl.arange(sz_win)[None, None, :]
  i_hin = nl.arange(sz_hin)[None, :, None]

  i_wout = nl.arange(sz_wout)[None, None, :]
  i_hout = nl.arange(sz_hout)[None, :, None]

  # Generate pool index patterns (requires two extra dimensions, for the pool window)
  i_0 = nl.arange(sz_p)[:, None, None, None, None] #
  i_1 = nl.arange(sz_hin//sz_pool)[None, :, None, None, None] # y_outer
  i_2 = nl.arange(sz_pool)[None, None, :, None, None] # y_inner
  i_3 = nl.arange(sz_win//sz_pool)[None, None, None, :, None] # x_outer
  i_4 = nl.arange(sz_pool)[None, None, None, None, :] # x_inner

  # Load input data from external memory to on-chip memory
  # Declare ndarray to force a 3D tensor (temporary requirement)
  in_tile = nl.ndarray([sz_p, sz_hin, sz_win], dtype=in_tensor.dtype)
  in_tile[:,:,:] = nl.load(in_tensor[i_p, i_hin, i_win])

  # Perform the pooling operation:
  # We use numpy's advanced indexing, in order to extend in_tile to 5D, and then reduce-average two dimension.
  # axis[0] is the index for p_dim, and thus doesn't participate in the reduction operation.
  # axis[1] and axis[2] together index the rows, with axis[2] responsible for inner strides
  # (i.e. inside a pooling window), and axis[1] responsible for the outer strides. As such, we reduce over axis[2].
  # Similarly, axis[3] and axis[4] together index the columns, and we thus reduce over axis[4].
  out_tile = nl.sum(in_tile[i_0, sz_pool*i_1+i_2, sz_pool*i_3+i_4], axis=[2,4]) / (pool_size*pool_size)

  # Store the results back to external memory
  nl.store(out_tensor[i_p, i_hout, i_wout], value=out_tile)


# Reference NumPy implementation
def np_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return np.nanmean(reshaped, axis=(2, 4))


if __name__ == "__main__":
  # Now let's run the kernel
  POOL_SIZE = 2
  C, HIN, WIN = 2, 6, 6
  HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

  in_tensor = np.arange(C * HIN * WIN, dtype=np.float16).reshape(C, HIN, WIN)
  out_nki = np.zeros((C, HOUT, WOUT), dtype=np.float16)

  tensor_avgpool_kernel_baremetal = nki.baremetal(tensor_avgpool_kernel_)
  tensor_avgpool_kernel_baremetal(in_tensor, out_nki, POOL_SIZE)

  out_np = np_average_pool_2D(in_tensor, POOL_SIZE)

  print(in_tensor, out_nki, out_np)

  match = (out_nki == out_nki).all()

  if match:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert match
