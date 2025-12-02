from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

@nki.jit
def tensor_split_kernel_(in_tensor):
  """NKI kernel to split an input tensor into two output tensors, along the column axis.

  The even columns of the input tensor will be gathered into the first output tensor,
  and the odd columns of the input tensor will be gathered into the second output tensor.

  Args:
      in_tensor: an input tensor
  Returns:
      out_tensor_even: a first output tensor (will hold the even columns of the input tensor)
      out_tensor_odd: a second output tensor (will hold the odd columns of the input tensor)
  """

  # This example only works for tensors with a partition dimension that fits in the SBUF
  assert in_tensor.shape[0] <= nl.tile_size.pmax

  # Extract tile sizes.
  sz_p, sz_f = in_tensor.shape
  sz_fout_even = sz_f - sz_f // 2
  sz_fout_odd = sz_f // 2

  # create output tensors
  out_tensor_even = nl.ndarray((sz_p, sz_fout_even), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  out_tensor_odd = nl.ndarray((sz_p, sz_fout_odd), dtype=in_tensor.dtype, buffer=nl.shared_hbm)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Store the results back to external memory
  nl.store(out_tensor_even, value=in_tile[:, 0:sz_f:2])
  nl.store(out_tensor_odd,  value=in_tile[:, 1:sz_f:2])

  return out_tensor_even, out_tensor_odd


if __name__ == "__main__":
    import torch
    from torch_xla.core import xla_model as xm

    device = xm.xla_device()

    X, Y = 4, 5
    in_tensor = torch.arange(X * Y, dtype=torch.bfloat16).reshape(X, Y).to(device=device)

    out1_tensor, out2_tensor = tensor_split_kernel_(in_tensor)
    print(in_tensor, out1_tensor, out2_tensor)
