import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import math

@nki_jit
def tensor_split_kernel_(in_tensor, out_tensor_even, out_tensor_odd):
  """NKI kernel to split an input tensor into two output tensors, along the column axis.

  The even columns of the input tensor will be gathered into the first output tensor,
  and the odd columns of the input tensor will be gathered into the second output tensor.

  Args:
      in_tensor: an input tensor
      out_tensor_even: a first output tensor (will hold the even columns of the input tensor)
      out_tensor_odd: a second output tensor (will hold the odd columns of the input tensor)
  """

  # Extract tile sizes.
  sz_p, sz_f = in_tensor.shape
  sz_fout_even, sz_fout_odd = out_tensor_even.shape[1], out_tensor_odd.shape[1]

  # We assume that all three tensors have the same partition dimension size
  # and it does not exceed pmax
  assert in_tensor.shape[0] == out_tensor_even.shape[0] == out_tensor_odd.shape[0]
  assert in_tensor.shape[0] <= nl.tile_size.pmax

  # Make sure even/odd output tensors have correct free dimension size
  assert sz_fout_even == math.ceil(sz_f / 2)
  assert sz_fout_odd == math.floor(sz_f / 2)

  # Generate tensor indices for the input/output tensors
  i_p = nl.arange(sz_p)[:, None]
  i_f = nl.arange(sz_f)[None, :]
  i_fout_even = nl.arange(sz_fout_even)[None, :]
  i_fout_odd = nl.arange(sz_fout_odd)[None, :]

  # Split pattern:
  i_f_even = (2 * i_fout_even)
  i_f_odd = (2 * i_fout_odd + 1)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor[i_p, i_f])

  # Perform the split
  # these assignments invoke copy instructions under the hood
  # which can execute on either Scalar or Vector Engine
  # (decided by compiler instruction scheduler)
  out_tile_even = in_tile[i_p, i_f_even]
  out_tile_odd = in_tile[i_p, i_f_odd]

  # Store the results back to external memory
  nl.store(out_tensor_even[i_p, i_fout_even], value=out_tile_even)
  nl.store(out_tensor_odd[i_p, i_fout_odd], value=out_tile_odd)


if __name__ == "__main__":
    import torch
    from torch_xla.core import xla_model as xm

    device = xm.xla_device()

    X, Y = 4, 5
    in_tensor = torch.arange(X * Y, dtype=torch.bfloat16).reshape(X, Y).to(device=device)

    out1_tensor = torch.zeros((X, Y-Y//2), dtype=torch.bfloat16).to(device=device)
    out2_tensor = torch.zeros((X, Y//2), dtype=torch.bfloat16).to(device=device)

    tensor_split_kernel_(in_tensor, out1_tensor, out2_tensor)
    print(in_tensor, out1_tensor, out2_tensor)
