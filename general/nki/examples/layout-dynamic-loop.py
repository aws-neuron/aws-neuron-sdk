import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import math

@nki_jit
def tensor_exp_kernel_(in_tensor, out_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  sz_p, sz_f = in_tensor.shape

  i_f = nl.arange(sz_f)[None, :]

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    # pad index to pmax, for simplicity
    i_p = p * nl.tile_size.pmax + nl.arange(nl.tile_size.pmax)[:, None]

    # Load input data from external memory to on-chip memory
    # only read up to sz_p
    in_tile = nl.load(in_tensor[i_p, i_f], mask=(i_p<sz_p))

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to external memory
    # only write up to sz_p
    nl.store(out_tensor[i_p, i_f], value=out_tile, mask=(i_p<sz_p))