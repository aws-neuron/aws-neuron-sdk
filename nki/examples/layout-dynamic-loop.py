import neuronxcc.nki.language as nl
from neuronxcc import nki
import math

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
  Returns:
      out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
  """
  sz_p, sz_f = in_tensor.shape
  out_tensor = nl.ndarray((sz_p, sz_f), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  for p in nl.affine_range(math.ceil(sz_p / nl.tile_size.pmax)):
    # Generate tensor indices for the input/output tensors
    p_start = k * nl.tile_size.pmax
    p_end = p_start + nl.tile_size.pmax
    i_p = slice(p_start, min(p_end, sz_p))

    # Load input data from external memory to on-chip memory
    in_tile = nl.load(in_tensor[i_p, 0:sz_f]

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to external memory
    nl.store(out_tensor[i_p, 0:sz_f], value=out_tile)

    return out_tensor
