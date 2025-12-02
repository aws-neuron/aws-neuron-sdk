import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit

@nki_jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of shape [256,512]
  Returns:
      out_tensor: an output tensor of shape [256,512]
  """
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  for k in nl.affine_range(2):
    # Generate tensor indices for the input/output tensors
    p_start = k * nl.tile_size.pmax
    p_end = p_start + nl.tile_size.pmax
    i_p = slice(p_start, p_end)

    # Load input data from HBM to on-chip memory
    in_tile = nl.load(in_tensor[i_p, 0:512])

    # perform the computation
    out_tile = nl.exp(in_tile)

    # store the results back to HBM
    nl.store(out_tensor[i_p, i_f], value=out_tile)

  return out_tensor
