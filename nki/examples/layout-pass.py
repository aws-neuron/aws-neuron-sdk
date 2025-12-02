import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def tensor_exp_kernel_(in_tensor):
  """NKI kernel to compute elementwise exponential of an input tensor

  Args:
      in_tensor: an input tensor of shape [128,512]
  Returns:
      out_tensor: an output tensor of shape [128,512]
  """

  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Load input data from HBM to on-chip memory
  in_tile = nl.load(in_tensor[0:128, 0:512)

  # perform the computation:
  out_tile = nl.exp(in_tile)

  # store the results back to HBM
  nl.store(out_tensor[0:128, 0:512], value=out_tile)

  return out_tensor


if __name__ == "__main__":
  import torch
  from torch_xla.core import xla_model as xm

  device = xm.xla_device()

  shape = (128, 512)
  in_tensor = torch.ones(shape,  dtype=torch.bfloat16).to(device=device)
  out_tensor = tensor_exp_kernel_(in_tensor)

  print(out_tensor) # an implicit XLA barrier/mark-step
