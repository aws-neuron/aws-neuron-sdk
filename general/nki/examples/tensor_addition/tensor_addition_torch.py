"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for tensor addition NKI tutorial.

"""
import torch
from torch_xla.core import xla_model as xm
from torch_neuronx import nki_jit

from tensor_addition_nki_kernels import nki_tensor_add_kernel_


def nki_tensor_add(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  grid_x = a_input.shape[0] // 128
  grid_y = a_input.shape[1] // 512
  c_output = torch.zeros(a_input.shape, dtype=a_input.dtype).to(device=device)

  # Decorate the NKI kernel for PyTorch tracing
  nki_tensor_add_kernel_torch = nki_jit(nki_tensor_add_kernel_)
  nki_tensor_add_kernel_torch[grid_x, grid_y](a_input, b_input, c_output)

  return c_output

if __name__ == "__main__":
  device = xm.xla_device()

  a = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)
  b = torch.rand((256, 1024), dtype=torch.bfloat16).to(device=device)

  output_nki = nki_tensor_add(a, b)
  print(f"output_nki={output_nki}")

  output_torch = a + b
  print(f"output_torch={output_torch}")

  allclose = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")

  assert allclose
