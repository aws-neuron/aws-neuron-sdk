"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

RMSNorm NKI PyTorch implementation.

"""

from torch_neuronx.xla_impl.ops import nki_jit
import torch
import os
from rmsnorm_nki_kernels import nki_rmsnorm_kernel

# Reference torch implementation
def torch_rmsnorm_kernel(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = a_tensor.pow(2)
  # Calculate means in the free dimension
  mean = in_square.mean(dim=1, keepdim=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * torch.rsqrt(mean)

  # Scale the output by the weight
  return tensor * g_tensor

from torch_xla.core import xla_model as xm
device = xm.xla_device()

nki_rmsnorm_kernel = nki_jit(nki_rmsnorm_kernel)

a_tensor = torch.rand((250, 512), dtype=torch.float32).to(device=device)
g_tensor = torch.rand((512), dtype=torch.float32).to(device=device)
output_nki = torch.zeros((250, 512), dtype=torch.float32).to(device=device)

nki_rmsnorm_kernel(a_tensor, g_tensor, output_nki)
print(f"output_nki={output_nki}")

output_torch = torch_rmsnorm_kernel(a_tensor, g_tensor)
print(f"output_torch={output_torch}")

if torch.allclose(output_torch, output_nki, atol=1e-5, rtol=1e-3):
  print("NKI and Torch match")
else:
  print("NKI and Torch differ")
