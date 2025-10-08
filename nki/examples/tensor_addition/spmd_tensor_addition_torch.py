"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for SPMD tensor addition NKI tutorial.

"""
# NKI_EXAMPLE_29_BEGIN
import torch
from torch_xla.core import xla_model as xm
# NKI_EXAMPLE_29_END

from spmd_tensor_addition_nki_kernels import nki_tensor_add


# NKI_EXAMPLE_29_BEGIN
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
  # NKI_EXAMPLE_29_END
