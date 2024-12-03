"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for matrix multiplication NKI tutorial.

"""

import torch
from torch_xla.core import xla_model as xm

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

if __name__ == "__main__":

  # NKI_EXAMPLE_17_BEGIN
  device = xm.xla_device()
  cpu = torch.device('cpu')

  # Test the small workload with basic kernel
  lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
  rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

  # Run NKI kernel
  output_small = nki_matmul_basic_(lhs_small.T, rhs_small)

  # Run torch reference
  output_small_torch = torch.matmul(lhs_small, rhs_small)

  # Compare results
  print("Checking correctness of nki_matmul_basic")
  if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")
    # NKI_EXAMPLE_17_END

  # NKI_EXAMPLE_22_BEGIN
  # Test the large workload with tiled kernels
  lhs = torch.rand((4096, 1024), dtype=torch.bfloat16, device=device)
  rhs = torch.rand((1024, 2048), dtype=torch.bfloat16, device=device)

  # Run torch reference
  output_torch = torch.matmul(lhs, rhs).to(device=cpu)

  def check_match(nki_func):
    output = nki_func(lhs.T, rhs)
    output_nki = output.to(device=cpu)
    if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
      print("NKI and Torch match")
    else:
      print("NKI and Torch differ")

  print("Checking correctness of nki_matmul_tiled")
  check_match(nki_matmul_tiled_)

  print("Checking correctness of nki_matmul_hoist_load")
  check_match(nki_matmul_hoist_load_)

  print("Checking correctness of nki_matmul_block_free_dimension")
  check_match(nki_matmul_block_free_dimension_)

  print("Checking correctness of nki_matmul_fully_optimized")
  check_match(nki_matmul_fully_optimized_)
  # NKI_EXAMPLE_22_END
