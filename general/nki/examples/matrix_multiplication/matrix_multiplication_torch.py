"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for matrix multiplication NKI tutorial.

"""

import torch
from torch_xla.core import xla_model as xm
from torch_neuronx import nki_jit

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

if __name__ == "__main__":

  device = xm.xla_device()
  cpu = torch.device('cpu')

  # Test the small workload with basic kernel
  lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
  rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)
  output_small = torch.zeros((64, 512), dtype=torch.bfloat16, device=device)

  # Run NKI kernel
  nki_matmul_basic_jit = nki_jit(nki_matmul_basic_)
  nki_matmul_basic_jit(lhs_small.T, rhs_small, output_small)

  # Run torch reference
  output_small_torch = torch.matmul(lhs_small, rhs_small)

  # Compare results
  print("Checking correctness of nki_matmul_basic")
  if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")

  # Test the large workload with tiled kernels
  lhs = torch.rand((4096, 1024), dtype=torch.bfloat16, device=device)
  rhs = torch.rand((1024, 2048), dtype=torch.bfloat16, device=device)
  output = torch.zeros((4096, 2048), dtype=torch.bfloat16, device=device)

  # Run torch reference
  output_torch = torch.matmul(lhs, rhs).to(device=cpu)

  def check_match(nki_func):
    jit_func = nki_jit(nki_func)
    jit_func(lhs.T, rhs, output)
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
