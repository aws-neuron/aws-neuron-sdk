"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for transpose2d NKI tutorial.
"""

import torch
from torch_xla.core import xla_model as xm
from torch_neuronx import nki_jit

from transpose2d_nki_kernels import tensor_transpose2D_kernel_


if __name__ == "__main__":
  device = xm.xla_device()

  P, X, Y = 5, 3, 4
  a = torch.arange(P*X*Y, dtype=torch.int8).reshape((P, X*Y)).to(device=device)
  a_t_nki = torch.zeros((P, Y*X), dtype=torch.int8).to(device=device)

  tensor_transpose2D_kernel_torch = nki_jit(tensor_transpose2D_kernel_)
  tensor_transpose2D_kernel_torch(a, a_t_nki, (X, Y))

  a_t_torch = torch.transpose(a.reshape(P, X, Y), 1, 2).reshape(P, X * Y)

  print(a, a_t_nki, a_t_torch)

  allclose = torch.allclose(a_t_torch, a_t_nki)
  if allclose:
    print("NKI and PyTorch match")
  else:
    print("NKI and PyTorch differ")

  assert allclose
