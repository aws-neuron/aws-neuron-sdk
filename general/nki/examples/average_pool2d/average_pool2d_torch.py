"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

PyTorch implementation for average pool 2D NKI tutorial.

"""
import torch
from torch_neuronx import nki_jit
from torch_xla.core import xla_model as xm

from average_pool2d_nki_kernels import tensor_avgpool_kernel_


if __name__ == "__main__":
  device = xm.xla_device()

  # Now let's run the kernel
  POOL_SIZE = 2
  C, HIN, WIN = 2, 6, 6
  HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

  in_tensor = torch.arange(C * HIN * WIN, dtype=torch.bfloat16).reshape(C, HIN, WIN).to(device=device)
  out_nki = torch.zeros((C, HOUT, WOUT), dtype=torch.bfloat16).to(device=device)

  tensor_avgpool_kernel_torch = nki_jit(tensor_avgpool_kernel_)
  tensor_avgpool_kernel_torch(in_tensor, out_nki, POOL_SIZE)

  out_torch = torch.nn.functional.avg_pool2d(in_tensor, POOL_SIZE, POOL_SIZE)

  print(in_tensor, out_nki, out_torch) # an implicit XLA barrier/mark-step

  if (out_nki == out_torch).all():
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")
