"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

Stable Diffusion Attention NKI kernel implementation.

"""

# NKI_EXAMPLE_32_BEGIN
import torch
from torch_xla.core import xla_model as xm

from sd_attention_nki_kernels import fused_self_attn_for_SD_small_head_size


if __name__ == "__main__":

  device = xm.xla_device()

  def cpu_golden_attn(q, k, v):
      softmax_scale = 0.125
      q_scaled = q * softmax_scale
      raw_score = torch.matmul(q_scaled, k.transpose(1, 0))
      
      norm_score = torch.nn.functional.softmax(raw_score, dim=-1)

      return torch.matmul(norm_score, v)

  q_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
  k_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
  v_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)

  output_nki = fused_self_attn_for_SD_small_head_size(q_tensor, k_tensor, v_tensor)

  output_torch = cpu_golden_attn(q_tensor, k_tensor, v_tensor)

  allclose = torch.allclose(output_torch, output_nki, atol=1e-5, rtol=1e-3)

  if allclose:
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")

  assert allclose
  # NKI_EXAMPLE_32_END
