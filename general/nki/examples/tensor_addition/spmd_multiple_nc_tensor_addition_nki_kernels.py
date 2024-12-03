"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI implementation for SPMD tensor addition with multiple Neuron cores NKI tutorial.

"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from spmd_tensor_addition_nki_kernels import nki_tensor_add_kernel_


# NKI_EXAMPLE_48_BEGIN
def nki_tensor_add_nc2(a_input, b_input):
  """NKI kernel caller to compute element-wise addition of two input tensors using multiple Neuron cores.

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs.
  a_input and b_input are sharded across Neuron cores, directly utilizing Trn2 architecture capabilities

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
  # In this case, we use a 2D grid where the size of each invocation is 128x512
  # Since we're sharding across neuron cores on the 1st dimension we want to do our slicing at 
  # 128 per core * 2 cores = 256
  grid_x = a_input.shape[0] // (128 * 2)
  grid_y = a_input.shape[1] // 512

  # In addition, we distribute the kernel to physical neuron cores around the first dimension
  # of the spmd grid.
  # This means:
  # Physical NC [0]: kernel[n, m] where n is even
  # Physical NC [1]: kernel[n, m] where n is odd
  # notice, by specifying this information in the SPMD grid, we can use multiple neuron cores
  # without updating the original `nki_tensor_add_kernel_` kernel.
  return nki_tensor_add_kernel_[nl.spmd_dim(grid_x, nl.nc(2)), grid_y](a_input, b_input)
  # NKI_EXAMPLE_48_END

if __name__ == "__main__":
  a = np.random.rand(512, 2048).astype(np.float16)
  b = np.random.rand(512, 2048).astype(np.float16)

  output_nki = nki_tensor_add_nc2(a, b)
  print(f"output_nki={output_nki}")

  output_np = a + b
  print(f"output_np={output_np}")

  allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert allclose
