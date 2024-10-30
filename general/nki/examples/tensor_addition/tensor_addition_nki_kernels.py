"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

NKI implementation for tensor addition NKI tutorial.

"""
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


def nki_tensor_add_kernel_(a_input, b_input, c_output):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output tile-sizes, of up-to [128,512]

  Args:
      a_input: a first input tensor, of shape [128,512]
      b_input: a second input tensor, of shape [128,512]
      c_output: an output tensor, of shape [128,512]
  """

  # Calculate tile offsets based on current 'program'
  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 512

  # Generate tensor indices to index tensors a and b
  ix = offset_i_x + nl.arange(128)[:, None]
  iy = offset_i_y + nl.arange(512)[None, :]

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  # We refer to an indexed portion of a tensor as an intermediate tensor
  a_tile = nl.load(a_input[ix, iy])
  b_tile = nl.load(b_input[ix, iy])

  # compute a + b
  c_tile = a_tile + b_tile

  # store the addition results back to device memory (c_output)
  nl.store(c_output[ix, iy], value=c_tile)


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
  c_output = np.zeros(a_input.shape, dtype=a_input.dtype)

  nki_tensor_add_kernel_baremetal = nki.baremetal(nki_tensor_add_kernel_)
  nki_tensor_add_kernel_baremetal[grid_x, grid_y](a_input, b_input, c_output)

  return c_output


if __name__ == "__main__":
  a = np.random.rand(256, 1024).astype(np.float16)
  b = np.random.rand(256, 1024).astype(np.float16)

  output_nki = nki_tensor_add(a, b)
  print(f"output_nki={output_nki}")

  output_np = a + b
  print(f"output_np={output_np}")

  allclose = np.allclose(output_np, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and NumPy match")
  else:
    print("NKI and NumPy differ")

  assert allclose
