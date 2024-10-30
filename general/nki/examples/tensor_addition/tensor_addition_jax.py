"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

JAX implementation for tensor addition NKI tutorial.

"""
import jax
import jax.numpy as jnp
from jax_neuronx import nki_call

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

  out_shape = jax.ShapeDtypeStruct((a_input.shape[0], a_input.shape[1]), dtype=a_input.dtype)

  return nki_call(
      nki_tensor_add_kernel_,
      a_input,
      b_input,
      grid=(grid_x, grid_y),
      out_shape=out_shape,
    )


if __name__ == "__main__":

  seed_a, seed_b = jax.random.split(jax.random.PRNGKey(42))
  a = jax.random.uniform(seed_a, (256, 1024), dtype=jnp.bfloat16)
  b = jax.random.uniform(seed_b, (256, 1024), dtype=jnp.bfloat16)

  output_nki = nki_tensor_add(a, b)
  print(f"output_nki={output_nki}")

  output_jax = a + b
  print(f"output_jax={output_jax}")

  allclose = jnp.allclose(output_jax, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and JAX match")
  else:
    print("NKI and JAX differ")

  assert allclose
