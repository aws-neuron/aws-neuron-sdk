"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

JAX implementation for SPMD tensor addition with multiple Neuron cores NKI tutorial.

"""
# NKI_EXAMPLE_50_BEGIN
import jax
import jax.numpy as jnp
# NKI_EXAMPLE_50_END

from spmd_multiple_nc_tensor_addition_nki_kernels import nki_tensor_add_nc2

# NKI_EXAMPLE_50_BEGIN
if __name__ == "__main__":

  seed_a, seed_b = jax.random.split(jax.random.PRNGKey(42))
  a = jax.random.uniform(seed_a, (512, 2048), dtype=jnp.bfloat16)
  b = jax.random.uniform(seed_b, (512, 2048), dtype=jnp.bfloat16)

  output_nki = nki_tensor_add_nc2(a, b)
  print(f"output_nki={output_nki}")

  output_jax = a + b
  print(f"output_jax={output_jax}")

  allclose = jnp.allclose(output_jax, output_nki, atol=1e-4, rtol=1e-2)
  if allclose:
    print("NKI and JAX match")
  else:
    print("NKI and JAX differ")

  assert allclose
  # NKI_EXAMPLE_50_END
