"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

JAX implementation for transpose2d NKI tutorial.

"""

# NKI_EXAMPLE_36_BEGIN
import jax
import jax.numpy as jnp
# NKI_EXAMPLE_36_END

from transpose2d_nki_kernels import tensor_transpose2D_kernel_

# NKI_EXAMPLE_36_BEGIN
if __name__ == "__main__":
  P, X, Y = 5, 37, 44
  a = jax.random.uniform(jax.random.PRNGKey(42), (P, X * Y))
  a_t_nki = tensor_transpose2D_kernel_(a, shape2D=(X, Y))

  a_t_jax = jnp.transpose(a.reshape(P, X, Y), axes=(0, 2, 1)).reshape(P, X * Y)
  print(a, a_t_nki, a_t_jax)

  allclose = jnp.allclose(a_t_jax, a_t_nki)
  if allclose:
    print("NKI and JAX match")
  else:
    print("NKI and JAX differ")

  assert allclose
# NKI_EXAMPLE_36_END
