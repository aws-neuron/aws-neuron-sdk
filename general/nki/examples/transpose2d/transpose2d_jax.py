"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

JAX implementation for transpose2d NKI tutorial.

"""

import jax
import jax.numpy as jnp
from functools import partial
from jax_neuronx import nki_call

from transpose2d_nki_kernels import tensor_transpose2D_kernel_


def transpose2D(in_tensor, shape2D):
  return nki_call(
    partial(tensor_transpose2D_kernel_, shape2D=shape2D),
    in_tensor,
    out_shape=jax.ShapeDtypeStruct(in_tensor.shape, dtype=in_tensor.dtype)
  )

if __name__ == "__main__":
  P, X, Y = 5, 37, 44
  a = jax.random.uniform(jax.random.PRNGKey(42), (P, X * Y))
  a_t_nki = transpose2D(a, (X, Y))

  a_t_jax = jnp.transpose(a.reshape(P, X, Y), axes=(0, 2, 1)).reshape(P, X * Y)
  print(a, a_t_nki, a_t_jax)

  allclose = jnp.allclose(a_t_jax, a_t_nki)
  if allclose:
    print("NKI and JAX match")
  else:
    print("NKI and JAX differ")

  assert allclose
