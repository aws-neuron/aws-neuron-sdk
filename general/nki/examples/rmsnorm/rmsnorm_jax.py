"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

RMSNorm NKI JAX implementation.

"""

import jax
import jax.numpy as jnp
from jax_neuronx import nki_call
from rmsnorm_nki_kernels import nki_rmsnorm_kernel

# Reference JAX implementation
def jax_rms_norm(a_tensor, g_tensor):
  # Square the tensor (element-wise)
  in_square = jnp.square(a_tensor)
  # Calculate means in the free dimension
  mean = in_square.mean(axis=1, keepdims=True)
  # Scale by reciprocal of sqrt(mean)
  tensor = a_tensor * jnp.reciprocal(jnp.sqrt(mean))

  # Scale the output by the weight
  return tensor * g_tensor

a_key, g_key = jax.random.split(jax.random.PRNGKey(42))
a_tensor = jax.random.uniform(a_key, (250, 512))
g_tensor = jax.random.uniform(g_key, (512,))

output_nki = nki_call(
  nki_rmsnorm_kernel,
  a_tensor, g_tensor,
  out_shape=jax.ShapeDtypeStruct(a_tensor.shape, dtype=a_tensor.dtype),
)

print(a_tensor)

print(f"output_nki={output_nki}")

output_jax = jax_rms_norm(a_tensor, g_tensor)
print(f"output_jax={output_jax}")

if jnp.allclose(output_jax, output_nki, atol=1e-5, rtol=1e-3):
  print("NKI and JAX match")
else:
  print("NKI and JAX differ")
