"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

JAX implementation for average pool 2D NKI tutorial.

"""
# NKI_EXAMPLE_40_BEGIN
import jax.numpy as jnp
# NKI_EXAMPLE_40_END
from average_pool2d_nki_kernels import tensor_avgpool_kernel


# NKI_EXAMPLE_40_BEGIN
# Reference JAX implementation
def jax_average_pool_2D(in_tensor, pool_size):
  c, h_in, w_in = in_tensor.shape
  reshaped = in_tensor.reshape(c, h_in // pool_size, pool_size, w_in // pool_size, pool_size)
  return jnp.nanmean(reshaped, axis=(2, 4))
  # NKI_EXAMPLE_40_END


# NKI_EXAMPLE_41_BEGIN
if __name__ == "__main__":
  POOL_SIZE = 2
  C, HIN, WIN = 2, 6, 6
  HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

  in_array = jnp.arange(C * HIN * WIN, dtype=jnp.float32).reshape(C, HIN, WIN)

  # NKI_EXAMPLE_39_BEGIN
  out_nki = tensor_avgpool_kernel(in_array, pool_size=POOL_SIZE)
  # NKI_EXAMPLE_39_END
  out_jax = jax_average_pool_2D(in_array, pool_size=POOL_SIZE)

  print(in_array, out_nki, out_jax)

  if jnp.allclose(out_nki, out_jax):
    print("NKI and JAX match")
  else:
    print("NKI and JAX differ")
    # NKI_EXAMPLE_41_END
