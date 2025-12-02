.. _error-code-esfh002:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error ESFH002.

NCC_ESFH002
===========

**Error message**: The compiler encountered a unsigned 64-bit integer constant with a value that cannot be safely converted to 32-bit representation. 

The Neuron hardware operates on 32-bit or narrower data types and attempts to convert 64-bit integers to 32-bit. 64-bit constants that exceed the 32-bit range and cannot be safely converted will fail compilation. Try to use uint32 for constants when possible and restructure code to avoid large constants.

Erroneous code example:

.. code-block:: python

   @jax.jit
   def foo():
      # direct uint64 constant in arithmetic operation
      x = jnp.array([1, 2, 3], dtype=jnp.uint64)
      # large constant that exceeds uint32 max
      large_constant = jnp.uint64(5_000_000_000)
      return x + large_constant

Use uint32 for constants when possible:

.. code-block:: python

   @jax.jit
   def test():
      x = jnp.array([1, 2, 3], dtype=jnp.uint32)
      large_constant = jnp.uint32(5_000_000_000)
      return x + large_constant