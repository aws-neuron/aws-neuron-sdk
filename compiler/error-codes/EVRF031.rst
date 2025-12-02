.. _error-code-evrf031:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF031.

NCC_EVRF031
===========

**Error message**: The compiler encountered a scatter out-of-bounds error. The indices created via iota instruction contain values that are beyond the size of the operand dimension.

Erroneous code example:

.. code-block:: python

    # size 3 in dimension 0
    operand = jnp.zeros((3, 4), dtype=jnp.float32)

    # iota generates indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices = lax.iota(jnp.int32, 10) # ERROR: size 10 > operand dimension 3
    indices = indices.reshape(10, 1)

    updates = jnp.ones((10, 4), dtype=jnp.float32) # ERROR: 10 updates but operand only has 3 rows

    result = lax.scatter(
        operand,
        indices, # ERROR: index values in [0, 10) but operand dimension only allows indices in [0, 3)
        updates,
        lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
        )
    )


Ensure that the iota size matches the operand dimension size:

.. code-block:: python

    N = 3
    D = 4
    operand = jnp.zeros((N, D), dtype=jnp.float32)

    # FIXED: match iota size to operand dimension
    indices = lax.iota(jnp.int32, N) # size N is same as operand dimension
    indices = indices.reshape(N, 1)

    # FIXED: updates size matches operand dimension
    updates = jnp.ones((N, D), dtype=jnp.float32)

    result = lax.scatter(
        operand,
        indices, # FIXED: indices now in valid range [0, 3)
        updates,
        lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
        )
    )
