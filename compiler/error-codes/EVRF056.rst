.. _error-code-evrf056:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF056.

NCC_EVRF056
===========

**Error message**: Operation gather encountered out of bound indices. Operation iota produces index values in range [0, N), while the operand dimension only allows indices in [0, M). This may indicate misconfigured model parameters (e.g., max_position_embeddings < sequence_length).

Erroneous code example:

.. code-block:: python

    # size 3 in dimension 0
    operand = jnp.zeros((3, 4), dtype=jnp.float32)

    # iota generates indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    indices = lax.iota(jnp.int32, 10)  # ERROR: size 10 > operand dimension 3
    indices = indices.reshape(10, 1)

    result = lax.gather(
        operand,
        indices,  # ERROR: index values in [0, 10) but operand dimension only allows indices in [0, 3)
        lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        slice_sizes=(1, 4)
    )

Ensure that the iota dimension size is less than or equal to the size of the corresponding operand dimension. Check that your model's ``max_position_embeddings`` is >= ``sequence_length``:

.. code-block:: python

    N = 3
    D = 4
    operand = jnp.zeros((N, D), dtype=jnp.float32)

    # FIXED: match iota size to operand dimension
    indices = lax.iota(jnp.int32, N)  # size N is same as operand dimension
    indices = indices.reshape(N, 1)

    result = lax.gather(
        operand,
        indices,  # FIXED: indices now in valid range [0, 3)
        lax.GatherDimensionNumbers(
            offset_dims=(1,),
            collapsed_slice_dims=(0,),
            start_index_map=(0,)
        ),
        slice_sizes=(1, D)
    )
