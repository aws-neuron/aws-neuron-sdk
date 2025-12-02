.. _error-code-evrf018:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF018.

NCC_EVRF018
===========

**Error message**: The compiler encountered a reduce-window operation with window dilation greater than 1, which is not supported.

Erroneous code example:

.. code-block:: python

    result = lax.reduce_window(
        jnp.ones((1, 4, 4, 1)), -jnp.inf, lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 1, 1, 1),
        padding='VALID',
        window_dilation=(1, 2, 2, 1) # 2 is greater than 1
    )


If possible, remove window_dilation or change values to be all 1s:

.. code-block:: python

    result = lax.reduce_window(
        jnp.ones((1, 4, 4, 1)), -jnp.inf, lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 1, 1, 1),
        padding='VALID',
        window_dilation=(1, 1, 1, 1) 
    )

Or consider manual dilation if necessary.
