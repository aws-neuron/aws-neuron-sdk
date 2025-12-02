.. _error-code-evrf017:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF017.

NCC_EVRF017
===========

**Error message**: The compiler encountered a reduce-window operation with base dilation (input dilation) greater than 1, which is not supported.

Erroneous code example:

.. code-block:: python

    result = lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, 1, 1),
        window_strides=(1, 1, 1, 1),
        padding='VALID',
        base_dilation=(1, 2, 1, 1) # ERROR: applying base dilation of 2 in dimension 1
    )

If possible, change base dilation to be all 1s:

.. code-block:: python

    result = lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, 1, 1, 1),
        window_strides=(1, 1, 1, 1),
        padding='VALID',
        base_dilation=(1, 1, 1, 1) # FIXED: all values are 1 (no dilation)
    )

Or consider manual dilation if necessary.
