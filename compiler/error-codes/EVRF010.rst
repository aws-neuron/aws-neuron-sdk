.. _error-code-evrf010:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF010.

NCC_EVRF010
===========

**Error message**: The compiler encountered simultaneous use of input and kernel dilation, which is not supported.

Erroneous code example:

.. code-block:: python

    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

    result = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=((2, 2), (2, 2)),
        lhs_dilation=(2, 2), # input dilation
        rhs_dilation=(2, 2), # kernel dilation
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )


If possible, use only only input or kernel dilation:

.. code-block:: python

    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

    result = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=((2, 2), (2, 2)),
        lhs_dilation=(1, 1), # no input dilation
        rhs_dilation=(2, 2),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )

Or apply dilation manually and apply convolution to the remainder.
