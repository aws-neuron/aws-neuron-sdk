.. _error-code-evrf011:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF011.

NCC_EVRF011
===========

**Error message**: The compiler encountered strided convolution combined with dilated input, which is not supported.

Erroneous code example:

.. code-block:: python

    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

    result = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(2, 2),    # strided convolution
        padding=((2, 2), (2, 2)),
        lhs_dilation=(2, 2),      # and dilated input
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )


If possible, remove stride or input dilation:

.. code-block:: python

    x = jnp.ones((1, 4, 4, 1), dtype=jnp.float32)
    kernel = jnp.ones((3, 3, 1, 1), dtype=jnp.float32)

    result = lax.conv_general_dilated(
        x, kernel,
        window_strides=(2, 2),  
        padding=((2, 2), (2, 2)),
        lhs_dilation=(1, 1),    # remove input dilation
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )


Or apply upsampling and downsampling separately.
