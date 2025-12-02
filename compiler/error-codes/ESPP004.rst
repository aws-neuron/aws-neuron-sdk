.. _error-code-espp004:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error ESPP004.

NCC_ESPP004
===========

**Error message**: The compiler encountered a data type that is not supported for code generation.

Erroneous code example:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    import jax
    from jax._src import dtypes
    from jax._src.lax import lax as lax_internal

    # float4_e2m1fn type not supported
    dtype = np.dtype(dtypes.float4_e2m1fn)
    val = lax_internal._convert_element_type(0, dtype, weak_type=False)

Use a supported data type:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    import jax
    from jax._src import dtypes
    from jax._src.lax import lax as lax_internal

    # float4_e2m1fn type not supported
    dtype = jnp.bfloat16
    val = lax_internal._convert_element_type(0, dtype, weak_type=False)

More information on supported data types https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
