.. _error-code-evrf066:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF066.

NCC_EVRF066
===========

**Error message**: NKI kernel '``<kernel-name>``' is configured with
LNC=\ ``<kernel-lnc>``, but neuronx-cc is using LNC=\ ``<compiler-lnc>``.

This error occurs when a NKI kernel and neuronx-cc use different Logical
Neuron Core (LNC) configuration values (degrees). LNC degree is a global compilation property of a model compute-graph, so all NKI kernels embedded in a model must use the same
LNC degree as neuronx-cc.

For more information on how to configure the LNC degree of NKI kernels, see:
:doc:`LNC Overview </nki/get-started/about/lnc>`.

Common causes
-------------

- Not explicitly setting the LNC degree for NKI kernels.

  .. code-block:: python

     kernel(...)  # Kernel implicitly defaults to LNC=1; neuronx-cc defaults to LNC=2 on Trn2 and later.

- Using the launch-grid bracket syntax, such as ``kernel[1](...)``, to set an
  LNC degree that differs from the LNC degree used by neuronx-cc. This syntax
  is being deprecated.

  .. code-block:: python

     kernel[1](...)  # Kernel explicitly uses LNC=1; neuronx-cc defaults to LNC=2 on Trn2 and later.

- Overriding the ``--lnc`` option passed to neuronx-cc, for example
  through ``NEURON_CC_FLAGS``, with an LNC degree that differs from the NKI
  kernel LNC degree.

  .. code-block:: bash

     export NEURON_LOGICAL_NC_CONFIG=2  # Set kernel LNC=2.
     export NEURON_CC_FLAGS="--lnc=1"

  .. code-block:: python

     kernel(...)  # Kernel explicitly uses LNC=2; neuronx-cc is overridden to LNC=1.

Resolution
----------

1. **Explicitly set the** ``NEURON_LOGICAL_NC_CONFIG``
   **environment variable** to the desired LNC degree so that neuronx-cc
   and all NKI kernels use the same LNC degree. ML frameworks generally pass the value
   of this environment variable to neuronx-cc through its ``--lnc`` flag.
2. **Remove any launch-grid bracket syntax** from the NKI kernel call site,
   such as ``kernel[1](...)``.
3. **Remove any conflicting** ``--lnc`` **overrides**, such as one set through
   ``NEURON_CC_FLAGS``.

LNC=1 example
~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   kernel(...)  # Kernel implicitly uses LNC=1; neuronx-cc defaults to LNC=2 on Trn2 and later.

**After:**

.. code-block:: bash

   export NEURON_LOGICAL_NC_CONFIG=1
   kernel(...)  # Kernel and neuronx-cc both use LNC=1.

LNC=2 example
~~~~~~~~~~~~~

**Before:**

.. code-block:: bash

   export NEURON_CC_FLAGS="--lnc=1"
   kernel[2](...)  # Kernel uses LNC=2; neuronx-cc is overridden to LNC=1.

**After:**

.. code-block:: bash

   export NEURON_LOGICAL_NC_CONFIG=2
   kernel(...)  # Kernel and neuronx-cc both use LNC=2.
