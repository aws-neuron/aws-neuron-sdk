.. _api_neuron_driver_shared_tensor_batch_op_h:

neuron_driver_shared_tensor_batch_op.h
=======================================

Shared tensor batch operation structures between runtime and driver.

**Source**: `src/libnrt/include/ndl/neuron_driver_shared_tensor_batch_op.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared_tensor_batch_op.h>`_

Typedefs
--------

nrt_tensor_batch_offset_t
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef uint64_t nrt_tensor_batch_offset_t;

Type for tensor batch operation offset.

**Source**: `neuron_driver_shared_tensor_batch_op.h:13 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared_tensor_batch_op.h#L13>`_

nrt_tensor_batch_size_t
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef uint64_t nrt_tensor_batch_size_t;

Type for tensor batch operation size.

**Source**: `neuron_driver_shared_tensor_batch_op.h:14 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared_tensor_batch_op.h#L14>`_

Structures
----------

nrt_tensor_batch_op_t
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_batch_op {
       nrt_tensor_batch_offset_t offset;
       nrt_tensor_batch_size_t size;
       void *buffer;
   } nrt_tensor_batch_op_t;

Tensor batch operation structure containing offset, size, and buffer pointer.

**Source**: `neuron_driver_shared_tensor_batch_op.h:17 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared_tensor_batch_op.h#L17>`_
