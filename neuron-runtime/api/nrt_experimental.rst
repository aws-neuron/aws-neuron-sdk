.. _api_nrt_experimental_h:

nrt_experimental.h
==================

Neuron Runtime Experimental API - Features under development and subject to change.

**Source**: `src/libnrt/include/nrt/nrt_experimental.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h>`_

.. note::

   Experimental APIs are provided for testing and feedback and may not be appropriate for production environments.

Enumerations
------------

nrt_tensor_usage_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum nrt_tensor_usage {
       NRT_TENSOR_USAGE_INPUT = 0,
       NRT_TENSOR_USAGE_OUTPUT,
   } nrt_tensor_usage_t;

Usage of a Tensor in the NEFF.

**Source**: `nrt_experimental.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L18>`_

Structures
----------

nrt_tensor_info_t
^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_info {
       char name[NRT_TENSOR_NAME_MAX];
       nrt_tensor_usage_t usage;
       size_t size;
       nrt_dtype_t dtype;
       uint32_t *shape;
       uint32_t ndim;
   } nrt_tensor_info_t;

Tensor information including name, usage, size, data type, and shape.

**Source**: `nrt_experimental.h:25 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L25>`_

nrt_tensor_info_array_t
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_info_array {
       uint64_t tensor_count;
       nrt_tensor_info_t tensor_array[];
   } nrt_tensor_info_array_t;

Array of tensor information.

**Source**: `nrt_experimental.h:34 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L34>`_

nrt_model_info_t
^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_model_info {
       uint32_t vnc;
   } nrt_model_info_t;

Model information structure.

**Source**: `nrt_experimental.h:139 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L139>`_

Functions
---------

nrt_get_model_tensor_info
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_get_model_tensor_info(nrt_model_t *model, nrt_tensor_info_array_t **tensor_info);

Return input/output tensor information for a given model.

**Parameters:**

* ``model`` [in] - Model for which tensor information needs to be extracted.
* ``tensor_info`` [out] - Pointer to store the result.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt_experimental.h:48 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L48>`_

nrt_trace_start
^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_trace_start(bool trace_mem);

Enable tracing for all VNCs visible to the app.

**Parameters:**

* ``trace_mem`` [in] - collect memory allocation info

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_experimental.h:68 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L68>`_

nrt_trace_stop
^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_trace_stop(const char *filename);

Serialize all data and disable tracing.

**Parameters:**

* ``filename`` [in] - filename to write to

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_experimental.h:75 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L75>`_

nrt_barrier
^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_barrier(int32_t vnc, uint32_t g_device_id, uint32_t g_device_count);

Implements a barrier by running a small all-reduce over all workers.

**Parameters:**

* ``vnc`` [in] - local VNC (within the instance)
* ``global_device_id`` [in] - global worker ID
* ``global_device_count`` [in] - total number of workers

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt_experimental.h:115 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_experimental.h#L115>`_
