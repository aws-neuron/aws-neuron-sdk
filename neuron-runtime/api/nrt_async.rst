.. _api_nrt_async_h:

nrt_async.h
===========

Neuron Runtime Asynchronous Execution API - Non-blocking operations for tensor I/O and model execution.

**Source**: `src/libnrt/include/nrt/nrt_async.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h>`_

.. note::

   The Neuron Runtime Async APIs are currently in early release and may change across Neuron versions.

Enumerations
------------

nrta_xu_t
^^^^^^^^^

.. code-block:: c

   typedef enum {
       NRTA_XU_TENSOR_READ = 0,
       NRTA_XU_TENSOR_WRITE,
       NRTA_XU_TENSOR_OP,
       NRTA_XU_COMPUTE,
       NRTA_XU_COLLECTIVES,
       NRTA_XU_TYPE_NUM
   } nrta_xu_t;

Execution unit types.

**Source**: `nrt_async.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L18>`_

Typedefs
--------

nrta_seq_t
^^^^^^^^^^

.. code-block:: c

   typedef uint64_t nrta_seq_t;

Monotonically increasing IDs of executions. The first 16 bits are an Execution Unit ID, while the last 48 bits are a strictly ordered Sequence Number.

**Source**: `nrt_async.h:31 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L31>`_

nrta_xu_id_t
^^^^^^^^^^^^

.. code-block:: c

   typedef uint16_t nrta_xu_id_t;

Execution unit ID type.

**Source**: `nrt_async.h:32 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L32>`_

Constants
---------

NRTA_SEQ_NUM_MAX
^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NRTA_SEQ_NUM_MAX ((1ull << 48) - 1)

Maximum sequence number value.

**Source**: `nrt_async.h:34 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L34>`_

Structures
----------

nrta_error_t
^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrta_error {
       nrta_seq_t seq_id;
       uint64_t error_code;
   } nrta_error_t;

Error information for asynchronous operations.

**Source**: `nrt_async.h:40 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L40>`_

Functions
---------

nrta_tensor_write
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_write(nrt_tensor_t *tensor, const void *buf, uint64_t offset, 
                                uint64_t size, int queue, nrta_error_tracker_t *err, 
                                nrta_seq_t *req_sequence);

Enqueues a tensor write request. Copies the data from a host buffer to a tensor allocated on a Neuron device.

**Parameters:**

* ``tensor`` [in] - Destination tensor
* ``buf`` [in] - Host buffer containing source data
* ``offset`` [in] - Offset into the tensor
* ``size`` [in] - Number of bytes to write
* ``queue`` [in] - XU queue to use
* ``err`` [in] - error tracker
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:59 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L59>`_

nrta_tensor_read
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_read(void *buf, nrt_tensor_t *tensor, uint64_t offset, 
                               uint64_t size, int queue, nrta_error_tracker_t *err, 
                               nrta_seq_t *req_sequence);

Enqueues a tensor read request. Copies the data from a tensor allocated on a Neuron device to a host buffer.

**Parameters:**

* ``buf`` [in] - Destination Host buffer
* ``tensor`` [in] - Source tensor
* ``offset`` [in] - Offset into the tensor
* ``size`` [in] - Number of bytes to read
* ``queue`` [in] - XU queue to use
* ``err`` [in] - error tracker
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:77 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L77>`_

nrta_tensor_copy
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_copy(nrt_tensor_t *src, uint64_t src_offset, nrt_tensor_t *dst, 
                               uint64_t dst_offset, uint64_t size, int queue, 
                               nrta_error_tracker_t *err, nrta_seq_t *req_sequence);

Enqueues a tensor copy request. Copies data between two tensors allocated on the same Logical Neuron Core.

**Parameters:**

* ``src`` [in] - Source tensor
* ``src_offset`` [in] - Offset into the source tensor
* ``dst`` [in] - Destination tensor
* ``dst_offset`` [in] - Offset into the destination tensor
* ``size`` [in] - Number of bytes to copy
* ``queue`` [in] - XU queue to use
* ``err`` [in] - error tracker
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:98 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L98>`_

nrta_execute_schedule
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_execute_schedule(nrt_model_t *model, const nrt_tensor_set_t *input, 
                                    nrt_tensor_set_t *output, int queue, 
                                    nrta_error_tracker_t *err, nrta_seq_t *req_sequence);

Schedules an asynchronous request to execute a model with specified inputs and outputs.

**Parameters:**

* ``model`` [in] - The model to schedule for execution
* ``input`` [in] - Set of input tensors for the model
* ``output`` [in] - Set of tensors to receive the outputs
* ``queue`` [in] - XU queue to use, must be 0
* ``err`` [in] - error tracker
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on successful preparation, appropriate error code otherwise

**Source**: `nrt_async.h:118 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L118>`_

nrta_is_completed
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_is_completed(nrta_seq_t seq, bool *is_completed);

Checks completion status of a scheduled request.

**Parameters:**

* ``seq`` [in] - Scheduled request sequence id
* ``is_completed`` [out] - true if the request is completed, false otherwise

**Returns:** NRT_SUCCESS if the request is completed, NRT_INVALID if the seq is not valid

**Source**: `nrt_async.h:159 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L159>`_

nrta_get_completion_handle
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_get_completion_handle(nrta_seq_t seq, int *fd);

Returns a pollable file descriptor that is READABLE when the execution request specified by seq is complete.

**Parameters:**

* ``seq`` [in] - sequence to track completion
* ``fd`` [out] - FD associate with the sequence.

**Note:** The file descriptor must be passed to ``close`` to free the handle once not needed anymore.

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:185 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L185>`_

nrta_error_tracker_create
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_error_tracker_create(uint32_t lnc_idx, nrta_error_tracker_t **error_tracker);

Creates an error tracker list.

**Parameters:**

* ``lnc_idx`` [in] - Logical Neuron Core this list will be used for
* ``error_tracker`` [out] - Created list.

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:195 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L195>`_

nrta_error_tracker_destroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   void nrta_error_tracker_destroy(nrta_error_tracker_t *error_tracker);

Frees an error tracker list.

**Parameters:**

* ``error_tracker`` [in] - Error tracker list to free

**Source**: `nrt_async.h:201 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L201>`_
