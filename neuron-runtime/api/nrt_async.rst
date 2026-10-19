.. meta::
   :description: Reference for the nrt_async.h header in the AWS Neuron Runtime: Neuron Runtime asynchronous execution API for non-blocking tensor I/O and model execution.
   :date-modified: 2026-10-13

.. _api_nrt_async_h:

nrt_async.h
===========

Neuron Runtime Asynchronous Execution API - Non-blocking operations for tensor I/O and model execution.

**Source**: `src/libnrt/include/nrt/nrt_async.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h>`__

Enumerations
------------

nrta_xu_t
^^^^^^^^^

.. code-block:: c

   typedef enum {
       NRTA_XU_TENSOR_OP = 0,
       NRTA_XU_COMPUTE,
       NRTA_XU_COLLECTIVES,
       NRTA_XU_TYPE_NUM
   } nrta_xu_t;

Execution unit types.

**Source**: `nrt_async.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L20>`__

Typedefs
--------

nrta_seq_t
^^^^^^^^^^

.. code-block:: c

   typedef uint64_t nrta_seq_t;

Monotonically increasing IDs of executions. The first 16 bits are an Execution Unit ID, while the last 48 bits are a strictly ordered Sequence Number.

**Source**: `nrt_async.h:31 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L33>`__

nrta_xu_id_t
^^^^^^^^^^^^

.. code-block:: c

   typedef uint16_t nrta_xu_id_t;

Execution unit ID type.

**Source**: `nrt_async.h:32 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L34>`__

Constants
---------

NRTA_SEQ_NUM_MAX
^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NRTA_SEQ_NUM_MAX ((1ull << 48) - 1)

Maximum sequence number value.

**Source**: `nrt_async.h:34 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L36>`__

Functions
---------

nrta_tensor_write
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_write(nrt_tensor_t *tensor, const void *buf, uint64_t offset, 
                                uint64_t size, int queue, NRT_STATUS *ret, 
                                nrta_seq_t *req_sequence);

Enqueues a tensor write request. Copies the data from a host buffer to a tensor allocated on a Neuron device.

**Parameters:**

* ``tensor`` [in] - Destination tensor
* ``buf`` [in] - Host buffer containing source data
* ``offset`` [in] - Offset into the tensor
* ``size`` [in] - Number of bytes to write
* ``queue`` [in] - XU queue to use
* ``ret`` [in] - pointer to store return value of the async request upon completion
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:59 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L57>`__

nrta_tensor_read
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_read(void *buf, nrt_tensor_t *tensor, uint64_t offset, 
                               uint64_t size, int queue, NRT_STATUS *ret, 
                               nrta_seq_t *req_sequence);

Enqueues a tensor read request. Copies the data from a tensor allocated on a Neuron device to a host buffer.

**Parameters:**

* ``buf`` [in] - Destination Host buffer
* ``tensor`` [in] - Source tensor
* ``offset`` [in] - Offset into the tensor
* ``size`` [in] - Number of bytes to read
* ``queue`` [in] - XU queue to use
* ``ret`` [in] - pointer to store return value of the async request upon completion
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:77 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L81>`__

nrta_tensor_copy
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_tensor_copy(nrt_tensor_t *src, uint64_t src_offset, nrt_tensor_t *dst, 
                               uint64_t dst_offset, uint64_t size, int queue, 
                               NRT_STATUS *ret, nrta_seq_t *req_sequence);

Enqueues a tensor copy request. Copies data between two tensors allocated on the same Logical Neuron Core.

**Parameters:**

* ``src`` [in] - Source tensor
* ``src_offset`` [in] - Offset into the source tensor
* ``dst`` [in] - Destination tensor
* ``dst_offset`` [in] - Offset into the destination tensor
* ``size`` [in] - Number of bytes to copy
* ``queue`` [in] - XU queue to use
* ``ret`` [in] - pointer to store return value of the async request upon completion
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:98 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L107>`__

nrta_execute_schedule
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_execute_schedule(nrt_model_t *model, const nrt_tensor_set_t *input, 
                                    nrt_tensor_set_t *output, int queue, 
                                    NRT_STATUS *ret, nrta_seq_t *req_sequence);

Schedules an asynchronous request to execute a model with specified inputs and outputs.

**Parameters:**

* ``model`` [in] - The model to schedule for execution
* ``input`` [in] - Set of input tensors for the model
* ``output`` [in] - Set of tensors to receive the outputs
* ``queue`` [in] - XU queue to use, must be 0
* ``ret`` [in] - pointer to store return value of the async request upon completion
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on successful preparation, appropriate error code otherwise

**Source**: `nrt_async.h:118 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L129>`__

nrta_cc_prepare
^^^^^^^^^^^^^^^^^^^^^
**NOTE: The nrta_cc_prepare and nrta_cc_schedule APIs are work-in-progress and subject to change.**

.. code-block:: c

   NRT_STATUS nrta_cc_prepare(nrt_cc_comm_t *comm, nrt_tensor_list_t *input, 
                              nrt_tensor_list_t *output, nrt_dtype_t dtype, 
                              nrt_op_type_t op, nrt_cc_op_type_t cc_op
                              nrt_cc_context_t **cc_ctx);

Prepares collective context and HW configuration needed for collectives operation.
Allocates a collective context handle that is returned to the caller which is freed in the schedule thread post CC op execution.

**Parameters:**

* ``comm`` [in] - Communicator containing the replica group
* ``input`` [in] - Input tensor list
* ``output`` [out] - Output tensor list
* ``dtype`` [in] - Data type of elements
* ``op`` [in] - Reduction operation (e.g., SUM, MAX) if applicable
* ``cc_op`` [in] - Collective operation (e.g., ALLREDUCE, ALLGATHER)
* ``cc_ctx`` [out] - Collective context

**Returns:** NRT_SUCCESS on successful preparation, appropriate error code otherwise

**Source**: `nrt_async.h:155 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L155>`__

nrta_cc_schedule
^^^^^^^^^^^^^^^^^^^^^
**NOTE: The nrta_cc_prepare and nrta_cc_schedule APIs are work-in-progress and subject to change.**

.. code-block:: c

   NRT_STATUS nrta_cc_schedule(nrt_cc_context_t **cc_ctx, int queue, 
                              NRT_STATUS *ret, nrta_seq_t *req_sequence);

Schedules an asynchronous request to execute collective operation

**Parameters:**

* ``cc_ctx`` [in] - Collective context
* ``queue`` [in] - XU queue to use, must be 0
* ``ret`` [in] - pointer to store return value of the async request upon completion
* ``req_sequence`` [out] - Sequence number of the scheduled request

**Returns:** NRT_SUCCESS on successful preparation, appropriate error code otherwise

**Source**: `nrt_async.h:172 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L172>`__

nrta_is_completed
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_is_completed(nrta_seq_t seq, bool *is_completed);

Checks completion status of a scheduled request.

**Parameters:**

* ``seq`` [in] - Scheduled request sequence id
* ``is_completed`` [out] - true if the request is completed, false otherwise

**Returns:** NRT_SUCCESS if the request is completed, NRT_INVALID if the seq is not valid

**Source**: `nrt_async.h:159 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L186>`__

nrta_get_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrta_get_sequence(uint32_t lnc, nrta_xu_t xu, int queue, nrta_seq_t *seq);

Returns sequence number of the last completed request.

**Parameters:**

 * ``lnc`` [in] - LNC
 * ``xu`` [in] - XU
 * ``queue`` [in] - XU's queue
 * ``seq`` [out] - last completed sequence number

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_async.h:185 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h#L198>`__
