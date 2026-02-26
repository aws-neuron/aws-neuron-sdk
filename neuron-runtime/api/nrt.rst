.. _api_nrt_h:

nrt.h
=====

Neuron Runtime (NRT) API - Main interface for loading and executing models on Neuron devices.

**Source**: `src/libnrt/include/nrt/nrt.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h>`_

Constants
---------

NRT_MAJOR_VERSION
^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NRT_MAJOR_VERSION 2

Major version of runtime.

**Source**: `nrt.h:21 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L21>`_

NRT_MINOR_VERSION
^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NRT_MINOR_VERSION 0

Minor version of runtime.

**Source**: `nrt.h:22 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L22>`_

Enumerations
------------

nrt_tensor_placement_t
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum {
       NRT_TENSOR_PLACEMENT_DEVICE,
       NRT_TENSOR_PLACEMENT_HOST,
       NRT_TENSOR_PLACEMENT_VIRTUAL,
   } nrt_tensor_placement_t;

Tensor placement options.

**Source**: `nrt.h:34 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L34>`_

nrt_framework_type_t
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum {
       NRT_FRAMEWORK_TYPE_INVALID = 0,
       NRT_FRAMEWORK_TYPE_NO_FW = 1,
       NRT_FRAMEWORK_TYPE_TENSORFLOW,
       NRT_FRAMEWORK_TYPE_PYTORCH,
       NRT_FRAMEWORK_TYPE_MXNET,
       NRT_FRAMEWORK_TYPE_PRECHECK,
   } nrt_framework_type_t;

Framework types supported by NRT.

**Source**: `nrt.h:40 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L40>`_

nrt_dtype_t
^^^^^^^^^^^

.. code-block:: c

   typedef enum nrt_dtype {
       NRT_DTYPE_UNKNOWN = 0x0,
       NRT_DTYPE_INVALID = 0x0,
       NRT_DTYPE_FP8_E3 = 0xD,
       NRT_DTYPE_FP8_E4 = 0xE,
       NRT_DTYPE_FP8_E5 = 0xF,
       NRT_DTYPE_FLOAT16 = 0x7,
       NRT_DTYPE_BFLOAT16 = 0x6,
       NRT_DTYPE_FLOAT32 = 0xA,
       NRT_DTYPE_FP32R = 0xB,
       NRT_DTYPE_UINT8 = 0x3,
       NRT_DTYPE_UINT16 = 0x5,
       NRT_DTYPE_UINT32 = 0x9,
       NRT_DTYPE_UINT64 = 0x1,
       NRT_DTYPE_INT8 = 0x2,
       NRT_DTYPE_INT16 = 0x4,
       NRT_DTYPE_INT32 = 0x8,
       NRT_DTYPE_INT64 = 0xC,
   } nrt_dtype_t;

Data types supported by NRT.

**Source**: `nrt.h:90 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L90>`_

nrt_op_type_t
^^^^^^^^^^^^^

.. code-block:: c

   typedef enum nrt_op_type {
       NRT_OP_ADD = 0x0,
       NRT_OP_FMA = 0x1,
       NRT_OP_MAX = 0x2,
       NRT_OP_MIN = 0x3,
       NRT_OP_INVALID = 0xF,
   } nrt_op_type_t;

Operation types for collectives.

**Source**: `nrt.h:83 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L83>`_

nrt_cc_op_type_t
^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum nrt_cc_op_type {
       NRT_CC_ALLGATHER,
       NRT_CC_ALLREDUCE,
       NRT_CC_REDUCESCATTER
   } nrt_cc_op_type_t;

Collective communication operation types.

**Source**: `nrt.h:111 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L111>`_

Structures
----------

nrt_instance_info_t
^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_instance_info {
       uint32_t family;
       uint32_t size;
       char arch_name[16];
       char device_revision[8];
   } nrt_instance_info_t;

Instance information structure.

**Source**: `nrt.h:117 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L117>`_

nrt_tensor_batch_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_batch {
       const nrt_tensor_t *tensor;
       const nrt_tensor_batch_op_t *ops;
       uint32_t num_ops;
   } nrt_tensor_batch_t;

A batch of tensor operations on a single tensor.

**Source**: `nrt.h:343 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L343>`_

nrt_tensor_device_allocation_info_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_device_allocation_info {
       uint64_t physical_address;
       size_t size;
       int hbm_index;
   } nrt_tensor_device_allocation_info_t;

Returns on device allocation info for a tensor.

**Source**: `nrt.h:442 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L442>`_

nrt_vnc_memory_stats_t
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_vnc_memory_stats {
       size_t bytes_used;
       size_t bytes_limit;
   } nrt_vnc_memory_stats_t;

NRT memory stats for a VNC.

**Source**: `nrt.h:509 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L509>`_

nrt_cc_comm_t
^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_cc_comm {
       uint32_t *replica_group;
       uint32_t rank;
       uint32_t rank_n;
       uint32_t ctx_device_id;
       uint32_t ctx_device_count;
       uint32_t vnc;
   } nrt_cc_comm_t;

Communicator for collective operations.

**Source**: `nrt.h:545 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L545>`_

nrt_tensor_list_t
^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_tensor_list {
       nrt_tensor_t **tensors;
       size_t num_tensors;
   } nrt_tensor_list_t;

List of tensors.

**Source**: `nrt.h:554 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L554>`_

Functions
---------

nrt_init
^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_init(nrt_framework_type_t framework, const char *fw_version, const char *fal_version);

Initialize neuron runtime.

**Parameters:**

* ``framework`` [in] - Type of the framework.
* ``fw_version`` [in] - Framework version as string. (eg 2.1)
* ``fal_version`` [in] - Framework Abstraction Layer version as string.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:133 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L133>`_

nrt_close
^^^^^^^^^

.. code-block:: c

   void nrt_close();

Closes all the devices and cleans up the runtime state.

**Source**: `nrt.h:138 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L138>`_

nrt_load
^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_load(const void *neff_bytes, size_t size, int32_t vnc, int32_t vnc_count, nrt_model_t **model);

Load given NEFF and place it in one or more neuron cores.

**Parameters:**

* ``neff_bytes`` [in] - Pointer to NEFF data.
* ``size`` [in] - Length of the NEFF data.
* ``vnc`` [in] - VNC index where the NEFF should be loaded(-1 means runtime would automatically load in first free VNC).
* ``vnc_count`` [in] - DEPRECATED: always use -1
* ``model`` [out] - Resulting model would be stored here.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:149 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L149>`_

nrt_unload
^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_unload(nrt_model_t *model);

Unload given model and free up device and host resources.

**Parameters:**

* ``model`` - Model to unload.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:172 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L172>`_

nrt_execute
^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_execute(nrt_model_t *model, const nrt_tensor_set_t *input_set, nrt_tensor_set_t *output_set);

Execute given model with given inputs and collect outputs.

**Parameters:**

* ``model`` [in] - Model to execute.
* ``input_set`` [in] - Set of input tensors.
* ``output_set`` [in] - Set of output tensors.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:256 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L256>`_

nrt_tensor_allocate
^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t tensor_placement, int vnc, size_t size, 
                                  const char *name, nrt_tensor_t **tensor);

Allocates a tensor that can be passed and used by a model for compute.

**Parameters:**

* ``tensor_placement`` [in] - Where the tensor would be allocated (device, host, or virtual memory)
* ``vnc`` [in] - Virtual Neuron Core id to allocate the tensor on. Pass in -1 if allocating tensors on host memory.
* ``size`` [in] - Size in bytes of the tensor to allocate.
* ``name`` [in] - OPTIONAL. Name of the tensor.
* ``tensor`` [out] - Pointer to newly created tensor will be stored here.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:283 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L283>`_

nrt_tensor_free
^^^^^^^^^^^^^^^

.. code-block:: c

   void nrt_tensor_free(nrt_tensor_t **tensor);

Deallocates a tensor created by "nrt_tensor_allocate".

**Parameters:**

* ``tensor`` [in] - Deallocates given tensor.

**Source**: `nrt.h:292 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L292>`_

nrt_tensor_read
^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_tensor_read(const nrt_tensor_t *tensor, void *buf, size_t offset, size_t size);

Copies data from tensor to passed in buffer.

**Parameters:**

* ``tensor`` [in] - Tensor used to reference the tensor to read from.
* ``buf`` [out] - Buffer used to store data read from the tensor.
* ``offset`` [in] - Offset into the tensor to read from.
* ``size`` [in] - Number of bytes to read.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:303 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L303>`_

nrt_tensor_write
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_tensor_write(nrt_tensor_t *tensor, const void *buf, size_t offset, size_t size);

Copies data from passed in buffer to tensor.

**Parameters:**

* ``tensor`` [in/out] - Tensor used to reference the tensor to write to.
* ``buf`` [in] - Buffer used to store data to write to the tensor.
* ``offset`` [in] - Offset into the tensor to write to.
* ``size`` [in] - Number of bytes to write.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:315 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L315>`_

nrt_tensor_copy
^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_tensor_copy(const nrt_tensor_t *src, size_t src_offset, nrt_tensor_t *dst, 
                              size_t dst_offset, size_t size);

Copies data between tensors.

**Parameters:**

* ``src`` [in] - Tensor to copy from.
* ``src_offset`` [in] - Offset into the source tensor to copy from.
* ``dst`` [out] - Tensor to copy to.
* ``dst_offset`` [in] - Offset into the destination tensor to copy to.
* ``size`` [in] - Number of bytes to copy.

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:381 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L381>`_

nrt_get_total_vnc_count
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_get_total_vnc_count(uint32_t *vnc_count);

Returns VirtualNeuronCores available in instance.

**Parameters:**

* ``vnc_count`` [out] - VirtualNeuronCores available in instance.

**Note:** This API can be called before nrt_init().

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt.h:203 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h#L203>`_
