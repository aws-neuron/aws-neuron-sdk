.. _api_neuron_driver_shared_h:

neuron_driver_shared.h
======================

Shared definitions between Neuron driver and runtime.

**Source**: `src/libnrt/include/ndl/neuron_driver_shared.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h>`_

Enumerations
------------

neuron_driver_feature_flag
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   enum neuron_driver_feature_flag {
       NEURON_DRIVER_FEATURE_DMABUF = 1ull << 0,
       NEURON_DRIVER_FEATURE_ASYNC_DMA = 1ull << 1,
       NEURON_DRIVER_FEATURE_BATCH_DMAQ_INIT = 1ull << 2,
       NEURON_DRIVER_FEATURE_BIG_CORE_MAPS = 1ull << 3,
       NEURON_DRIVER_FEATURE_MEM_ALLOC_TYPE = 1ull << 4,
       NEURON_DRIVER_FEATURE_HBM_SCRUB = 1ull << 5,
       NEURON_DRIVER_FEATURE_MEM_ALLOC64 = 1ull << 6,
       NEURON_DRIVER_FEATURE_CONTIGUOUS_SCRATCHPAD = 1ull << 7,
       NEURON_DRIVER_FEATURE_ZEROCOPY = 1ull << 8,
   };

Feature flags for driver capabilities.

**Source**: `neuron_driver_shared.h:11 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L11>`_

neuron_pod_ctrl_req
^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   enum neuron_pod_ctrl_req {
       NEURON_NPE_POD_CTRL_REQ_POD = 0,
       NEURON_NPE_POD_CTRL_REQ_SINGLE_NODE = 1,
       NEURON_NPE_POD_CTRL_REQ_KILL = 2,
       NEURON_NPE_POD_CTRL_SET_MODE = 3,
   };

Pod control request types.

**Source**: `neuron_driver_shared.h:40 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L40>`_

neuron_ultraserver_mode
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   enum neuron_ultraserver_mode {
       NEURON_ULTRASERVER_MODE_UNSET = 0,
       NEURON_ULTRASERVER_MODE_X4 = 1,
       NEURON_ULTRASERVER_MODE_X2H = 2,
       NEURON_ULTRASERVER_MODE_X2V = 3,
       NEURON_ULTRASERVER_MODE_X1 = 4,
   };

Ultraserver configuration modes.

**Source**: `neuron_driver_shared.h:47 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L47>`_

neuron_dma_queue_type
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   enum neuron_dma_queue_type {
       NEURON_DMA_QUEUE_TYPE_TX = 0,
       NEURON_DMA_QUEUE_TYPE_RX,
       NEURON_DMA_QUEUE_TYPE_COMPLETION,
   };

DMA queue types.

**Source**: `neuron_driver_shared.h:63 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L63>`_

NQ_DEVICE_TYPE
^^^^^^^^^^^^^^

.. code-block:: c

   enum NQ_DEVICE_TYPE {
       NQ_DEVICE_TYPE_NEURON_CORE = 0,
       NQ_DEVICE_TYPE_TOPSP,
       NQ_DEVICE_TYPE_MAX
   };

Notification queue device types.

**Source**: `neuron_driver_shared.h:115 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L115>`_

NQ_TYPE
^^^^^^^

.. code-block:: c

   enum NQ_TYPE {
       NQ_TYPE_TRACE = 0,
       NQ_TYPE_NOTIFY,
       NQ_TYPE_EVENT,
       NQ_TYPE_ERROR,
       NQ_TYPE_TRACE_DMA,
       NQ_TYPE_THROTTLE,
       NQ_TYPE_MAX
   };

Notification queue types.

**Source**: `neuron_driver_shared.h:123 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L123>`_

mem_alloc_category_t
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum {
       NEURON_MEMALLOC_TYPE_UNKNOWN_HOST,
       NEURON_MEMALLOC_TYPE_CODE_HOST,
       NEURON_MEMALLOC_TYPE_TENSORS_HOST,
       NEURON_MEMALLOC_TYPE_CONSTANTS_HOST,
       NEURON_MEMALLOC_TYPE_MISC_HOST,
       NEURON_MEMALLOC_TYPE_NCDEV_HOST,
       NEURON_MEMALLOC_TYPE_NOTIFICATION_HOST,
       NEURON_MEMALLOC_TYPE_UNKNOWN_DEVICE,
       NEURON_MEMALLOC_TYPE_CODE_DEVICE,
       NEURON_MEMALLOC_TYPE_TENSORS_DEVICE,
       NEURON_MEMALLOC_TYPE_CONSTANTS_DEVICE,
       NEURON_MEMALLOC_TYPE_SCRATCHPAD_DEVICE,
       NEURON_MEMALLOC_TYPE_MISC_DEVICE,
       NEURON_MEMALLOC_TYPE_NCDEV_DEVICE,
       NEURON_MEMALLOC_TYPE_COLLECTIVES_DEVICE,
       NEURON_MEMALLOC_TYPE_SCRATCHPAD_NONSHARED_DEVICE,
       NEURON_MEMALLOC_TYPE_NOTIFICATION_DEVICE,
       NEURON_MEMALLOC_TYPE_DMA_RINGS_HOST,
       NEURON_MEMALLOC_TYPE_DMA_RINGS_DEVICE,
       NEURON_MEMALLOC_TYPE_CONTIGUOUS_SCRATCHPAD_DEVICE,
       NEURON_MEMALLOC_TYPE_MAX
   } mem_alloc_category_t;

Memory allocation categories for sysfs counters.

**Source**: `neuron_driver_shared.h:234 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L234>`_

Structures
----------

neuron_dma_eng_state
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   struct neuron_dma_eng_state {
       __u32 revision_id;
       __u32 max_queues;
       __u32 num_queues;
       __u32 tx_state;
       __u32 rx_state;
   };

DMA engine state information.

**Source**: `neuron_driver_shared.h:76 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L76>`_

neuron_dma_queue_state
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   struct neuron_dma_queue_state {
       __u32 hw_status;
       __u32 sw_status;
       __u64 base_addr;
       __u32 length;
       __u32 head_pointer;
       __u32 tail_pointer;
       __u64 completion_base_addr;
       __u32 completion_head;
   };

DMA queue state information.

**Source**: `neuron_driver_shared.h:84 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L84>`_

neuron_uuid
^^^^^^^^^^^

.. code-block:: c

   struct neuron_uuid {
       __u8 value[32];
   };

UUID structure for model identification.

**Source**: `neuron_driver_shared.h:163 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L163>`_

neuron_app_info
^^^^^^^^^^^^^^^

.. code-block:: c

   struct neuron_app_info {
       __s32 pid;
       __u8 nc_lock_map;
       struct neuron_uuid uuid_data[APP_INFO_MAX_MODELS_PER_DEVICE];
       size_t host_mem_size;
       size_t device_mem_size;
   };

Application information including PID, locked neuron cores, and memory usage.

**Source**: `neuron_driver_shared.h:175 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L175>`_

neuron_memcpy_batch_t
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct neuron_memcpy_batch {
       __u64 mem_handle;
       __u64 mem_handle_offset;
       const nrt_tensor_batch_op_t *ops_ptr;
       __u32 num_ops;
       __u16 bar4_wr_threshold;
       __u16 flags;
       void *context;
   } neuron_memcpy_batch_t;

A batch of copy operations for efficient data transfer.

**Source**: `neuron_driver_shared.h:220 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L220>`_

nds_header_t
^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_header {
       char signature[4];
       int version;
   } nds_header_t;

Neuron Datastore header structure.

**Source**: `neuron_driver_shared.h:330 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L330>`_

Constants
---------

NEURON_DMA_H2T_DEFAULT_QID
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEURON_DMA_H2T_DEFAULT_QID (-1)

H2T DMA Default Queue id.

**Source**: `neuron_driver_shared.h:108 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L108>`_

NEURON_MAX_PROCESS_PER_DEVICE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEURON_MAX_PROCESS_PER_DEVICE 16

Maximum processes per device.

**Source**: `neuron_driver_shared.h:167 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L167>`_

NDS_MAX_NEURONCORE_COUNT
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NDS_MAX_NEURONCORE_COUNT (4)

Maximum neuron core count for NDS.

**Source**: `neuron_driver_shared.h:323 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/neuron_driver_shared.h#L323>`_
