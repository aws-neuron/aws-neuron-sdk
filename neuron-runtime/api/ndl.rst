.. _api_ndl_h:

ndl.h
=====

Neuron Driver Library (NDL) API - Low-level interface to Neuron devices.

**Source**: `src/libnrt/include/ndl/ndl.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h>`_

Enumerations
------------

NQ_DEV_TYPE
^^^^^^^^^^^

.. code-block:: c

   typedef enum NQ_DEV_TYPE {
       NQ_DEV_TYPE_NEURON_CORE = 0,
       NQ_DEV_TYPE_TOPSP,
       NQ_DEV_TYPE_MAX,
   } ndl_nq_dev_t;

Device type enumeration for notification queues.

**Source**: `ndl.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L18>`_

Constants
---------

NEURON_MAX_DEVICES
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEURON_MAX_DEVICES MAX_NEURON_DEVICE_COUNT

Maximum neuron devices supported on a system.

**Source**: `ndl.h:24 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L24>`_

NEURON_DEVICE_PREFIX
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEURON_DEVICE_PREFIX "/dev/neuron"

Device file prefix for Neuron devices.

**Source**: `ndl.h:25 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L25>`_

MAX_HBM_PER_DEVICE
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define MAX_HBM_PER_DEVICE 4

Maximum HBM (High Bandwidth Memory) regions per device.

**Source**: `ndl.h:28 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L28>`_

MAX_NEURON_DEVICE_COUNT
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define MAX_NEURON_DEVICE_COUNT 64

Maximum neuron devices supported on a system.

**Source**: `ndl.h:78 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L78>`_

MAX_NC_PER_DEVICE
^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define MAX_NC_PER_DEVICE 8

Maximum neuron cores per device.

**Source**: `ndl.h:81 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L81>`_

Structures
----------

ndl_version_info_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndl_version_info {
       uint16_t driver_major_version;
       uint16_t driver_minor_version;
       char driver_full_version[DRIVER_VERSION_MAX_SIZE];
       uint16_t library_major_version;
       uint16_t library_minor_version;
   } ndl_version_info_t;

Version information for driver and library.

**Source**: `ndl.h:31 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L31>`_

ndl_device_init_param_t
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndl_device_init_param {
       bool initialize_device;
       int num_dram_regions;
       bool map_hbm;
   } ndl_device_init_param_t;

Device initialization parameters.

**Source**: `ndl.h:59 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L59>`_

ndl_device_t
^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndl_device {
       uint8_t device_index;
       uint8_t device_type;
       uint16_t device_revision;
       uint8_t connected_device_count;
       uint8_t connected_devices[MAX_NEURON_DEVICE_COUNT];
       uint64_t csr_base[2];
       uint64_t csr_size[2];
       ndl_copy_buf_t cpy_bufs[MAX_NC_PER_DEVICE];
       void *hbm_va[MAX_HBM_PER_DEVICE];
       size_t hbm_size;
       uint32_t hbm_va_cnt;
       uint32_t shift_hbm_size;
       uint64_t hbm_offset[MAX_HBM_PER_DEVICE];
       uint8_t context[];
   } ndl_device_t;

Device structure containing device information and resources.

**Source**: `ndl.h:83 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L83>`_

ndl_mem_info_t
^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndl_mem_info {
       ndl_device_t *device;
       __u64 driver_handle;
       uint64_t pa;
       uint64_t mmap_offset;
       uint64_t size;
       uint32_t align;
       void *mmap_va;
       uint32_t host_memory;
       int nc_id;
   } ndl_mem_info_t;

Memory allocation information.

**Source**: `ndl.h:107 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L107>`_

ndl_notification_context_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndl_notification_context {
       union {
           uint8_t nc_id;
           uint8_t nq_dev_id;
       };
       ndl_nq_dev_t nq_dev_type;
       uint8_t nq_type;
       uint8_t engine_index;
       uint32_t size;
       int fd;
       uint64_t offset;
       uint64_t mem_handle;
       void *va;
       ndl_mem_info_t *mem_info;
   } ndl_notification_context_t;

Notification queue context.

**Source**: `ndl.h:119 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L119>`_

Functions
---------

ndl_get_version
^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_get_version(ndl_version_info_t *version);

Get version info.

**Parameters:**

* ``version`` [out] - Buffer to store the version information.

**Returns:** 0 on success, -1 on failed to read driver version.

**Source**: `ndl.h:45 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L45>`_

ndl_open_device
^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_open_device(int device_index, ndl_device_init_param_t *params, ndl_device_t **device);

Called by app the first time when it accesses the device.

**Parameters:**

* ``device_index`` [in] - device index that is to be opened
* ``params`` [in] - device initialization parameters
* ``device`` [out] - device specific information

**Returns:** 0 on success, -1 on failure

**Source**: `ndl.h:141 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L141>`_

ndl_close_device
^^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_close_device(ndl_device_t *device);

Called by app when it is done. After this, device cannot be accessed.

**Parameters:**

* ``device`` [in] - Device to close.

**Returns:** 0 on success, -1 on failure

**Source**: `ndl.h:150 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L150>`_

ndl_available_devices
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_available_devices(int *device_indexes, int device_indexes_size);

Get all the device index.

**Parameters:**

* ``device_indexes`` [out] - Buffer to store device indexes.
* ``device_indexes_size`` [in] - Size of the buffer in dwords.

**Returns:** Number of devices found.

**Source**: `ndl.h:159 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L159>`_

ndl_memory_alloc
^^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_memory_alloc(ndl_device_t *device, size_t size, uint64_t align, uint32_t host_memory, 
                        uint32_t dram_channel, uint32_t dram_region, uint32_t nc_id, 
                        uint32_t mem_alloc_type, uint64_t *mem_handle);

Allocates memory.

**Parameters:**

* ``device`` [in] - Device to be associated with the allocation.
* ``size`` [in] - Number of bytes to allocate.
* ``host_memory`` [in] - If true allocate from host memory instead of using device memory.
* ``dram_channel`` [in] - DRAM channel to use in the device memory.
* ``dram_region`` [in] - DRAM region to use in the device memory.
* ``nc_id`` [in] - NC ID to use in the device
* ``mem_alloc_type`` [in] - Type of memory allocation
* ``mem_handle`` [out] - Allocated memory handle would be stored here.

**Returns:** 0 on success.

**Source**: `ndl.h:227 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L227>`_

ndl_memory_map
^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_memory_map(uint64_t mem_handle, void **va);

Map given memory handle into virtual address space.

**Parameters:**

* ``mem_handle`` [in] - Handle to map.
* ``va`` [out] - Resulting virtual address.

**Returns:** 0 on success

**Source**: `ndl.h:240 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L240>`_

ndl_memory_free
^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_memory_free(uint64_t mem_handle);

Frees already allocated memory.

**Parameters:**

* ``mem_handle`` [in] - Memory handle to be freed.

**Returns:** 0 on success.

**Source**: `ndl.h:255 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L255>`_

ndl_notification_init
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int ndl_notification_init(ndl_device_t *device, int nq_dev_id, ndl_nq_dev_t nq_dev_type, 
                             uint8_t nq_type, uint8_t engine_index, uint32_t size, 
                             bool on_host_memory, uint32_t dram_channel, uint32_t dram_region,
                             uint64_t *notification_context);

Configure notification queue.

**Parameters:**

* ``device`` [in] - Device
* ``nq_dev_id`` [in] - Notification device index
* ``nq_dev_type`` [in] - Notification device type
* ``nq_type`` [in] - Notification queue type
* ``engine_index`` [in] - Engine index
* ``size`` [in] - Size in bytes
* ``on_host_memory`` [in] - If true, NQ is created on host memory
* ``dram_channel`` [in] - If NQ is created on device, DRAM channel to use
* ``dram_region`` [in] - If NQ is created on device, DRAM region to use
* ``notification_context`` [out] - Resulting NQ context.

**Returns:** 0 on success.

**Source**: `ndl.h:625 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L625>`_

ndl_reset_ncs
^^^^^^^^^^^^^

.. code-block:: c

   int ndl_reset_ncs(int device_index, int nc_map, uint32_t *request_id);

Reset given NCs within a device.

**Parameters:**

* ``device_index`` [in] - Device to reset.
* ``nc_map`` [in] - NCs to reset (-1 to reset entire device)
* ``request_id`` [out] - ID for this reset request

**Returns:** 0 on success.

**Source**: `ndl.h:476 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/ndl/ndl.h#L476>`_
