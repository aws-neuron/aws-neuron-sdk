.. _api_neuron_ds_h:

neuron_ds.h
===========

Neuron Datastore (NDS) API - Shared memory datastore for runtime metrics and model information.

**Source**: `src/libnrt/include/nrt/nds/neuron_ds.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h>`_

Constants
---------

OBJECT_TYPE_MODEL_NODE_INFO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define OBJECT_TYPE_MODEL_NODE_INFO (0)

NDS object type for model node information.

**Source**: `neuron_ds.h:19 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L19>`_

OBJECT_TYPE_PROCESS_INFO
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define OBJECT_TYPE_PROCESS_INFO (1)

NDS object type for process information.

**Source**: `neuron_ds.h:20 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L20>`_

MODEL_MEM_USAGE_LOCATION_COUNT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define MODEL_MEM_USAGE_LOCATION_COUNT 2

Number of memory usage locations tracked.

**Source**: `neuron_ds.h:24 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L24>`_

Enumerations
------------

feature_bitmap_bit_index_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum feature_bitmap_bit_index {
       BIT_INDEX_TEST_FEATURE = 0,
       BIT_INDEX_MULTICORE_FEATURE = 1,
       BIT_INDEX_COUNT = BIT_INDEX_MULTICORE_FEATURE + 1
   } feature_bitmap_bit_index_t;

Feature bitmap's bit index information.

**Source**: `neuron_ds.h:88 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L88>`_

Structures
----------

nds_mem_usage_info_t
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_mem_usage_info {
       size_t total_size;
       uint32_t chunk_count;
   } nds_mem_usage_info_t;

Aggregated data for all chunks of the same type/location.

**Source**: `neuron_ds.h:45 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L45>`_

nds_model_node_info_t
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_model_node_info {
       uint32_t model_id;
       uint32_t model_node_id;
       char name[256];
       char uuid[16];
       uint8_t nc_index;
       uint8_t sg_index;
   } nds_model_node_info_t;

Loaded model node information.

**Source**: `neuron_ds.h:51 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L51>`_

nds_model_node_mem_usage_info_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_model_node_mem_usage_info {
       nds_mem_usage_info_t model_mem_usage[MODEL_MEM_USAGE_LOCATION_COUNT][NDS_DMA_MEM_USAGE_SLOT_COUNT];
   } nds_model_node_mem_usage_info_t;

Loaded model node memory usage information.

**Source**: `neuron_ds.h:61 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L61>`_

nds_version_info_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_version_info {
       uint8_t major;
       uint8_t minor;
       uint32_t build;
   } nds_version_info_t;

Version information.

**Source**: `neuron_ds.h:66 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L66>`_

nds_process_info_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nds_process_info {
       int8_t framework_type;
       char tag[32];
       nds_version_info_t framework_version;
       nds_version_info_t fal_version;
       nds_version_info_t runtime_version;
   } nds_process_info_t;

Process information-related struct.

**Source**: `neuron_ds.h:73 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L73>`_

Functions
---------

nds_open
^^^^^^^^

.. code-block:: c

   int nds_open(ndl_device_t *device, pid_t pid, nds_instance_t **inst);

Opens NDS for the given pid. If pid == 0, it acquires it for the current PID and it's opened in read-write mode. If pid != 0, it acquires it for the provided PID and it's opened as read-only.

**Parameters:**

* ``device`` [in] - ndl_device used to open this NDS
* ``pid`` [in] - pid for which to open the NDS, if 0 - it's opened as r/w for the current process
* ``inst`` [out] - address of a pointer which will contain the instance handle

**Returns:** non zero in case of error

**Source**: `neuron_ds.h:102 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L102>`_

nds_close
^^^^^^^^^

.. code-block:: c

   int nds_close(nds_instance_t *inst);

Releases the NDS instance and frees the data associated with it (mandatory for readers).

**Parameters:**

* ``inst`` [in] - NDS instance to close

**Returns:** non zero in case of error, the pointer gets deleted regardless

**Source**: `neuron_ds.h:110 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L110>`_

nds_increment_nc_counter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nds_increment_nc_counter(nds_instance_t *inst, int pnc_index, uint32_t counter_index, uint64_t increment);

Increments a simple per-nc counter.

**Parameters:**

* ``inst`` [in] - NDS instance
* ``pnc_index`` [in] - Neuroncore index
* ``counter_index`` [in] - Counter index
* ``increment`` [in] - Amount to increment

**Returns:** 0 on success.

**Source**: `neuron_ds.h:123 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L123>`_

nds_get_nc_counter
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nds_get_nc_counter(nds_instance_t *inst, int pnc_index, uint32_t counter_index, uint64_t *value);

Gets a simple per-nc counter.

**Parameters:**

* ``inst`` [in] - NDS instance
* ``pnc_index`` [in] - Neuroncore index
* ``counter_index`` [in] - Counter index
* ``value`` [out] - Counter value

**Returns:** 0 on success.

**Source**: `neuron_ds.h:145 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L145>`_

nds_increment_nd_counter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nds_increment_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t increment);

Increments a simple per-nd counter - may overflow.

**Parameters:**

* ``inst`` [in] - NDS instance
* ``counter_index`` [in] - Counter index
* ``increment`` [in] - Amount to increment

**Returns:** 0 on success.

**Source**: `neuron_ds.h:167 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L167>`_

nds_get_nd_counter
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nds_get_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t *value);

Gets a simple per-nd counter.

**Parameters:**

* ``inst`` [in] - NDS instance
* ``counter_index`` [in] - Counter index
* ``value`` [out] - Counter value

**Returns:** 0 on success.

**Source**: `neuron_ds.h:193 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L193>`_

nds_obj_new
^^^^^^^^^^^

.. code-block:: c

   nds_obj_handle_t nds_obj_new(nds_instance_t *inst, int type);

Creates a new NDS object with the given type.

**Parameters:**

* ``inst`` [in] - NDS instance
* ``type`` [in] - type of object to create

**Returns:** handle for newly created object

**Source**: `neuron_ds.h:220 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L220>`_

nds_obj_commit
^^^^^^^^^^^^^^

.. code-block:: c

   int nds_obj_commit(nds_obj_handle_t obj);

Writes an NDS object to the NDS memory.

**Parameters:**

* ``obj`` [in] - NDS object handle

**Returns:** 0 on success.

**Source**: `neuron_ds.h:213 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L213>`_

nds_read_all_model_nodes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nds_read_all_model_nodes(nds_instance_t *inst, nds_obj_handle_t **models, size_t *count);

Reads all model info data and returns it as an array (needs to be deleted by caller).

**Parameters:**

* ``inst`` [in] - NDS instance
* ``models`` [out] - Pointer where to write the address of an array of length count containing object handles
* ``count`` [out] - Number of models loaded (present in the models array)

**Returns:** non-NULL on success.

**Source**: `neuron_ds.h:250 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nds/neuron_ds.h#L250>`_
