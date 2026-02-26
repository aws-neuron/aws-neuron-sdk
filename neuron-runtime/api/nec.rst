.. _api_nec_h:

nec.h
=====

Neuron Elastic Collectives (NEC) API - Collective operations for distributed computing on Neuron devices.

**Source**: `src/libnrt/include/nrt/nec.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h>`_

Overview
--------

This is the main component for Neuron Elastic Collectives in Neuron Runtime (NRT). This provides collective operations to applications offloaded by the device including collective comm init, receiving (post) operations, building resources for the operation, triggering the operation and polling its completion.

Constants
---------

NEC_MAX_CHANNELS
^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEC_MAX_CHANNELS 32

Maximum channels (matches MAXCHANNELS in NCCL).

**Source**: `nec.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L18>`_

NEC_MAX_COMM_N
^^^^^^^^^^^^^^

.. code-block:: c

   #define NEC_MAX_COMM_N 12

Max supported replica-groups in NEFF.

**Source**: `nec.h:26 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L26>`_

NEC_MAX_STREAM_N
^^^^^^^^^^^^^^^^

.. code-block:: c

   #define NEC_MAX_STREAM_N 4

The maximum number of concurrent cc execution.

**Source**: `nec.h:56 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L56>`_

Enumerations
------------

nec_pod_type_t
^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum nec_pod_type {
       NEC_POD_TYPE_NONE,
       NEC_POD_TYPE_P2P,
       NEC_POD_TYPE_SWITCH,
       NEC_POD_TYPE_INVALID
   } nec_pod_type_t;

Pod type enumeration (translated from what KaenaDriver returns).

**Source**: `nec.h:103 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L103>`_

enc_pattern_t
^^^^^^^^^^^^^

.. code-block:: c

   typedef enum enc_pattern {
       ENC_PATTERN_RING,
       ENC_PATTERN_MESH,
       ENC_PATTERN_INVALID,
   } enc_pattern_t;

Communication pattern types.

**Source**: `nec.h:244 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L244>`_

Structures
----------

nccl_comm_info_t
^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nccl_comm_info {
       uint64_t cluster_id;
       time_t epoch;
       int neuron_dev;
       int rank;
       int rank_n;
       int local_rank_n;
       int local_rack_rank_n;
       int node;
       int node_n;
       bool enable_pod;
       bool use_net;
       int pod;
       int pod_n;
       int pod_node;
       int pod_node_n;
       struct enc_peer_info *peers;
       int channel_n;
       struct enc_ring rings[NEC_MAX_CHANNELS];
       int kangaring_channel_n;
       int* kangaring_paths[NEC_MAX_CHANNELS];
       int mla_cycle_n;
       int* mla_cycles[NEC_MAX_CHANNELS];
   } nccl_comm_info_t;

Comm info to query from NCCL.

**Source**: `nec.h:732 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L732>`_

enc_neuron_device_info_t
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct enc_neuron_device_info {
       int nec_dev_id;
       int mla_idx;
       int tpb_idx;
       int host_device_id;
       int routing_id;
       uint64_t pod_id;
       nec_pod_type_t pod_type;
       uint32_t pod_node_id;
       uint32_t virtual_server_id;
       enc_proxy_histogram_config_t histogram_config;
   } enc_neuron_device_info_t;

Neuron Device information. This data structure is used to send the device information from KaenaRuntime to KaenaNCCL for nccl communicator building.

**Source**: `nec.h:787 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L787>`_

nec_version_info_t
^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nec_version_info {
       uint64_t major;
       uint64_t minor;
       uint64_t patch;
       uint64_t maintenance;
       char git_hash[16];
       uint64_t compatibility_version;
       uint8_t future_fields[];
   } nec_version_info_t;

NEC version information.

**Source**: `nec.h:920 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L920>`_

Functions
---------

nec_get_device_count
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int nec_get_device_count(int *available_devices_array, uint32_t array_size);

Query device information - get device count.

**Parameters:**

* ``available_devices_array`` [out] - Array to store available device IDs
* ``array_size`` [in] - Size of the array

**Returns:** Number of available devices

**Source**: `nec.h:917 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L917>`_

nec_get_virtual_core_size
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nec_get_virtual_core_size(uint32_t *virtual_core_size);

Query vcore size.

**Parameters:**

* ``virtual_core_size`` [out] - Virtual core size

**Returns:** NRT_STATUS_SUCCESS on success

**Source**: `nec.h:923 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L923>`_

nec_get_version_info
^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nec_get_version_info(nec_version_info_t *version_info);

Get NEC version information.

**Parameters:**

* ``version_info`` [out] - Version information structure

**Returns:** NRT_STATUS_SUCCESS on success

**Source**: `nec.h:932 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nec.h#L932>`_
