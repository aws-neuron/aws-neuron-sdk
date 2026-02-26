.. _api_nrt_version_h:

nrt_version.h
=============

Neuron Runtime version information API.

**Source**: `src/libnrt/include/nrt/nrt_version.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_version.h>`_

Constants
---------

RT_VERSION_DETAIL_LEN
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   #define RT_VERSION_DETAIL_LEN 128

Maximum length for version detail string.

**Source**: `nrt_version.h:12 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_version.h#L12>`_

GIT_HASH_LEN
^^^^^^^^^^^^

.. code-block:: c

   #define GIT_HASH_LEN 64

Maximum length for git hash string.

**Source**: `nrt_version.h:13 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_version.h#L13>`_

Structures
----------

nrt_version_t
^^^^^^^^^^^^^

.. code-block:: c

   typedef struct nrt_version {
       uint64_t rt_major;
       uint64_t rt_minor;
       uint64_t rt_patch;
       uint64_t rt_maintenance;
       char rt_detail[RT_VERSION_DETAIL_LEN];
       char git_hash[GIT_HASH_LEN];
   } nrt_version_t;

NRT version information structure.

**Source**: `nrt_version.h:15 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_version.h#L15>`_

Functions
---------

nrt_get_version
^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_get_version(nrt_version_t *ver, size_t size);

Get the NRT library version.

**Parameters:**

* ``ver`` [out] - Pointer to nrt version struct
* ``size`` [in] - Length of the data needed to be filled in the nrt_version_struct

**Returns:** NRT_STATUS_SUCCESS on success.

**Source**: `nrt_version.h:28 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_version.h#L28>`_
