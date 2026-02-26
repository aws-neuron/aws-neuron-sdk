.. _api_nrt_status_h:

nrt_status.h
============

Neuron Runtime status codes and error handling.

**Source**: `src/libnrt/include/nrt/nrt_status.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_status.h>`_

Enumerations
------------

NRT_STATUS
^^^^^^^^^^

.. code-block:: c

   typedef enum {
       NRT_SUCCESS = 0,
       NRT_FAILURE = 1,
       NRT_INVALID = 2,
       NRT_INVALID_HANDLE = 3,
       NRT_RESOURCE = 4,
       NRT_TIMEOUT = 5,
       NRT_HW_ERROR = 6,
       NRT_QUEUE_FULL = 7,
       NRT_LOAD_NOT_ENOUGH_NC = 9,
       NRT_UNSUPPORTED_NEFF_VERSION = 10,
       NRT_FAIL_HOST_MEM_ALLOC = 11,
       NRT_UNINITIALIZED = 13,
       NRT_CLOSED = 14,
       NRT_QUEUE_EMPTY = 15,
       NRT_EXEC_UNIT_UNRECOVERABLE = 101,
       NRT_EXEC_BAD_INPUT = 1002,
       NRT_EXEC_COMPLETED_WITH_NUM_ERR = 1003,
       NRT_EXEC_COMPLETED_WITH_ERR = 1004,
       NRT_EXEC_NC_BUSY = 1005,
       NRT_EXEC_OOB = 1006,
       NRT_COLL_PENDING = 1100,
       NRT_EXEC_HW_ERR_COLLECTIVES = 1200,
       NRT_EXEC_HW_ERR_HBM_UE = 1201,
       NRT_EXEC_HW_ERR_NC_UE = 1202,
       NRT_EXEC_HW_ERR_DMA_ABORT = 1203,
       NRT_EXEC_SW_NQ_OVERFLOW = 1204,
       NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE = 1205,
       NRT_NETWORK_PROXY_FAILURE = 1206,
   } NRT_STATUS;

Status codes returned by NRT API functions.

**Status Codes:**

* ``NRT_SUCCESS`` - Operation completed successfully
* ``NRT_FAILURE`` - Non-specific failure
* ``NRT_INVALID`` - Invalid input (e.g., invalid NEFF, bad instruction, input tensor name/size mismatch)
* ``NRT_INVALID_HANDLE`` - Invalid handle passed
* ``NRT_RESOURCE`` - Failed to allocate a resource for requested operation
* ``NRT_TIMEOUT`` - Operation timed out
* ``NRT_HW_ERROR`` - Hardware failure
* ``NRT_QUEUE_FULL`` - Not enough space in the execution input queue
* ``NRT_LOAD_NOT_ENOUGH_NC`` - Failed to allocate enough NCs for loading a NEFF
* ``NRT_UNSUPPORTED_NEFF_VERSION`` - Unsupported version of NEFF
* ``NRT_UNINITIALIZED`` - NRT API called before nrt_init()
* ``NRT_CLOSED`` - NRT API called after nrt_close()
* ``NRT_QUEUE_EMPTY`` - Accessed a queue with no data
* ``NRT_EXEC_UNIT_UNRECOVERABLE`` - Encountered fatal error, Execution Unit cannot recover
* ``NRT_EXEC_BAD_INPUT`` - Invalid input submitted to exec()
* ``NRT_EXEC_COMPLETED_WITH_NUM_ERR`` - Execution completed with numerical errors (produced NaN)
* ``NRT_EXEC_COMPLETED_WITH_ERR`` - Execution completed with other errors
* ``NRT_EXEC_NC_BUSY`` - Neuron core is locked (in use) by another model/process
* ``NRT_EXEC_OOB`` - One or more indirect memcopies and/or embedding updates are out of bound
* ``NRT_COLL_PENDING`` - Collective operation is still pending
* ``NRT_EXEC_HW_ERR_COLLECTIVES`` - Stuck in collectives op (missing notification(s))
* ``NRT_EXEC_HW_ERR_HBM_UE`` - HBM encountered an unrepairable uncorrectable error
* ``NRT_EXEC_HW_ERR_NC_UE`` - On-chip memory of Neuron Core encountered a parity error
* ``NRT_EXEC_HW_ERR_DMA_ABORT`` - DMA engine encountered an unrecoverable error
* ``NRT_EXEC_SW_NQ_OVERFLOW`` - Software notification queue overflow
* ``NRT_EXEC_HW_ERR_REPAIRABLE_HBM_UE`` - HBM encountered a repairable uncorrectable error
* ``NRT_NETWORK_PROXY_FAILURE`` - EFA network proxy operation failed

**Source**: `nrt_status.h:13 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_status.h#L13>`_

Functions
---------

nrt_get_status_as_str
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   const char *nrt_get_status_as_str(NRT_STATUS status);

Get string representation of a status code.

**Parameters:**

* ``status`` [in] - Status code to convert to string.

**Returns:** String representation of the status code.

**Source**: `nrt_status.h:58 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_status.h#L58>`_
