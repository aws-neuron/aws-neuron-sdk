.. _api_ndebug_stream_h:

ndebug_stream.h
===============

Neuron Debug Stream API - Consume debug events from the runtime per Logical Neuron Core.

**Source**: `src/libnrt/include/nrt/ndebug_stream.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h>`_

Overview
--------

The ``ndebug_stream`` APIs provide applications a way to consume debug events from the runtime. These debug events are emitted by the runtime per Logical Neuron Core and can be used by applications to get information on events that occurred on the device (ie prints, breakpoints, etc.).

**Connecting, polling, and consuming:** Applications that want to consume debug events will first need to connect to a Logical Neuron Core's debug stream via a call to ``nrt_debug_client_connect``. Once a client is connected to a core's debug stream, the runtime will push debug events emitted by the Logical Neuron Core to the stream for clients to consume.

**Closing a Connection:** Once a connection is not needed anymore, clients can close the connection using the ``nrt_debug_client_connect_close`` API.

**Events:** Events consist of a header describing the payload type, and a payload representing the contents of the event. Events can be consumed by clients via the ``nrt_debug_client_read*`` API(s).

Enumerations
------------

ndebug_stream_event_type_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef enum ndebug_stream_event_type {
       NDEBUG_STREAM_EVENT_TYPE_INVALID = 0,
       NDEBUG_STREAM_EVENT_TYPE_DEBUG_TENSOR_READ = 1,
   } ndebug_stream_event_type_t;

Debug stream event types.

**Source**: `ndebug_stream.h:51 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L51>`_

Structures
----------

ndebug_stream_event_header_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndebug_stream_event_header {
       uint64_t data_size;
       uint32_t type;
       char reserved[52];
   } ndebug_stream_event_header_t;

Debug stream event header.

**Source**: `ndebug_stream.h:56 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L56>`_

ndebug_stream_payload_debug_tensor_read_t
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   typedef struct ndebug_stream_payload_debug_tensor_read {
       char prefix[512];
       uint32_t logical_nc_id;
       uint32_t pipe;
       char tensor_dtype[16];
       uint64_t tensor_shape[8];
       uint64_t tensor_data_size;
       char reserved0[416];
       char tensor_data[];
   } ndebug_stream_payload_debug_tensor_read_t;

Payload for debug tensor read events.

**Source**: `ndebug_stream.h:62 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L62>`_

Functions
---------

nrt_debug_client_connect
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_debug_client_connect(int logical_nc_idx, int *stream_fd);

Establish a connection to a specified Logical Neuron Core's debug stream.

**Parameters:**

* ``logical_nc_idx`` [in] - Core's debug stream to connect to.
* ``stream_fd`` [out] - Connection handle to reference and interact with the stream.

**Returns:** NRT_SUCCESS on success.

**Note:** Only one client can connect to a Logical Neuron Core's stream at any given time. Attempts to connect to a stream with multiple clients will result in a NRT_INVALID return status.

**Source**: `ndebug_stream.h:82 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L82>`_

nrt_debug_client_connect_close
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   void nrt_debug_client_connect_close(int stream_fd);

Closes connection created by ``nrt_debug_client_connect``.

**Parameters:**

* ``stream_fd`` [in] - Connection handle to close.

**Source**: `ndebug_stream.h:88 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L88>`_

nrt_debug_client_read_one_event
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_debug_client_read_one_event(int stream_fd, ndebug_stream_event_header_t *header, void **payload);

Consumes a single event from the stream.

**Parameters:**

* ``stream_fd`` [in] - Stream to consume an event from
* ``header`` [out] - Consumed event's header.
* ``payload`` [out] - Consumed event's payload. **IMPORTANT**: it is the user's responsibility to free this payload pointer.

**Returns:** NRT_SUCCESS on success.

**Note:** This function must be called from the same process that owns the Logical Neuron Core. Calling this function from any other process results in undefined behavior.

**Source**: `ndebug_stream.h:102 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/ndebug_stream.h#L102>`_
