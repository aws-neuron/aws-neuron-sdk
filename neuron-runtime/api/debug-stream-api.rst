.. _nrt-debug-stream-api:

========================================
Neuron Debug Stream API Documentation
========================================

Overview
========

The ``ndebug_stream`` APIs provide applications a way to consume debug events from the runtime. These debug events are emitted by the runtime per Logical Neuron Core and can be used by applications to get information on events that occurred on the device (such as device prints, breakpoints, etc.).

Debug events are streamed through a connection interface, allowing applications to monitor and display information from Neuron Cores during execution.

Connecting, Polling, and Consuming
===================================

Connection Process
------------------

Applications that want to consume debug events must follow these steps:

1. **Connect** to a Logical Neuron Core's debug stream via ``nrt_debug_client_connect``
2. **Poll** for events using Linux kernel polling APIs on the returned file descriptor
3. **Consume** events using the ``nrt_debug_client_read_one_event`` API
4. **Close** the connection when finished using ``nrt_debug_client_connect_close``

Once a client is connected to a core's debug stream, the runtime will push debug events emitted by the Logical Neuron Core to the stream for clients to consume.

Polling for Events
------------------

The stream file descriptor obtained from ``nrt_debug_client_connect`` is a standard Linux file descriptor and can be passed into any Linux polling API (such as ``epoll``, ``poll``, or ``select``). This allows applications to efficiently wait for debug events without busy waiting.

.. important::
   While the ``stream_fd`` is pollable, all other non-polling functionality must go through the provided ``nrt_debug_client*`` APIs. The stream contents can only be accessed using the ``nrt_debug_client_read*`` API(s).

Events
======

Events consist of two parts:

1. A header describing the payload type
2. A payload representing the contents of the event

Each event sent to the application is wrapped as a datagram. The header is a fixed-sized struct that describes the contents of the payload, including the size and how to interpret it.

Event Types
-----------

Currently, the system supports these event types:

+-------------------------------------------------+------------------------------------------+
| Event Type                                      | Description                              |
+=================================================+==========================================+
| ``NDEBUG_STREAM_EVENT_TYPE_DEBUG_TENSOR_READ``  | Debug tensor read events from the core   |
+-------------------------------------------------+------------------------------------------+

API Reference
=============

nrt_debug_client_connect
------------------------

.. code-block:: c

   NRT_STATUS nrt_debug_client_connect(int logical_nc_idx, int *stream_fd);

Establishes a connection to a specified Logical Neuron Core's debug stream.

**Parameters:**

* ``logical_nc_idx [in]`` - Core's debug stream to connect to
* ``stream_fd [out]`` - Connection handle to reference and interact with the stream

**Returns:**

* ``NRT_SUCCESS`` on success

.. note::
   Only one client can connect to a Logical Neuron Core's stream at any given time. Attempts to connect to a stream with multiple clients will result in a ``NRT_INVALID`` return status.

nrt_debug_client_connect_close
------------------------------

.. code-block:: c

   void nrt_debug_client_connect_close(int stream_fd);

Closes a connection created by ``nrt_debug_client_connect``.

**Parameters:**

* ``stream_fd [in]`` - Connection handle to close

nrt_debug_client_read_one_event
-------------------------------

.. code-block:: c

   NRT_STATUS nrt_debug_client_read_one_event(int stream_fd, ndebug_stream_event_header_t *header, void **payload);

Consumes a single event from the stream.

**Parameters:**

* ``stream_fd [in]`` - Stream to consume an event from
* ``header [out]`` - Consumed event's header
* ``payload [out]`` - Consumed event's payload

**Returns:**

* ``NRT_SUCCESS`` on success
* ``NRT_QUEUE_EMPTY`` if no events are available

.. important::
   It is the user's responsibility to free the payload pointer.

.. note::
   This function must be called from the same process that owns the Logical Neuron Core. Calling this function from any other process results in undefined behavior.

Data Structures
===============

ndebug_stream_event_type
------------------------

.. code-block:: c

   typedef enum ndebug_stream_event_type {
       NDEBUG_STREAM_EVENT_TYPE_INVALID = 0,
       NDEBUG_STREAM_EVENT_TYPE_DEBUG_TENSOR_READ = 1,
   } ndebug_stream_event_type_t;

Enumeration of the different types of debug events that can be emitted.

ndebug_stream_event_header
--------------------------

.. code-block:: c

   typedef struct ndebug_stream_event_header {
       uint64_t data_size;
       uint32_t type;
       char reserved[52];
   } ndebug_stream_event_header_t;

Header structure for debug stream events.

**Fields:**

* ``data_size`` - Size of the payload data in bytes
* ``type`` - Type of event (see ``ndebug_stream_event_type_t``)
* ``reserved`` - Reserved bytes for future use

ndebug_stream_payload_debug_tensor_read
---------------------------------------

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

Payload structure for debug tensor read events.

**Fields:**

* ``prefix`` - The prefix string to print
* ``logical_nc_id`` - The logical core the print event originated from
* ``pipe`` - The pipe to write the printed string to
* ``tensor_dtype`` - Tensor data type
* ``tensor_shape`` - Tensor shape dimensions (up to 8 dimensions)
* ``tensor_data_size`` - Size in bytes of the tensor content
* ``reserved0`` - Reserved bytes for future use
* ``tensor_data`` - The contents of the tensor to display (flexible array member)

Notes and Important Considerations
==================================

1. These APIs do not allow for interprocess communication. Debug events are only pushed to the process that owns the Logical Neuron Core.

2. These APIs do not provide thread safety for multiple threads accessing the SAME stream (thread safety for different streams is guaranteed).

3. There can only be one outstanding connection per stream. Any attempts to initialize multiple connections will result in an error.

4. Events are only emitted AFTER a client connects to a Logical Neuron Core's stream. Any event that would have been emitted before connecting to the stream is dropped.

5. Events will be dropped if the number of unconsumed events in a stream exceeds the stream's buffer size. Clients must consume events fast enough to prevent dropped events.

6. Clients can configure the stream's buffer size via the ``NEURON_RT_DEBUG_STREAM_BUFFER_SIZE`` environment variable. The buffer size currently defaults to 64K debug events.

7. The payload buffer returned by ``nrt_debug_client_read_one_event`` must be freed by the caller.
