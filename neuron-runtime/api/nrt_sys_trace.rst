.. _api_nrt_sys_trace_h:

nrt_sys_trace.h
===============

Neuron Runtime System Trace API - Capture and fetch system trace events from Neuron devices.

**Source**: `src/libnrt/include/nrt/nrt_sys_trace.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h>`_

Functions
---------

System Trace Capture
^^^^^^^^^^^^^^^^^^^^

nrt_sys_trace_config_allocate
""""""""""""""""""""""""""""""

.. code-block:: c

   NRT_STATUS nrt_sys_trace_config_allocate(nrt_sys_trace_config_t **options);

Allocate memory for the options structure which is needed to start profiling using nrt_sys_trace_start.

**Parameters:**

* ``options`` [in] - pointer to a pointer to options nrt_sys_trace_config struct

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_sys_trace.h:29 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L29>`_

nrt_sys_trace_config_set_max_events_per_nc
"""""""""""""""""""""""""""""""""""""""""""

.. code-block:: c

   void nrt_sys_trace_config_set_max_events_per_nc(nrt_sys_trace_config_t *options, uint64_t max_events_per_nc);

Sets max number of events that can be stored across all ring buffers.

**Parameters:**

* ``options`` [in,out] - Pointer to the options structure.
* ``max_events_per_nc`` [in] - Max number of events that can be stored in each ring buffer.

**Source**: `nrt_sys_trace.h:50 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L50>`_

nrt_sys_trace_config_set_capture_enabled_for_nc
""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: c

   void nrt_sys_trace_config_set_capture_enabled_for_nc(nrt_sys_trace_config_t *options, uint32_t nc_idx, bool enabled);

Sets system trace capture enabled for a specific NeuronCore. Ring buffers won't be allocated for disabled NeuronCores.

**Parameters:**

* ``options`` [in,out] - Pointer to the options structure.
* ``nc_idx`` [in] - NeuronCore index.
* ``enabled`` [in] - Capture enabled flag.

**Source**: `nrt_sys_trace.h:60 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L60>`_

nrt_sys_trace_get_event_types
""""""""""""""""""""""""""""""

.. code-block:: c

   NRT_STATUS nrt_sys_trace_get_event_types(const char ***event_types, size_t *count);

Returns an allocated array of all valid event type strings.

**Parameters:**

* ``event_types`` [out] - Pointer to array of const char* (allocated).
* ``count`` [out] - Number of event types.

**Returns:** NRT_SUCCESS on success, error code otherwise.

**Note:** The user is responsible for freeing the array and each string, or can use nrt_sys_trace_free_event_types() for convenience.

**Source**: `nrt_sys_trace.h:79 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L79>`_

nrt_sys_trace_start
"""""""""""""""""""

.. code-block:: c

   NRT_STATUS nrt_sys_trace_start(nrt_sys_trace_config_t *options);

Initialization for system trace capture including allocating memory for event ring buffers.

**Parameters:**

* ``options`` [in] - Configuration options for system trace capture

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_sys_trace.h:106 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L106>`_

nrt_sys_trace_stop
""""""""""""""""""

.. code-block:: c

   NRT_STATUS nrt_sys_trace_stop();

Teardown for system trace capture including freeing allocated memory for event ring buffers.

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_sys_trace.h:109 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L109>`_

System Trace Fetch
^^^^^^^^^^^^^^^^^^

nrt_sys_trace_fetch_events
"""""""""""""""""""""""""""

.. code-block:: c

   NRT_STATUS nrt_sys_trace_fetch_events(char **buffer, size_t *written_size, const nrt_sys_trace_fetch_options_t *options);

Fetches system trace events from process memory and returns them as a JSON-formatted string. Once events are fetched, they cannot be fetched again.

**Parameters:**

* ``buffer`` [out] - On successful return, will point to a dynamically allocated, null-terminated JSON string containing the trace events. The caller must free the allocated memory by calling nrt_sys_trace_buffer_free(buffer).
* ``written_size`` [out] - A pointer to a size_t variable that will be set to the number of bytes written into the allocated buffer.
* ``options`` [in] - Pointer to options such as max number of events to fetch.

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_sys_trace.h:143 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L143>`_

nrt_sys_trace_buffer_free
""""""""""""""""""""""""""

.. code-block:: c

   void nrt_sys_trace_buffer_free(char *buffer);

Free the buffer allocated by nrt_sys_trace_fetch_events. Should be called after the events are no longer needed.

**Parameters:**

* ``buffer`` [in] - Pointer to buffer to be freed.

**Source**: `nrt_sys_trace.h:151 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_sys_trace.h#L151>`_
