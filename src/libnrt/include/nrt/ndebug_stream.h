/*
 * Copyright 2025, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

/**
 * Overview:
 * The `ndebug_stream` APIs provide applications a way to consume debug events from the runtime (see
 * `ndebug_stream_event_type_t` for the different event types). These debug events are emitted by the
 * runtime per Logical Neuron Core and can be used by applications to get information on events that
 * occured on the device (ie prints, breakpoints, etc.).
 *
 * Connecting, polling, and consuming:
 * Applications that want to consume debug events will first need to connect to a Logical Neuron Core's debug stream via a call to
 * `nrt_debug_client_connect`. Once a client is connected to a core's debug stream, the runtime will will push debug events emitted
 * by the Logical Neuron Core to the stream for clients to consume. To be notified of emitted debug events, clients can utilize the
 * polling APIs provided by the Linux kernel. The `stream_fd` handle obtained from the `nrt_debug_client_connect` is a typical Linux
 * file descriptor and can be passed into any Linux polling API. It is important to note though, that while the `stream_fd` is pollable,
 * all other non-polling related functionality must go through the provided `nrt_debug_client*` APIs. For example, the stream contents
 * can only be accessed from the `nrt_debug_client_read*` API(s) and any other methods of accessing the stream data leads undefined/undesireable
 * behavior.
 *
 * Closing a Connection:
 * Once a connection is not needed anymore, clients can close the connection using the `nrt_debug_client_connect_close` API.
 *
 * Events:
 * Events consist of a header describing the payload type, and a payload representing the contents of the event. Events can be consumed by
 * clients via the `nrt_debug_client_read*` API(s).
 *
 * Notes:
 *  * These APIs do not allow for interprocess communication. Debug events are only pushed to the process that owns the Logical Neuron Core.
 *  * These APIs do not provide thread safety for multiple threads accessing the SAME stream (thread safety for different streams is guarenteed).
 *  * There can only be one outstanding connection per stream. Any attempts to initialize multiple connectiongs will result in an error.
 *  * Events are only emitted AFTER a client connects to a Logical Neuron Core's stream. Any event that would have been emitted before connectioning
 *    to the stream is dropped.
 *  * Events will be dropped if the number of unconsumed events in a stream exceeds the stream's buffer size. Clients must consume events fast
 *    enough to prevent dropped events. Additionally, Clients can configure the stream's buffer size via the `NEURON_RT_DEBUG_STREAM_BUFFER_SIZE`
 *    environment variable. The buffer size currently defaults to 64K debug events.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <nrt/nrt_status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ndebug_stream_event_type {
    NDEBUG_STREAM_EVENT_TYPE_INVALID = 0,
    NDEBUG_STREAM_EVENT_TYPE_DEBUG_TENSOR_READ = 1,
} ndebug_stream_event_type_t;

typedef struct ndebug_stream_event_header {
    uint64_t data_size;
    uint32_t type;
    char reserved[52];
} ndebug_stream_event_header_t;

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

/** Establish a connection to a specified Logical Neuron Core's debug stream.
 *
 * @param logical_nc_idx[in]    - Core's debug stream to connect to.
 * @param stream_fd[out]        - Connection handle to reference and interact with the stream.
 *
 * @return NRT_SUCCESS on success.
 *
 * @note Only one client can connect to a Logical Neuron Core's stream at any given time.
 *       Attempts to connect to a stream with multiple clients will result in a NRT_INVALID
 *       return status.
 *
 */
NRT_STATUS nrt_debug_client_connect(int logical_nc_idx, int *stream_fd);

/** Closes connection created by `nrt_debug_client_connect`
 *
 * @param stream_fd[in] - Connection handle to close.
 *
 */
void nrt_debug_client_connect_close(int stream_fd);

/** Consumes a single event from the stream.
 *
 * @param stream_fd[in] - Stream to consume an event from
 * @param header[out]   - Comsuned event's header. See `ndebug_stream_event_header_t`.
 * @param payload[out]  - Consumed event's payload. See `ndebug_stream_payload*` and `ndebug_stream_event_type_t`.
 *                        **IMPORTANT**: it is the user's responsibility to free this payload pointer.
 *
 * @return NRT_SUCCESS on success.
 *
 * @note This function must be called from the same process that owns the Logical Neuron Core. Calling this
 *       function from any other process results in undefined behavior.
 *
 */
NRT_STATUS nrt_debug_client_read_one_event(int stream_fd, ndebug_stream_event_header_t *header, void **payload);

#ifdef __cplusplus
}
#endif
