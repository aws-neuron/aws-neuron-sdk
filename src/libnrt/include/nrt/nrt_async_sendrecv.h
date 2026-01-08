#pragma once

#include "nrt/nrt.h"
#include "nrt/nrt_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nrt_async_sendrecv_comm nrt_async_sendrecv_comm_t;
typedef struct nrt_async_sendrecv_request nrt_async_sendrecv_request_t;

/**
 * Get the maximum number of async sendrecv communicators per logical neuron core
 *
 * @param num[out]   - The maximum number of async sendrecv communicators per logical neuron core
 * @return NRT_SUCCESS on success
 *         NRT_FAILURE for errors
 */
NRT_STATUS nrt_async_sendrecv_get_max_num_communicators_per_lnc(int* num);

/**
 * Get the maximum number of pending requests per async sendrecv communicator
 *
 * @param num[out]   - The maximum number of pending requests per async sendrecv  communicator
 * @return NRT_SUCCESS on success
 *         NRT_FAILURE for errors
 */
NRT_STATUS nrt_async_sendrecv_get_max_num_pending_request(int* num);

/** Initialize asynchronous tensor send and receive on logical neuron core
 *
 * Logical neuron core ID is the absolute ID of the logical core on
 * the host machine. The ID is uneffected by device remapping via
 * docker and selection of visible logical cores.
 *
 * This function may only be called when runtime is initialized. This
 * function must have a matching call to nrt_async_sendrecv_close() before
 * nrt_close() is called.
 * This function returns error in case preceeding call to
 * nrt_async_sendrecv_close() on the logical neuron core returned error.
 *
 * @param lnc[in]   - Logical neuron core ID on the current server
 * @return NRT_SUCCESS if logical core has been initialized successfully
 *         NRT_FAILURE for errors
 */
NRT_STATUS nrt_async_sendrecv_init(int lnc);

/** Closes asynchronous tensor send and receive of logical neuron core and cleans up resources
 *
 * A call to this function must have a preceeding matching call to
 * nrt_async_sendrecv_init().  After this function was invoked, all sendrecv
 * communicators and requests associated with this logical neuron core
 * are closed and cannot be accessed anymore invoking functions with those
 * communicators or requests is regarded undefined behavior.
 * Cases where this function is called and one of the communicators is
 * not connected yet are considered an error. Cases where this
 * function is called and send or receive requests are still inflight
 * are considered an error.
 *
 * @param lnc[in]   - Logical neuron core ID on the current server
 * @return NRT_SUCCESS if logical core has been closed successfully
 *         NRT_FAILURE for errors
 */
NRT_STATUS nrt_async_sendrecv_close(int lnc);

/** Create send communicator
 *
 * Before send communicator can be used to initiate sending a tensor,
 * connection to receive communicator must be established. Use
 * function nrt_async_sendrecv_test_comm() to test whether connection is
 * established.
 * Async sendrecv for logical neuron core lnc must have been
 * initialized via call to nrt_async_sendrecv_init() before this function is
 * invoked.
 * This function is thread-safe.
 *
 * @param peer_ip[in]    - IP adress of peer logical neuron core
 * @param peer_lnc[in]   - Logical neuron core ID on the peer server
 * @param lnc[in]        - Logical neuron core ID on the current server
 * @param send_comm[out] - Pointer to send communicator
 * @return NRT_SUCCESS  if logical core has been created successfully
 *         NRT_RESOURCE if the number of created communicators exceeds the limit of NRT_ASYNC_SENDRECV_MAX_NUM_COMMUNICATORS_PER_LNC
 *         NRT_FAILURE  for other errors
 */
NRT_STATUS nrt_async_sendrecv_connect(const char* peer_ip, int peer_lnc, int lnc, nrt_async_sendrecv_comm_t** send_comm);

/** Create receive communicator
 *
 * Before receive communicator can be used to initiate receiveing a tensor,
 * connection to receive communicator must be established. Use
 * function nrt_async_sendrecv_test_comm() to test whether connection is
 * established.
 * Async sendrecv for logical neuron core lnc must have been
 * initialized via call to nrt_async_sendrecv_init() before this function is
 * invoked.
 * This function is thread-safe.
 *
 * @param peer_ip[in]    - IP adress of peer logical neuron core
 * @param peer_lnc[in]   - Logical neuron core ID on the peer server
 * @param lnc[in]        - Logical neuron core ID on the current server
 * @param recv_comm[out] - Pointer to receive communicator
 * @return NRT_SUCCESS  if logical core has been created successfully
 *         NRT_RESOURCE if the number of created communicators exceeds the limit of NRT_ASYNC_SENDRECV_MAX_NUM_COMMUNICATORS_PER_LNC
 *         NRT_FAILURE  for other errors
 */
NRT_STATUS nrt_async_sendrecv_accept(const char* peer_ip, int peer_lnc, int lnc, nrt_async_sendrecv_comm_t** recv_comm);

/** Test whether connection has been established
 *
 * @param comm[in]  - The send or receive communicator
 * @param done[out] - True if connection to peer communicator is established
 * @return NRT_SUCCESS if test performed without error
 *         NRT_INVALID_HANDLE if handle is invalid
 *         NRT_TIMEOUT        if the communicator fails to establish connection within time limit
 *         NRT_FAILURE        for other errors
 */
NRT_STATUS nrt_async_sendrecv_test_comm(nrt_async_sendrecv_comm_t* comm, bool* done);

/** Asynchronously send a tensor
 *
 * This is a non-blocking function.
 *
 * This function is thread-safe. This function is only allowed to be
 * invoked on a communicator that is sucessfully tested to be
 * connected via call to nrt_async_sendrecv_test_comm().
 *
 * @param tensor[in]        - Tensor to receive to
 * @param offset[in]        - Offset into the tensor to receive to
 * @param length[in]        - Number of bytes to read
 * @param send_comm[in]     - Send communicator
 * @param request[out]      - Pointer to receive request
 * @return NRT_SUCCESS        on success
 *         NRT_INVALID_HANDLE if handle is invalid
 *         NRT_RESOURCE       if the number of pending requests exceeds the limit of NRT_ASYNC_SENDRECV_MAX_NUM_PENDING_REQUEST
 *         NRT_FAILURE        for other errors
 */
NRT_STATUS nrt_async_sendrecv_send_tensor(nrt_tensor_t* tensor, size_t offset, size_t length, nrt_async_sendrecv_comm_t* send_comm, nrt_async_sendrecv_request_t** request);

/** Asynchronously receive a tensor
 *
 * This is a non-blocking function.
 *
 * This function is thread-safe. This function is only allowed to be
 * invoked on a communicator that is sucessfully tested to be
 * connected via call to nrt_async_sendrecv_test_comm().
 *
 * @param tensor[in]        - Tensor to receive to
 * @param offset[in]        - Offset into the tensor to receive to
 * @param length[in]        - Number of bytes to read
 * @param recv_comm[in]     - Receive communicator
 * @param request[out]      - Pointer to receive request
 * @return NRT_SUCCESS        on success
 *         NRT_INVALID_HANDLE if handle is invalid
 *         NRT_RESOURCE       if the number of pending requests exceeds the limit of NRT_ASYNC_SENDRECV_MAX_NUM_PENDING_REQUEST
 *         NRT_FAILURE        for other errors
 */
NRT_STATUS nrt_async_sendrecv_recv_tensor(nrt_tensor_t* tensor, size_t offset, size_t length, nrt_async_sendrecv_comm_t* recv_comm, nrt_async_sendrecv_request_t** request);

/** Test the completion status of a asynchronous request
 *
 * This function is thread-safe when invoked with different
 * requests. This function is not allowed to be invoked concurrently
 * by multiple threads with the same request at the same time. When
 * this function returned request to be completed, this function is
 * not allowed to be invoked again with the same request.
 *
 * @param request[in]       - Request to test
 * @param done[out]         - Whether the request has completed
 * @param size[out]         - Number of bytes sent/received
 * @return NRT_SUCCESS        on success
 *         NRT_INVALID_HANDLE if handle is invalid
 *         NRT_TIMEOUT        if the request fails to complete data transfer within time limit
 *         NRT_FAILURE        for other errors
 */
NRT_STATUS nrt_async_sendrecv_test_request(nrt_async_sendrecv_request_t* request, bool* done, size_t* size);

/** Flush received messae to ensure full arrival in memory
 *
 * Ensure that received messages of successfully tested async sendrecv
 * receive operations prior to call to this function fully arrived in
 * memory after this function completes.
 *
 * @param lnc[in]        - Receiving logical neuron core ID
 * @return NRT_SUCCESS  if flush operation succeeded
 *         NRT_FAILURE  for other errors
 */
NRT_STATUS nrt_async_sendrecv_flush(int lnc);

#ifdef __cplusplus
}
#endif
