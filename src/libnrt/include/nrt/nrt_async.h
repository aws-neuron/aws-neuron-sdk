/*
 * Copyright 2025, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
// Use quoted includes in nrt headers including other nrt headers. Most clients
// (ptxla, jax, etc.) build with bazel, and bazel has issue with angle-brackets.
// See https://bazel.build/docs/bazel-and-cpp#include-paths for details.
#include "nrt/nrt.h"

#ifdef __cplusplus
extern "C" {
#endif

// execution units
typedef enum {
  NRTA_XU_TENSOR_READ = 0,
  NRTA_XU_TENSOR_WRITE,
  NRTA_XU_TENSOR_OP, // For tensor ops other than read and write
  NRTA_XU_COMPUTE,
  NRTA_XU_COLLECTIVES,

  // For new XU types, must only add after existing ones
  NRTA_XU_TYPE_NUM
} nrta_xu_t;


// nrta_seq_t's are monotomically increasing ids of executions
// The first 16 bits are a Execution Unit ID, while the last
// 48 bits are a strictly ordered Sequence Number
typedef uint64_t nrta_seq_t;
typedef uint16_t nrta_xu_id_t;

#define NRTA_SEQ_NUM_MAX      ((1ull << 48) - 1)
#define NRTA_SEQ_NUM_MASK     NRTA_SEQ_NUM_MAX
#define NRTA_SEQ_GET_SEQ_NUM(seq_id)  (seq_id & NRTA_SEQ_NUM_MASK)
#define NRTA_SEQ_GET_XU_ID(seq_id)    (seq_id >> 48)


typedef struct nrta_error {
    nrta_seq_t seq_id;
    uint64_t error_code; // NRT_STATUS, but typed as uint64 to ensure consistent representation across compilers
} nrta_error_t;
static_assert(sizeof(nrta_error_t) == 16, "nrta_error_t must be of size 16");

// data structure used to store errors encountered during execution
typedef struct nrta_error_tracker nrta_error_tracker_t;

/** Enqueues a tensor write request.  Copies the data from a host buffer to a
 *  tensor allocated on a Neuron device.  Uses TENSOR_WRITE execution unit based
 *  on the LNC that allocated the tensor.
 *
 * @param tensor[in]          - Destination tensor
 * @param buf[in]             - Host buffer containing source data
 * @param offset[in]          - Offset into the tensor
 * @param size[in]            - Number of bytes to write
 * @param queue[in]           - XU queue to use,
 * @param err[in]             - error tracker
 * @param req_sequence[out]   - Sequence number of the scheduled request
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_tensor_write(nrt_tensor_t *tensor,
                             const void *buf,
                             uint64_t offset,
                             uint64_t size,
                             int queue,
                             nrta_error_tracker_t *err,
                             nrta_seq_t *req_sequence);

/** Enqueues a tensor read request.  Copies the data from a tensor allocated on a Neuron device
 *  to a host buffer. Uses TENSOR_READ execution unit based
 *  on the LNC that allocated the tensor.
 *
 * @param buf[in]             - Destination Host buffer
 * @param tensor[in]          - Source tensor
 * @param offset[in]          - Offset into the tensor
 * @param size[in]            - Number of bytes to read
 * @param queue[in]           - XU queue to use,
 * @param err[in]             - error tracker
 * @param req_sequence[out]   - Sequence number of the scheduled request
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_tensor_read(void *buf,
                            nrt_tensor_t *tensor,
                            uint64_t offset,
                            uint64_t size,
                            int queue,
                            nrta_error_tracker_t *err,
                            nrta_seq_t *req_sequence);

/** Enqueues a tensor copy request.  Copies data between two tensors allocated
 *  on the same Logical Neuron Core.  Uses TENSOR_OP execution unit.
 *
 * NOTE: the tensors must be allocated until the copy completes
 *
 * @param src[in]             - Source tensor
 * @param src_offset[in]      - Offset into the source tensor
 * @param dst[in]             - Destination tensor
 * @param dst_offset[in]      - Offset into the destination tensor
 * @param size[in]            - Number of bytes to copy
 * @param queue[in]           - XU queue to use
 * @param err[in]             - error tracker
 * @param req_sequence[out]   - Sequence number of the scheduled request
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_tensor_copy(nrt_tensor_t *src,
                            uint64_t src_offset,
                            nrt_tensor_t *dst,
                            uint64_t dst_offset,
                            uint64_t size,
                            int queue,
                            nrta_error_tracker_t *err,
                            nrta_seq_t *req_sequence);

/** Schedules an asynchronous request to execute a model with specified inputs
 *  and outputs. Uses COMPUTE execution unit of an LNC of the loaded model.
 *
 * @param model[in]           - The model to schedule for execution
 * @param input_set[in]       - Set of input tensors for the model
 * @param output_set[in]      - Set of tensors to receive the outputs
 * @param queue[in]           - XU queue to use, must be 0
 * @param err[in]             - error tracker
 * @param req_sequence[out]   - Sequence number of the scheduled request
 *
 * @return NRT_SUCCESS on successful preparation, appropriate error code otherwise
 */
NRT_STATUS nrta_execute_schedule(nrt_model_t *model,
                                 const nrt_tensor_set_t *input,
                                 nrt_tensor_set_t *output,
                                 int queue,
                                 nrta_error_tracker_t *err,
                                 nrta_seq_t *req_sequence);

/** Prepares collective context and HW configuration needed for collectives operation.
 *  Allocates a collective context handle that is returned to the caller
 *  which is freed in the schedule thread post CC op execution.
 *
 * @param comm[in]              - Communicator containing the replica group
 * @param input[in]             - Input tensor list
 * @param output[out]           - Output tensor list
 * @param dtype[in]             - Data type of elements
 * @param op[in]                - Reduction operation (e.g., SUM, MAX) if applicable
 * @param cc_op[in]             - Collective operation (e.g., ALLREDUCE, ALLGATHER)
 * @param cc_ctx[out]           - Collective context
 *
 * @return NRT_SUCCESS on successful preparation, appropriate error code otherwise
 */
NRT_STATUS nrta_cc_prepare(nrt_cc_comm_t *comm,
                           nrt_tensor_list_t *input,
                           nrt_tensor_list_t *output,
                           nrt_dtype_t dtype,
                           nrt_op_type_t op,
                           nrt_cc_op_type_t cc_op,
                           nrt_cc_context_t **cc_ctx);

/** Schedules an asynchronous request to execute collective operation
 *
 * @param cc_ctx[in]           - Collective context
 * @param queue[in]            - XU queue to use, must be 0
 * @param err[in]              - error tracker
 * @param req_sequence[out]    - Sequence number of the scheduled request
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrta_cc_schedule(nrt_cc_context_t **cc_ctx,
                            int queue,
                            nrta_error_tracker_t *err,
                            nrta_seq_t *req_sequence);

// completion status

/** Checks completion status of a scheduled request
 *
 * @param seq[in]           - Scheduled request sequence id
 * @param is_completed[out] - true if the request is completed, false otherwise
 *
 * @return NRT_SUCCESS if the request is completed, NRT_INVALID if the seq is not valid
 */
NRT_STATUS nrta_is_completed(nrta_seq_t seq, bool *is_completed);


/** Returns sequence number of the last completed request
 *
 * @param lnc[in]           - LNC
 * @param xu[in]            - XU
 * @param queue[in]         - XU's queue
 * @param seq[out]          - last completed sequence number
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_get_sequence(uint32_t lnc, nrta_xu_t xu, int queue, nrta_seq_t *seq);


/** Returns a pollable file descriptor that is READABLE when the execution request
 * specified by seq is complete.
 *
 * Note that users should only use the `poll` family of functions and `close` on this file
 * descriptor. Any other FD function is invalid and can lead to undefined behavior.
 *
 * The file descriptor must be passed to `close` to free the handle once the handle is not
 * needed anymore.
 *
 * @param seq[in]           - sequence to track completion
 * @param fd[out]           - FD associate with the sequence.
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_get_completion_handle(nrta_seq_t seq, int *fd);


/** Creates an error tracker list
 *
 * @param lnc_idx[in]           - Logical Neuron Core this list will be used for
 * @param error_tracker[out]    - Created list.
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_error_tracker_create(uint32_t lnc_idx, nrta_error_tracker_t **error_tracker);

/** Frees an error tracker list
 *
 * @param error_tracker[in] - Error tracker list to free
 *
 */
void nrta_error_tracker_destroy(nrta_error_tracker_t *error_tracker);

/** Gets list of errors from error tracker list
 *
 * @param error_tracker[in] - Error tracker list to get errors from
 * @param list[out]         - Array of errors obtained from teh error tracker
 * @param error_count[out]  - Number of errors in the list
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrta_error_tracker_get_list(nrta_error_tracker_t *error_tracker, const nrta_error_t **list, size_t *error_count);

#ifdef __cplusplus
}
#endif
