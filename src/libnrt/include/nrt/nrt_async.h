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
  NRTA_XU_TENSOR_READ,
  NRTA_XU_TENSOR_WRITE,
  NRTA_XU_COMPUTE,
  NRTA_XU_COLLECTIVES,
  NRTA_XU_SEND_RECV // unused
} nrta_xu_t;

typedef uint64_t nrta_seq_t;     // execution sequence
#define NRTA_SEQ_LNC(seq) ((seq)>>56) // seq upper 8b encode LNC
#define NRTA_SEQ_XU(seq)  (nrta_xu_t)(((seq)>>48)&0xFF) // seq encodes XU

// data structure used to store errors encuntered during execution
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
 * @param dep_sequences[in]   - the list of dependencies to wait for, could be NULL if no dependencies
 * @param dep_count[in]       - the number of dependencies in the list, could be 0
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
                             nrta_seq_t *dep_sequences,
                             uint32_t dep_count,
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
 * @param dep_sequences[in]   - the list of dependencies to wait for, could be NULL if no dependencies
 * @param dep_count[in]       - the number of dependencies in the list, could be 0
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
                            nrta_seq_t *dep_sequences,
                            uint32_t dep_count,
                            nrta_error_tracker_t *err,
                            nrta_seq_t *req_sequence);
 
/** Schedules an asynchronous request to execute a model with specified inputs
 *  and outputs. Specifies dependencies that must be satisfied before the model execution can start.
 *  Uses COMPUTE execution unit of an LNC of the loaded model.
 *
 * @param model[in]           - The model to schedule for execution
 * @param input_set[in]       - Set of input tensors for the model
 * @param output_set[in]      - Set of tensors to receive the outputs
 * @param queue[in]           - XU queue to use, must be 0
 * @param dep_sequences[in]   - the list of dependencies to wait for, could be NULL if no dependencies
 * @param dep_count[in]       - the number of dependencies in the list, could be 0
 * @param err[in]             - error tracker
 * @param req_sequence[out]   - Sequence number of the scheduled request
 *
 * @return NRT_SUCCESS on successful preparation, appropriate error code otherwise
*/
NRT_STATUS nrta_execute_schedule(nrt_model_t *model,
                                 const nrt_tensor_set_t *input,
                                 nrt_tensor_set_t *output,
                                 int queue,
                                 nrta_seq_t *dep_sequences,
                                 uint32_t dep_sequence_count,
                                 nrta_error_tracker_t *err,
                                 nrta_seq_t *req_sequence);

#if 0 // TODO skip for now, schedule above should auto start as soon as all deps are satisfied 
/** Starts executioin of a previously scheduled model.
*
* @param xu[in]  - XU and queue identifier returned by nrta_get_xuq()
* @return NRT_SUCCESS on success
*/
NRT_STATUS nrta_start_next(nrta_xuq_t xuq);
#endif

// completion status

/** Checks completion status of a scheduled request
 *
 * @param seq[in]           - Scheduled request sequence number
 *
 * @return NRT_SUCCESS if the request is completed, NRT_BUSY if the request is not completed
*/
NRT_STATUS nrta_is_completed(nrta_seq_t seq_number);


/** Returns sequence number of the last completed request
 *
 * @param lnc[in]           - LNC
 * @param xu[in]            - XU
 * @param queue[in]         - XU's queue
 * @param seq[out]          - last completed sequence number
 *
 * @return NRT_SUCCESS on success
*/
NRT_STATUS nrta_get_sequence(uint32_t  lnc, nrta_xu_t xu, int queue, nrta_seq_t *seq);


/** Returns event FD that can be used to wait for completion of the specified sequence
 *
 * @param seq[in]           - sequence to associate event FD with
 * @param event_fd[out]     - event FD associate with the sequence.
 * @return NRT_SUCCESS on success
*/

NRT_STATUS nrta_mark(nrta_seq_t seq_num, int *event_fd);
#ifdef __cplusplus
}
#endif
