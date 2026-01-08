/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "nrt/nrt_status.h"
#include "nrt/nrt.h"

#ifdef __cplusplus
extern "C" {
#endif


/** Usage of a Tensor in the NEFF
 */
typedef enum nrt_tensor_usage {
    NRT_TENSOR_USAGE_INPUT = 0,     // Tensor is used for ifmap
    NRT_TENSOR_USAGE_OUTPUT,        // Tensor is used for ofmap
} nrt_tensor_usage_t;

#define NRT_TENSOR_NAME_MAX 256

/** dtypes for tensor data
 */
typedef enum nrt_dtype {
    NRT_DTYPE_UNKNOWN = 0,
    NRT_DTYPE_FLOAT32,
    NRT_DTYPE_FLOAT16,
    NRT_DTYPE_BFLOAT16,
    NRT_DTYPE_INT8,
    NRT_DTYPE_UINT8,
    NRT_DTYPE_INT16,
    NRT_DTYPE_UINT16,
    NRT_DTYPE_INT32,
    NRT_DTYPE_UINT32,
    NRT_DTYPE_INT64,
    NRT_DTYPE_UINT64
} nrt_dtype_t;

typedef struct nrt_tensor_info {
    char name[NRT_TENSOR_NAME_MAX];     // Name of the tensor
    nrt_tensor_usage_t usage;           // Type of the tensor
    size_t size;                        // Tensor size in bytes
    nrt_dtype_t dtype;                  // data type
    uint32_t *shape;                    // an array representing data shape
    uint32_t ndim;                      // the number of dimensions
} nrt_tensor_info_t;

typedef struct nrt_tensor_info_array {
    uint64_t tensor_count;              // Total number of tensors in the NEFF
    nrt_tensor_info_t tensor_array[];   // Array of tensor info
} nrt_tensor_info_array_t;

/* Function definition for async exec status callbacks */
typedef void (*NRT_ASYNC_EXEC_STATUS_CALLBACK)(void *params, uint32_t model_id, uint32_t vnc, uint64_t job_id, NRT_STATUS status);

/** Return input/output tensor information for a given model.
*
* @param model[in]         - Model for which tensor information needs to be extracted.
* @param tensor_info[out]  - Pointer to store the result.
*
* @return NRT_STATUS_SUCCESS on success.
*/
NRT_STATUS nrt_get_model_tensor_info(nrt_model_t *model, nrt_tensor_info_array_t **tensor_info);

/** Return the instance count for this model handle (optimal number of concurrent threads that can call nrt_execute).
*
* @param model[in]         - Model for the instance count needs to be returned.
* @param instance[out]     - Pointer to store the result.
*
* @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_instance_count(nrt_model_t *model, uint32_t *instance_count);


/** Free input/output tensor information for a given model.
*
* @param tensor_info[in]  - Pointer to store the result.
*
* @return NRT_STATUS_SUCCESS on success.
*/
NRT_STATUS nrt_free_model_tensor_info(nrt_tensor_info_array_t *tensor_info);

/** Enable tracing for all VNCs visible to the app
 *
 * @param trace_mem[in] - collect memory allocation info
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_trace_start(bool trace_mem);

/** Serialize all data and disable tracing
 *
 * @param filename[in] - filename to write to
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_trace_stop(const char *filename);

/** temporary, to be removed. See comment in neuron_nccl.cc
*/
void *nrt_get_libnccl_net(int *err, char *err_msg, size_t err_msg_size);

/** Structs to pass around ucode image info
*/
typedef struct nrt_ucode_img {
    uint8_t *bin;
    size_t size;
} nrt_ucode_img;

typedef struct nrt_ucode_info {
    nrt_ucode_img iram;
    nrt_ucode_img dram;
} nrt_ucode_info;

/** Specify pooling engine ucode iram and dram images that will get loaded by nrt_init().
*   To use this API, it MUST be called BEFORE nrt_init().
*   Swapping ucode after nrt_init() is NOT supported. Ucode images are only loaded once.
*   This API provides a temporary workaround for swapping ucode.
*/
NRT_STATUS nrt_set_pool_eng_ucode(const nrt_ucode_info *ucode_info);

/** Copies data to memory mapped Neuron device memory
*
* @param dest[in]          - Pointer to destination memory (mmaped device memory)
* @param src[in]           - Pointer to source memory
* @param size[in]          - Copy size
*
* @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_memcpy_to_device(void *dest, const void *src, size_t size);

/** Register a return status callback to post exec status to when running in async exec mode.
 *  Calling this multiple times will replace the previouly registered callback.
 *
 * @param callback[in]  - Callback to post nrt exec status to for async execution.
 * @param params[in]    - Params for the async exec thread to pass to the callback upon
 *                        execution completion. Can be NULL.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_register_async_exec_callback(NRT_ASYNC_EXEC_STATUS_CALLBACK callback, void *params);

/** Implements a barrier by running a small all-reduce over all workers
*
* @param vnc[in]                 - local VNC (within the instance)
* @param global_device_id[in]    - global worker ID
* @param global_device_count[in] - total number of workers
*
* @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_barrier(int32_t vnc, uint32_t g_device_id, uint32_t g_device_count);

/** Perform all-rank AllGather
*
* @param vnc[in]              - local VNC (within the instance)
* @param g_device_id[in]      - global worker ID
* @param g_device_count[in]   - total number of workers
* @param rank_input_size[in]  - input size
* @param input[in]            - ptr to input data from this rank
* @param output[out]          - ptr to output buffer of size (g_device_count*rank_input_size)
*
* @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_all_gather(int32_t vnc, uint32_t g_device_id, uint32_t g_device_count,
                          uint32_t rank_input_size, void *input, void *output);

/** Blocks caller until all queued executions on async worker thread are drained.
 *
 * @param start_vnc - VNC index to block on.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_async_drain_queued_execs(int32_t start_vnc);

typedef struct nrt_model_info {
    uint32_t vnc;
    // additional fields can be added here in the future
    // do not remove previously added fields because it will cause
    // memory corruption if the caller was compiled using a different 
    // version of this header.
} nrt_model_info_t;
/** Returns information about loaded model
 *
 * @param model [in]          - the model
 * @param info [out]          - the information about the model
 * @param info_size_in [in]   - the size of the info structure (used for version control)
 * @param info_size_out [out] - the number of bytes written (for version control)
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_get_model_info(const nrt_model_t *model, nrt_model_info_t *info, size_t info_size_in, size_t *info_size_out);

#ifdef __cplusplus
}
#endif
