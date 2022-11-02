/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <nrt/nrt_status.h>
#include <nrt/nrt.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nrt_version nrt_version_t;

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

/** Enable tracing for all NCs visible to the app
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
void *nrt_get_libnccl_net(int *err);

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

/** Get the NRT library version
 *
 * @param ver[out]          - Pointer to nrt version struct
 * @param size[in]          - Length of the data needed to be filled in the nrt_version_struct
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_version(nrt_version_t *ver, size_t size);

/** Copies data to memory mapped Neuron device memory
*
* @param dest[in]          - Pointer to destination memory (mmaped device memory)
* @param src[in]           - Pointer to source memory
* @param size[in]          - Copy size
*
* @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_memcpy_to_device(void *dest, const void *src, size_t size);
#ifdef __cplusplus
}
#endif
