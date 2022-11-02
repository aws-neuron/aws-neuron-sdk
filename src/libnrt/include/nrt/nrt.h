/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <nrt/nrt_status.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Major and minor version of runtime. */
#define NRT_MAJOR_VERSION 2
#define NRT_MINOR_VERSION 0

typedef struct nrt_model nrt_model_t;

typedef struct nrt_tensor nrt_tensor_t;

typedef enum {
    NRT_TENSOR_PLACEMENT_DEVICE,
    NRT_TENSOR_PLACEMENT_HOST,
    NRT_TENSOR_PLACEMENT_VIRTUAL,
} nrt_tensor_placement_t;

typedef enum {
    NRT_FRAMEWORK_TYPE_INVALID = 0,             // Invalid
    NRT_FRAMEWORK_TYPE_NO_FW = 1,               // Framework less execution
    NRT_FRAMEWORK_TYPE_TENSORFLOW,              // Tensorflow
    NRT_FRAMEWORK_TYPE_PYTORCH,                 // Pytorch
    NRT_FRAMEWORK_TYPE_MXNET,                   // Mxnet
} nrt_framework_type_t;

/** Initialize neuron runtime.
 *
 * @param framework[in]      - Type of the framework.
 * @param fw_version[in]     - Framework version as string. (eg 2.1)
 * @param fal_version[in]    - Framework Abstraction Layer version as string.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_init(nrt_framework_type_t framework, const char *fw_version, const char *fal_version);

/** Closes all the devices and cleans up the runtime state.
 */
void nrt_close();

/** Load given NEFF and place it in one or more neuron cores.
 *
 * @param neff_bytes[in]    - Pointer to NEFF data.
 * @param size[in]          - Length of the NEFF data.
 * @param start_nc[in]      - Starting NC index where the NEFF should be loaded(-1 means runtime would automatically load in first free NC).
 * @param nc_count[in]      - Number of NCs to use(-1 means runtime would automatically determine the need).
 * @param model[out]        - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load(const void *neff_bytes, size_t size, int32_t start_nc, int32_t nc_count, nrt_model_t **model);

/** Load given NEFF for collective operations and place it in one or more neuron cores.
 *
 * Global NCCL communicator is created inside the API according to g_device_id and g_device_count.
 *
 * @param neff_bytes[in]    - Pointer to NEFF data.
 * @param size[in]          - Length of the NEFF data.
 * @param start_nc[in]      - Starting NC index where the NEFF should be loaded(-1 means runtime would automatically load in first free NC).
 * @param nc_count[in]      - Number of NCs to use(-1 means runtime would automatically determine the need).
 * @param g_device_id[in]   - Global device ID participating collective operations
 * @param g_device_count[in]- Number of devices participating collective operations
 * @param model[out]        - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load_collectives(const void *neff_bytes, size_t size, int32_t start_nc, int32_t nc_count,
                                uint32_t g_device_id, uint32_t g_device_count, nrt_model_t **model);

/** Unload given model and free up device and host resources.
 *
 * @param model - Model to unload.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_unload(nrt_model_t *model);

/** Get the number of NCs used by a loaded model
 *
 * @param model[in] - Model.
 * @param nc_count[out] - The number of NCs used by the model.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_nc_count(const nrt_model_t *model, uint32_t *nc_count);

/** Returns NeuronCores available in instance.
 *
 * @param nc_count[out] - NeuronCores available in instance.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_total_nc_count(uint32_t *nc_count);

/** Returns NeuronCores visible to the application.
 *
 * @param nc_count[out] - NeuronCores visible to the application.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_visible_nc_count(uint32_t *nc_count);

/** A container to hold multiple tensors */
typedef void nrt_tensor_set_t;

/** Allocates a new tensor set.
 *
 * @param result[out]       - Pointer to newly allocated tensor set would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_allocate_tensor_set(nrt_tensor_set_t **result);

/** Destroys given tensor_set and frees memory.
 *
 * @param tensor_set[in]    - Tensors set to be freed.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
void nrt_destroy_tensor_set(nrt_tensor_set_t **tensor_set);

/** Add/replace given tensor to tensor set
 *
 * @param tensor_set[in]    - Tensor set to which the tensor is added.
 * @param tensor_name[in]   - Name of the tensor.
 * @param tensor[in]        - Pointer to tensor. This pointer should be valid till nrt_destroy_tensor_set() is called.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_add_tensor_to_tensor_set(nrt_tensor_set_t *tensor_set, const char *tensor_name, nrt_tensor_t *tensor);

/** Get a tensor's info from a tensor set.
 *
 * @param tensor_set[in]    - Tensor set.
 * @param tensor_name[in]   - Name of the tensor.
 * @param tensor[out]       - Pointer to tensor would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_tensor_from_tensor_set(nrt_tensor_set_t *tensor_set, const char *tensor_name, nrt_tensor_t **tensor);

/** Execute given model with given inputs and collect outputs.
 *
 * @param model[in] - Model to execute.
 * @param input_set[in] - Set of input tensors.
 * @param output_set[in] - Set of output tensors.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_execute(nrt_model_t *model, const nrt_tensor_set_t *input_set, nrt_tensor_set_t *output_set);

/** Execute given model with given inputs, repeat execution specified number of times and collect outputs.
 *
 * @param model[in] - Model to execute.
 * @param input_set[in] - Set of input tensors.
 * @param output_set[in] - Set of output tensors.
 * @param repeat_count[in] - Number of to repeat execution.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_execute_repeat(nrt_model_t *model, const nrt_tensor_set_t *input_set, nrt_tensor_set_t *output_set, int repeat_count);

/** Allocates a tensor that can be passed and used by a model for compute.
 *
 * @param tensor_placement[in]  - Where the tensor would be allocated (device, host, or virtual memory)
 * @param logical_nc_id[in]     - Logical Neuron Core id to allocate the tensor on. Pass in NRT_HOST_NEURON_CORE_ID if allocating tensors on host memory.
 * @param size[in]              - Size in bytes of the tensor to allocate.
 * @param name[in]              - OPTIONAL. Name of the tensor.
 * @param tensor[out]           - Pointer to newly created tensor will be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t tensor_placement, int logical_nc_id, size_t size, const char *name, nrt_tensor_t **tensor);

/** Deallocates a tensor created by "nrt_tensor_allocate".
 *
 * @param tensor[in]    - Deallocates given tensor.
 *
 * @return None
 */
void nrt_tensor_free(nrt_tensor_t **tensor);

/** Copies data from tensor to passed in buffer.
 *
 * @param tensor[in]    - Tensor used to reference the tensor to read from.
 * @param buf[out]      - Buffer used to store data read from the tensor.
 * @param offset[in]    - Offset into the tensor to read from.
 * @param size[in]      - Number of bytes to read.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_read(const nrt_tensor_t *tensor, void *buf, size_t offset, size_t size);

/** Copies data from passed in buffer to tensor.
 *
 * @param tensor[in/out]    - Tensor used to reference the tensor to write to.
 * @param buf[in]           - Buffer used to store data to write to the tensor.
 * @param offset[in]        - Offset into the tensor to write to.
 * @param size[in]          - Number of bytes to write.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_write(nrt_tensor_t *tensor, const void *buf, size_t offset, size_t size);


/** Copies data between tensors.
 *
 * When copying between two device tensors, they must both be allocated on the SAME Neuron Core.
 * A NRT_INVALID will be returned in the failing case.
 *
 * @param src[in]           - Tensor to copy from.
 * @param src_offset[in]    - Offset into the source tensor to copy from.
 * @param dst[out]          - Tensor to copy to.
 * @param dst_offset[in]    - Offset into the destination tensor to copy to.
 * @param size[in]          - Number of bytes to copy.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_copy(const nrt_tensor_t *src, size_t src_offset, nrt_tensor_t *dst, size_t dst_offset, size_t size);

/** Gets the size of the passed in tensor.
 *
 * @param tensor[in]    - Tensor used to reference the tensor to get size of.
 *
 * @return Size of the tensor.
 */
size_t nrt_tensor_get_size(const nrt_tensor_t *tensor);

/** Set the memory + offset pointed to by tensor to value
 *
 * @param tensor[in]        - allocated tensor
 * @param offset[in]        - offset within the tensor
 * @param value[in]         - value to set with
 * @param size[in]          - size of memory to set
 *
 * @return 0 on success.
 */
NRT_STATUS nrt_tensor_memset(nrt_tensor_t *tensor, uint64_t offset, int value, size_t size);

/** Allocates an empty tensor, i.e. the tensor structure w/o any attached storage
 *
 * @param name[in]              - OPTIONAL. Name of the tensor.
 * @param tensor[out]           - Pointer to newly created tensor will be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_allocate_empty(const char *name, nrt_tensor_t **tensor);

/** Attaches caller supplied buffer to a tensor.  Any storage previously attached to the tensor is detached
 *  and freed if was owned by the tensor.
 *  The buffer is supplied by the caller and must persist through the entire lifetime of the tensor.
 *
 * @param tensor[in]            - Tensor
 * @param buffer[in]            - Caller supplied buffer to use as tensor's storage
 * @param size[in]              - Buffer Size
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_attach_buffer(nrt_tensor_t *tensor, void *buffer, size_t size);

/** Creates a tensor to point to a slice of another tensor
 *  does not do a deep copy, just points the "slice" tensor storage to the "source" tensor storage
 *
 * @param tensor_source[in] - Tensor to point at
 * @param offset[in]        - Offset from the beginning of the source tensor to point at
 * @param size[in]          - Size of the slice
 * @param name[in]          - Optional name for the new tensor
 * @param tensor_slice[in]  - Newly allocated tensor to point to the storage of the source tensor
 *
 */
NRT_STATUS nrt_tensor_allocate_slice( const nrt_tensor_t *tensor_source, size_t offset, size_t size, const char *name, nrt_tensor_t **tensor_slice);

/** Given a tensor get the virtual address.
 *
 * @param tensor[in]        - Tensor for which the VA needs to be obtained
 *
 * @return va on success, NULL on failure.
 */
void *nrt_tensor_get_va(const nrt_tensor_t *tensor);

#ifdef __cplusplus
}
#endif
