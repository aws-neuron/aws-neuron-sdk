/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
// Use quoted includes in nrt headers including other nrt headers. Most clients
// (ptxla, jax, etc.) build with bazel, and bazel has issue with angle-brackets.
// See https://bazel.build/docs/bazel-and-cpp#include-paths for details.
#include "nrt/nrt_status.h"
#include "ndl/neuron_driver_shared_tensor_batch_op.h"


#ifdef __cplusplus
extern "C" {
#endif

/** Major and minor version of runtime. */
#define NRT_MAJOR_VERSION 2
#define NRT_MINOR_VERSION 0

typedef struct nrt_model nrt_model_t;

typedef struct nrt_tensor nrt_tensor_t;

typedef struct nrt_cc_context nrt_cc_context_t;

/**
 * WARNING: Do not change the value of existing enums!
 * These values will be used by libnrt consumers, we
 * cannot change the defines under them, only append.
 */
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
    NRT_FRAMEWORK_TYPE_PRECHECK,                // Neuron Node Precheck
} nrt_framework_type_t;

enum {
    NRT_INSTANCE_UNKNOWN    = 0,
    NRT_INSTANCE_INF1       = 1,
    NRT_INSTANCE_TRN1       = 2,
    NRT_INSTANCE_TRN1N      = 3,
    NRT_INSTANCE_INF2       = 4,
    NRT_INSTANCE_TRN2       = 5,
    NRT_INSTANCE_TRN2N      = 6,
    NRT_INSTANCE_INF2E      = 7,
    NRT_INSTANCE_TRN2P      = 8,
    NRT_INSTANCE_TRN2U      = 9,
    NRT_INSTANCE_TRN2E      = 10,
    NRT_INSTANCE_TRN2EU     = 11,
    NRT_INSTANCE_TRN2AC     = 12,
    NRT_INSTANCE_TRN2UAC    = 13,
    NRT_INSTANCE_TRN3       = 14,
    NRT_INSTANCE_TRN3PDS98  = 15
};

enum {
    NRT_INSTANCE_SIZE_1XL,
    NRT_INSTANCE_SIZE_2XL,
    NRT_INSTANCE_SIZE_4XL,
    NRT_INSTANCE_SIZE_6XL,
    NRT_INSTANCE_SIZE_8XL,
    NRT_INSTANCE_SIZE_24XL,
    NRT_INSTANCE_SIZE_32XL,
    NRT_INSTANCE_SIZE_48XL,
    NRT_INSTANCE_SIZE_3XL,
    // Note: Add new sizes right above this line to prevent breaking backward compatibility

    NRT_INSTANCE_SIZE_UNKNOWN,
    NRT_INSTANCE_SIZE_NUM = NRT_INSTANCE_SIZE_UNKNOWN,
};

typedef enum nrt_op_type {
    NRT_OP_ADD     = 0x0,
    NRT_OP_FMA     = 0x1,
    NRT_OP_MAX     = 0x2,
    NRT_OP_MIN     = 0x3,
    NRT_OP_INVALID = 0xF,
} nrt_op_type_t;

typedef enum nrt_dtype {
    NRT_DTYPE_UNKNOWN  = 0x0,
    NRT_DTYPE_INVALID  = 0x0,
    NRT_DTYPE_FP8_E3   = 0xD,
    NRT_DTYPE_FP8_E4   = 0xE,
    NRT_DTYPE_FP8_E5   = 0xF,
    NRT_DTYPE_FLOAT16  = 0x7,
    NRT_DTYPE_BFLOAT16 = 0x6,
    NRT_DTYPE_FLOAT32  = 0xA,
    NRT_DTYPE_FP32R    = 0xB,
    NRT_DTYPE_UINT8    = 0x3,
    NRT_DTYPE_UINT16   = 0x5,
    NRT_DTYPE_UINT32   = 0x9,
    NRT_DTYPE_UINT64   = 0x1,
    NRT_DTYPE_INT8     = 0x2,
    NRT_DTYPE_INT16    = 0x4,
    NRT_DTYPE_INT32    = 0x8,
    NRT_DTYPE_INT64    = 0xC,
} nrt_dtype_t;

typedef enum nrt_cc_op_type {
    NRT_CC_ALLGATHER,
    NRT_CC_ALLREDUCE,
    NRT_CC_REDUCESCATTER
} nrt_cc_op_type_t;

typedef struct nrt_instance_info {
    uint32_t family;
    uint32_t size;
    char arch_name[16];
    char device_revision[8];
} nrt_instance_info_t;

NRT_STATUS nrt_get_instance_info(nrt_instance_info_t *info, size_t instance_info_len);

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
 * @param vnc[in]           - VNC index where the NEFF should be loaded(-1 means runtime would automatically load in first free VNC).
 * @param vnc_count[in]     - DEPRECATED: always use -1
 * @param model[out]        - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load(const void *neff_bytes, size_t size, int32_t vnc, int32_t vnc_count, nrt_model_t **model);

/** Load given NEFF for collective operations and place it in one or more neuron cores.
 *
 * If global NCCL communicator was not previously created, we will create it inside this API with the assumption that
 * global device id is same as ctx_device_id and global device count is same as ctx_device_count.
 *
 * @param neff_bytes[in]        - Pointer to NEFF data.
 * @param size[in]              - Length of the NEFF data.
 * @param vnc[in]               - VNC index where the NEFF should be loaded(-1 means runtime would automatically load in first free VNC).
 * @param vnc_count[in]         - DEPRECATED: always use -1
 * @param ctx_device_id[in]     - Device ID relative to the number of devices participating in this NEFF
 * @param ctx_device_count[in]  - Number of devices participating in collectives operations in this NEFF
 * @param model[out]            - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load_collectives(const void *neff_bytes, size_t size, int32_t vnc, int32_t vnc_count,
                                uint32_t ctx_device_id, uint32_t ctx_device_count, nrt_model_t **model);

/** Unload given model and free up device and host resources.
 *
 * @param model - Model to unload.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_unload(nrt_model_t *model);

/** Get the number of VNCs used by a loaded model. (deprecated)
 *
 * @param model[in] - Model.
 * @param vnc_count[out] - The number of VNCs used by the model.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_nc_count(const nrt_model_t *model, uint32_t *vnc_count);

/** Get the number of VNCs used by a loaded model. (deprecated)
 *
 * @param model[in] - Model.
 * @param vnc_count[out] - The number of VNCs used by the model.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_vnc_count(const nrt_model_t *model, uint32_t *vnc_count);

/** Returns VirtualNeuronCores available in instance. (deprecated)
 *
 * @param vnc_count[out] - VirtualNeuronCores available in instance.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_total_nc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores available in instance.
 *
 * @param vnc_count[out] - VirtualNeuronCores available in instance.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_total_vnc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores visible to the application. (deprecated)
 *
 * @param vnc_count[out] - VirtualNeuronCores visible to the application.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_visible_nc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores visible to the application.
 *
 * @param vnc_count[out] - VirtualNeuronCores visible to the application.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_visible_vnc_count(uint32_t *vnc_count);

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

/** Build (initialize and setup) NCCL global communicator.
 *
 * @param vnc[in]               - Local VNC (within the instance)
 * @param g_device_id[in]       - Global device id
 * @param g_device_count[in]    - Max world size of all neffs that will be executed
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_build_global_comm(int32_t vnc, uint32_t g_device_id, uint32_t g_device_count);

/** Allocates a tensor that can be passed and used by a model for compute.
 *
 * @param tensor_placement[in]  - Where the tensor would be allocated (device, host, or virtual memory)
 * @param vnc[in]               - Virutal Neuron Core id to allocate the tensor on. Pass in -1 if allocating tensors on host memory.
 * @param size[in]              - Size in bytes of the tensor to allocate.
 * @param name[in]              - OPTIONAL. Name of the tensor.
 * @param tensor[out]           - Pointer to newly created tensor will be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t tensor_placement, int vnc, size_t size, const char *name, nrt_tensor_t **tensor);

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

/** A batch of tensor operations on a single tensor */
// the definition of nrt_tensor_batch_op_t is in neuron_driver_shared_tensor_batch_op.h
typedef struct nrt_tensor_batch {
    const nrt_tensor_t *tensor;        // Tensor handle
    const nrt_tensor_batch_op_t *ops;  // Array of operations for this tensor
    uint32_t num_ops;            // Number of operations for this tensor
} nrt_tensor_batch_t;

/** Batch read data from multiple tensors.
 *
 * @param batches[in]     - An array of batches, each of which describes operations on one tensor
 * @param num_batches[in] - Number of batches (tensors) in the array
 * @param unsafe[in]      - If true, skip tensor tracking/blocking (use with caution)
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_read_batch(const nrt_tensor_batch_t *batches, uint64_t num_batches, bool unsafe);

/** Batch write data to multiple tensors.
 *
 * @param batches[in]     - An array of batches, each of which describes operations on one tensor
 * @param num_batches[in] - Number of batches (tensors) in the array
 * @param unsafe[in]      - If true, skip tensor tracking/blocking (use with caution)
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_write_batch(const nrt_tensor_batch_t *batches, uint64_t num_batches, bool unsafe);

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

/** Returns on device allocation info for a tensor
 *
 * @param tensor[in]        - Tensor for which the information needs to be obtained
 * @param alloc_info[out]   - On device allocation information
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
typedef struct nrt_tensor_device_allocation_info {
    uint64_t physical_address; // physical address in device memory space
    size_t size;               // allocation size, could be larger than the tensor size
    int hbm_index;             // which of the HBMs the tensor is placed
} nrt_tensor_device_allocation_info_t;
NRT_STATUS nrt_tensor_get_device_allocation_info(const nrt_tensor_t *tensor, nrt_tensor_device_allocation_info_t *alloc_info);

/**
 * @brief A Runtime API to check if a given output tensor is fully written/complete.
 *        If timeout is given as unbounded, it emits a warning at the first 30 seconds.
 *
 * @param output_tensor:  The given output tensor.
 * @param timeout:        The maximum total duration to wait for tensor completion in microseconds.
 *                        If timeout is negative, the wait is unbounded. The caller is in charge of handling the timeout behaviors.
 *                        o/w, it checks completion until the timeout.
 * @param expected_completion_count:  The number of completions expected by the caller.
 *
 * @return NRT_STATUS:    It returns NRT_SUCCESS if the tensor is complete;
 *                        It returns NRT_INVALID, if the output tensor is given as NULL;
 *                        It returns NRT_TIMEOUT if the tensor is not reaching the expected_completion_count within the timeout.
 */
NRT_STATUS nrt_tensor_check_output_completion(const nrt_tensor_t *output_tensor,
                                              int64_t timeout,
                                              uint64_t expected_completion_count);

/**
 * @brief A Runtime API to reset the completion counter inside an output tensor to 0.
 *
 * @param output_tensor:  The given output tensor.
 * @return NRT_STATUS:    It returns NRT_SUCCESS if reset is successful;
 *                        It returns NRT_INVALID, if the output tensor is given as NULL.
 */
NRT_STATUS nrt_tensor_reset_output_completion(nrt_tensor_t *output_tensor);

/**
 * @brief Get the anonymous file-descriptor of dma-buf associated with
 * a Neuron device memory region if it was registered for EFA peer direct
 *
 * @param addr[in]          - Device buffer virtual address
 * @param size[in]          - Device buffer size (in bytes)
 * @param fd[out]           - dma-buf fd
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_get_dmabuf_fd(uint64_t va, uint64_t size, int* fd);


/**  Get the host based device id from the device id presented to runtime (which may container based device id)
 * @param neuron_dev[in]      - device id
 * @param host_device_id[out] - host device id
 * @return NRT_SUCCESS if call was successful, NRT_INVALID otherwise
 */
NRT_STATUS nrt_host_device_id_get( int neuron_dev, uint32_t *host_device_id);

/**  Return array of routing IDs indexed by host device ID. This is the definitive routing ID mapping provided from the driver
 * @param coutn[in/out]           - [in] number of entries in the mapping table provided. [out] count of entries returned
 * @param host_did_to_rid_map[in] - table/map of routing IDs indexed by host device ID
 * @return NRT_SUCCESS if call was successful, NRT_INVALID otherwise
 */
NRT_STATUS nrt_host_device_id_rid_map_get(uint32_t *count, uint32_t *host_did_to_rid_map);

/**
 * Get the HBM virtual address and size for a specific HBM index.
 * @param device_id[in]         - Device ID
 * @param hbm_idx[in]           - HBM index
 * @param addr[out]             - Pointer to store the virtual address
 * @param size[out]             - Pointer to store the size of the HBM region
 * @return NRT_SUCCESS if call was successful and HBM region was mapped
 *         NRT_INVALID_HANDLE if there are no more HBM regions to map for this device
 *         NRT_INVALID if the interface isn't supported or for invalid parameters
 *         NRT_FAILURE for other errors
 */
NRT_STATUS nrt_get_hbm_mmap_va(int device_id, int hbm_idx, void **addr, size_t *size);


typedef struct nrt_vnc_memory_stats {
    size_t bytes_used;
    size_t bytes_limit;
    // NOTE: For backward compatibility, when making updates, don't delete existing fields, and
    //  ALWAYS add to the end of this struct!
} nrt_vnc_memory_stats_t;

/** Get the NRT memory stats for a VNC.
 *
 * @param vnc[in]             - Local VNC (within the instance)
 * @param stats[out]          - Pointer to a nrt_vnc_memory_stats struct
 * @param stats_size_in[in]   - Caller expected size of the nrt_vnc_memory_stats struct, for compatibility purposes
 * @param stats_size_out[out] - Library written size of the nrt_vnc_memory_stats struct, for compatibility purposes
 *
 * @return NRT_STATUS_SUCCESS on success.
 */

NRT_STATUS nrt_get_vnc_memory_stats(uint32_t vnc, nrt_vnc_memory_stats_t *stats, size_t stats_size_in, size_t *stats_size_out);

/** Get BDF of the EFA device attached to a Neuron device identified by VA of HBM allocation on that device
 *
 * @param va[in]            - VA of a memory allocated on a Neuron devices
 * @param efa_bdf[out]      - a buffer (of sufficient size) to store BDF of the connected EFA device
 * @param len[in/out]       - in: length of buffer (including NULL), out: length of string (excluding NULL)
 *
 * @return NRT_SUCCESS on success
 *         NRT_RESOUCE if the buffer is not large enough to store the BDF string
 *         NRT_FAILURE for other errors
 */

NRT_STATUS nrt_get_attached_efa_bdf(const void *va, char *efa_bdf, size_t *len);

/******************************
 * Out-of-NEFF collectives    *
 ******************************/

typedef struct nrt_cc_comm {
    uint32_t *replica_group; /* a list of participants */
    uint32_t rank; /* my rank in the replica_group */
    uint32_t rank_n; /* size of replica_group */

    uint32_t ctx_device_id;
    uint32_t ctx_device_count;
    uint32_t vnc;
} nrt_cc_comm_t;

typedef struct nrt_tensor_list {
    nrt_tensor_t **tensors;
    size_t num_tensors;
} nrt_tensor_list_t;

/** Build (initialize and setup) global communicator for host-driven collective operations.
 *
 * @param vnc[in]               - Local VNC (within the instance)
 * @param g_device_id[in]       - Global device id
 * @param g_device_count[in]    - Max world size of all participating workers
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_cc_global_comm_init(uint32_t vnc, uint32_t g_device_id, uint32_t g_device_count);

#ifdef __cplusplus
}
#endif
