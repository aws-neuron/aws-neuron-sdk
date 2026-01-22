/*
 * Copyright 2020-2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <pthread.h>

#include "neuron_driver_shared.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NQ_DEV_TYPE {
    NQ_DEV_TYPE_NEURON_CORE = 0,
    NQ_DEV_TYPE_TOPSP,
    NQ_DEV_TYPE_MAX,
} ndl_nq_dev_t;

#define NEURON_MAX_DEVICES MAX_NEURON_DEVICE_COUNT
#define NEURON_DEVICE_PREFIX "/dev/neuron"
#define NEURON_DRIVER_LIBRARY_MAJOR 1
#define NEURON_DRIVER_LIB_MINOR 0
#define MAX_HBM_PER_DEVICE 4

#define DRIVER_VERSION_MAX_SIZE 32
typedef struct ndl_version_info {
    uint16_t driver_major_version;      // Major version of the driver
    uint16_t driver_minor_version;      // Minor version of the driver
    char driver_full_version[DRIVER_VERSION_MAX_SIZE];
    uint16_t library_major_version;     // Major version of the library
    uint16_t library_minor_version;     // Minor version of the library
} ndl_version_info_t;

/** Get version info.
 *
 * @param[out] version       - Buffer to store the version information.
 *
 * @return 0 on success.
 *         -1 on failed to read driver version.
 */
int ndl_get_version(ndl_version_info_t *version);

/** Gets the range of compatible version
 *
 * @param min_compatible_version_min [out]  - Lowest supported version
 * @param max_compatible_version_max [out]  - Highest supported version
 *
 * @return 0 on success.
 *
 */
int ndl_get_compatible_version(uint32_t *min_compatible_version, uint32_t *max_compatible_version);

typedef struct ndl_device_init_param {
    bool initialize_device; // if set to true, device is initialized as part of open()
    int num_dram_regions; // splits device DRAMs into given number of regions.
    bool map_hbm; // if set to true, HBM will be mapped during device open
} ndl_device_init_param_t;


#define NDL_COPY_BUF_SIZE (2ull * 1024 * 1024)
typedef struct ndl_copy_buf {
    uint64_t mem_handle;
    void *mmap_va;
    pthread_mutex_t lock;
} ndl_copy_buf_t;

// Maximum neuron devices supported on a system.
#define MAX_NEURON_DEVICE_COUNT 64

// Maximum neuron cores per device
#define MAX_NC_PER_DEVICE 8

typedef struct ndl_device {
    uint8_t device_index;                               // Device Index
    uint8_t device_type;                                // Device Type (V1, V2..)
    uint16_t device_revision;                           // Revision id of board
    uint8_t connected_device_count;                     // Number of devices connected to this device
    uint8_t connected_devices[MAX_NEURON_DEVICE_COUNT]; // Array of devices(IDs) connected to this device
    uint64_t csr_base[2];                               // BAR0/BAR2 base
    uint64_t csr_size[2];                               // BAR0/BAR2 size
    ndl_copy_buf_t cpy_bufs[MAX_NC_PER_DEVICE];         // MMAP buffers for efficiently copying data in/out of the device
    void *hbm_va[MAX_HBM_PER_DEVICE];                   // HBM virtual addresses
    size_t hbm_size;                                    // HBM sizes
    uint32_t hbm_va_cnt;                                // Number of active HBM regions
    uint32_t shift_hbm_size;                            // Cached number of bits to shift
    uint64_t hbm_offset[MAX_HBM_PER_DEVICE];            // HBM offsets
    uint8_t context[];                                  // Library reserved fields
} ndl_device_t;

typedef struct ndl_device_nc {
    ndl_device_t *device;
    uint32_t nc_id;
} ndl_device_nc_t;

typedef struct ndl_device_context {
    int nd_fd;
} ndl_device_context_t;

typedef struct ndl_mem_info {
    ndl_device_t *device;
    __u64 driver_handle;
    uint64_t pa;
    uint64_t mmap_offset;
    uint64_t size;
    uint32_t align;
    void *mmap_va;
    uint32_t host_memory;
    int nc_id;
} ndl_mem_info_t;

typedef struct ndl_notification_context {
    union {
        uint8_t nc_id; // neuron core index
        uint8_t nq_dev_id; // notification device index
    };
    ndl_nq_dev_t nq_dev_type; // notification device type
    uint8_t nq_type; // type of the notification queue
    uint8_t engine_index; // engine index
    uint32_t size; // size of the NQ in bytes
    int fd; // file descriptor of /dev/ndX/ncY/nqZ
    uint64_t offset; //mmap offset in the nd
    uint64_t mem_handle;
    void *va; // mmapped address

    ndl_mem_info_t *mem_info; // NQ memory info
} ndl_notification_context_t;

/**
 * Called by app the first time when it accesses the device.
 *
 * @param[in] device_index       - device index that is to be opened
 * @param[in] num_tdram_regions  - number of tdram regions
 * @param[out] device            - device specific information
 *
 * @return 0 on success.
 *         -1 on failure
 */
int ndl_open_device(int device_index, ndl_device_init_param_t *params, ndl_device_t **device);

/**
 * Called by app when it is done. After this, device cannot be accessed
 *
 * @param[in] device    - Device to close.
 *
 * @return 0 on success.
 *         -1 on failure
 */
int ndl_close_device(ndl_device_t *device);

/**
 * Get all the device index
 *
 * @param[out] device_indexes       - Buffer to store device indexes.
 * @param[in] device_indexes_size   - Size of the buffer in dwords.
 *
 * @return Number of devices found.
 */
int ndl_available_devices(int *device_indexes, int device_indexes_size);

/** Read from one or more registers.
 *
 * @param device[in]        - Device handle.
 * @param bar[in]           - BAR to read.
 * @param addresses[in]     - Array of register addresses.
 * @param count[in]         - Number of registers in the array.
 * @param buffer[out]       - Buffer to store read data.
 *
 * @return 0 on success.
 */
int ndl_bar_read(ndl_device_t *device, uint8_t bar, uint64_t *addresses, uint32_t count, uint32_t *buffer);

/** Write to one or more registers.
 *
 * @param device[in]        - Device handle.
 * @param bar[in]           - BAR to write.
 * @param addresses[in]     - Array of register addresses.
 * @param count[in]         - Number of registers in the array.
 * @param data[in]          - Data to write.
 *
 * @return 0 on success.
 */
int ndl_bar_write(ndl_device_t *device, uint8_t bar, uint64_t *addresses, uint32_t count, uint32_t *data);

/** Read hw counters from one or more addresses
 *
 * @param device[in]        - Device handle.
 * @param addresses[in]     - Array of register addresses.
 * @param count[in]         - Number of registers in the array.
 * @param buffer[out]       - Buffer to store read data.
 *
 * @return 0 on success.
 */
int ndl_read_hw_counters(ndl_device_t *device, uint64_t *addresses, uint32_t count, uint32_t *data);

/**
 * Retrieves the cached HBM virtual address for the specified device.
 *
 * @param device[in]        - Device handle.
 * @param hbm_idx[in]       - HBM index.
 * @param va[out]           - Resulting virtual address.
 * @param size[out]         - Size of the HBM
 * 
 * @return 0 on success, -EINVAL on failure, and -ENOENT when there are no more entries to be found.
 */
int ndl_get_hbm_va(ndl_device_t *device, int hbm_idx, void **va, size_t *size);

/** Allocates memory.
 *
 * @param device[in]        - Device to be associated with the allocation.
 * @param size[in]          - Number of bytes to allocate.
 * @param host_memory[in]   - If true allocate from host memory instead of using device memory.
 * @param dram_channel[in]  - DRAM channel to use in the device memory.
 * @param dram_region[in]   - DRAM region to use in the device memory.
 * @param nc_id[in]         - NC ID to use in the device
 * @param mem_alloc_type[in]- Type of memory allocation 
 * @param mem_handle[out]   - Allocated memory handle would be stored here.
 *
 * @return 0 on success.
 */
int ndl_memory_alloc(ndl_device_t *device, size_t size, uint64_t align, uint32_t host_memory, uint32_t dram_channel, uint32_t dram_region,
                        uint32_t nc_id, uint32_t mem_alloc_type, uint64_t *mem_handle);

/** Given a mem handle gets it PA - HACK to be removed
 * @param mem_handle[in]     - Memory handle
 * @parama pa[out]           - Physical address of handle
 *
 * @return the PA
 */
int ndl_memory_get_pa(uint64_t mem_handle, uint64_t *pa);

/** Map given m memory handle into virtual address space.
 *
 * @param mem_handle[in]     - Handle to map.
 * @param va[out]            - Resulting virtual address.
 *
 * @return 0 on success
 */
int ndl_memory_map(uint64_t mem_handle, void **va);

/** Unmap given memory handle from virtual address space.
 *
 * @param mem_handle[in]     - Handle to unmap.
 *
 * @return 0 on success
 */
int ndl_memory_unmap(uint64_t mem_handle);

/** Frees already allocated memory.
 *
 * @param mem_handle[in]   - Memory handle to be freed.
 *
 * @return 0 on success.
 */
int ndl_memory_free(uint64_t mem_handle);

/** Copy data from buffer to mem_handle.
 *
 * @param mem_handle[in]    - Handle on which data needs to be copied in.
 * @param buffer            - Buffer from which data needs to be copied.
 * @param offset            - Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 *
 * @return 0 on success.
 */
int ndl_memory_buf_copyin(uint64_t mem_handle, void *buffer, uint64_t offset, size_t size);

/** Copy data from mem_handle to buffer.
 *
 * @param mem_handle[in]    - Handle from which data needs to be copied out.
 * @param buffer            - Buffer to which data needs to be copied.
 * @param offset            - Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 *
 * @return 0 on success.
 */
int ndl_memory_buf_copyout(uint64_t mem_handle, void *buffer, uint64_t offset, size_t size);

/** Copy data from buffer to mem_handle (zero copy, buffer is pinned and used directly).
 *
 * @param mem_handle[in]    - Handle on which data needs to be copied in.
 * @param buffer            - Buffer from which data needs to be copied.
 * @param offset            - Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 *
 * @return 0 on success.
 */
int ndl_memory_buf_zerocopyin(uint64_t mem_handle, void *buffer, uint64_t offset, size_t size, int qid, uint32_t bar4_wr_threshold);

/** Copy data from mem_handle to buffer (zero copy, buffer is pinned and used directly).
 *
 * @param mem_handle[in]    - Handle from which data needs to be copied out.
 * @param buffer            - Buffer to which data needs to be copied.
 * @param offset            - Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 * @param qid               - H2T queue to use.  NEURON_DMA_H2T_DEFAULT_QID is default
 *
 * @return 0 on success.
 */
int ndl_memory_buf_zerocopyout(uint64_t mem_handle, void *buffer, uint64_t offset, size_t size, int qid);

/** Batch transfer data between host buffers and device memory.
 *
 * @param mem_handle[in]    - Device memory handle
 * @param ops[in]           - Array of batch operations
 * @param num_ops[in]       - Number of operations in batch
 * @param direction[in]     - Transfer direction (0=write to device, 1=read from device)
 * @param qid[in]           - H2T queue to use (-1 for default)
 *
 * @return 0 on success.
 */
int ndl_memory_buf_batch_copy(neuron_memcpy_batch_t *batches, uint64_t num_batches, uint32_t direction, int qid);

/** Copy data from buffer to addr in engine.
 *
 * @param device[in]        - Device information.
 * @param nc_id [in]        - Neuron core id.
 * @param dst [in]          - Address on which data needs to be copied in.
 * @param buffer            - Buffer from which data needs to be copied.
 * @param offset            - Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 * @param qid               - H2T queue to use.  NEURON_DMA_H2T_DEFAULT_QID is default
 *
 * @return 0 on success.
 */
int ndl_program_engine(ndl_device_t *device, uint32_t nc_id, uint64_t dst, void *buffer, uint64_t offset, size_t size);

/** Memset the given memhandle with passed byte value
 *
 * @param src_mem_handle[in]- Handle which needs to be filled with byte value
 * @param offset            - Src Offset in the mem handle.
 * @param value             - Byte value to fill the memory with
 * @param size              - Size in bytes to be copied.
 *
 * @return 0 on success.
 */
int ndl_memset(const uint64_t addr, uint64_t offset, const int value, const size_t size);

/** Copy data between mem_handles.
 *
 * @param src_mem_handle[in]- Handle from which data needs to be copied out.
 * @param dst_mem_handle[in]- Handle from which data needs to be copied to.
 * @param src_offset        - Src Offset in the mem handle.
 * @param dst_offset        - Dest Offset in the mem handle.
 * @param size              - Size in bytes to be copied.
 *
 * @return 0 on success.
 */
int ndl_memory_copy(uint64_t src_mem_handle, uint64_t dst_mem_handle, uint64_t src_offset, uint64_t dst_offset,
                    size_t size);


/** Copy data between mem_handles asynchronously.
 *
 * @param src_mem_handle[in]   - Handle from which data needs to be copied out.
 * @param dst_mem_handle[in]   - Handle from which data needs to be copied to.
 * @param src_offset           - Src Offset in the mem handle.
 * @param dst_offset           - Dest Offset in the mem handle.
 * @param size                 - Size in bytes to be copied.
 * @param prefetch_addr        - Host destination address associate with copy out operation to prefetch
 * @param wait_handle [in/out] - wait_handle [in] is for prev request, [out] is handle for this request
 *
 * @return 0 on success.
 */
int ndl_memory_copy_as(uint64_t src_mem_handle, uint64_t dst_mem_handle, uint64_t src_offset, uint64_t dst_offset, 
                       size_t size, uint64_t prefetch_addr, int * wait_handle);


/** Copy data between mem_handles.
 *
 * @param mem_handle[in]  - Handle from which data for this tran (either src or dst)
 * @param wait_handle     - wait_handle for an async dma
 *
 * @return 0 on success.
 */
int ndl_memory_copy_as_wait(uint64_t mem_handle, int wait_handle);

/** Set the dma engine state
 *
 * @param device_index[in]  - Device index.
 * @param eng_id[in]        - Eng ID that is initialized.
 * @param state[in]         - State that is set UDMA_NORMAL/UDMA_DISABLE etc
 *
 * @return 0 on success.
 */
int ndl_dma_eng_set_state(int device_index, uint32_t eng_id, uint32_t state);

/** Get the dma engine state
 *
 * @param device_index[in]  - Device index.
 * @param eng_id[in]        - Engine index which status needs to be collected.
 * @param state[out]        - Buffer to store engine state.
 *
 * @return 0 on success.
 */
int ndl_dma_eng_get_state(int device_index, uint32_t eng_id, struct neuron_dma_eng_state *state);

/** Get DMA queue state
 *
 * @param device_index[in]  - Device index.
 * @param eng_id [in]       - DMA engine index.
 * @param qid [in]          - DMA queue index.
 * @param tx [out]          - Tx queue state.
 * @param rx [out]          - Rx queue state.
 *
 * @return 0 on success.
 */
int ndl_dma_queue_get_state(int device_index, uint8_t eng_id, uint8_t qid, struct neuron_dma_queue_state *tx, struct neuron_dma_queue_state *rx);

/** Copy DMA descriptors to userspace.
 *
 *  This API needs root privilege.
 *
 * @param device_index[in]  - Device index.
 * @param eng_id [in]       - DMA engine index.
 * @param qid [in]          - DMA queue index.
 * @param type [in]         - Type of the queue.
 * @param index [in]        - Start descriptor index.
 * @param count [in]        - Number of descriptor needs to be copied.
 * @param buffer [out]      - Buffer to store the descriptors.
 *
 * @return 0 on success.
 */
int ndl_dma_descriptor_copyout(int device_index, uint8_t eng_id, uint8_t qid, enum neuron_dma_queue_type type, uint32_t start_index, uint32_t count, void *buffer);

/** Initialize the dma queue for a given engine
 *
 * @param device_index[in]  - Device index
 * @param eng_id[in]        - Engine for which the queue is initialized
 * @param qid[in]           - Queue id that needs to be initialized
 * @param tx_desc_count[in] - number of tx desc's need to be allocated
 * @param rx_desc_count[in] - number of rx desc's need to be allocated
 * @param tx_handle[in]     - TX mem handle
 * @param rx_handle[in]     - RX mem handle
 * @param rxc_handle[in]    - Completion mem handle
 *
 * @return 0 on success.
 */
int ndl_dma_queue_init(int device_index, uint32_t eng_id, uint32_t qid, uint32_t tx_desc_count, uint32_t rx_desc_count,
                       uint64_t tx_handle, uint64_t rx_handle, uint64_t rxc_handle, uint32_t axi_port);

struct ndl_queue_init {
    __u32 eng_id; // [in] DMA engine index
    __u32 qid; // [in] Queue index in the DMA engine
    __u32 tx_desc_count; // [in] number of tx desc's need to be allocated
    __u32 rx_desc_count; // [in] number of rx desc's need to be allocated
    __u64 tx_handle; // [in] mem handle for the tx ring
    __u64 rx_handle; // [in] mem handle for the rx ring
    __u64 rxc_handle; // [in] mem handle for the rxc ring
    __u32 axi_port; // [in] axi port
};

#define MAX_NDL_QUEUE_INIT_BATCH 256
struct ndl_queue_init_batch {
    __u32 count;
    struct ndl_queue_init entries[MAX_NDL_QUEUE_INIT_BATCH];
};

/** Initialize a batch of dma queues
 *
 * @param device_index[in]  - Device index
 * @param batch[in]         - Batch of dma queue initialization requests
 *
 * @return 0 on success.
 */
int ndl_dma_queue_init_batch(int device_idx, struct ndl_queue_init_batch *batch);

/** Release the dma queue for a given engine - only used in tests
 *
 * @param device_index[in]  - Device index
 * @param eng_id[in]        - Engine for which the queue is initialized
 * @param qid[in]           - Queue id that needs to be initialized
 *
 * @return 0 on success.
 */
int ndl_dma_queue_release(int device_index, uint32_t eng_id, uint32_t qid);

/** Starts DMA by copying the given number of descriptors or prefetch s2m
 *
 * @param device_index[in]  - Device index
 * @param eng_id[in]        - Engine for which the queue is initialized
 * @param qid[in]           - Queue id that needs to be initialized
 * @param tx_desc_count[in] - number of tx desc's need to be copied, could be 0 if called for s2m prefetch
 * @param rx_desc_count[in] - number of rx desc's need to be copied
 *
 * @return 0 on success.
 */
int ndl_dma_queue_copy_start(int device_index, uint32_t eng_id, uint32_t qid, uint32_t tx_desc_count, uint32_t rx_desc_count);

/** Acks the completed desc count for the eng/queue - only used in tests
 *
 * @param device_index[in]  - Device index
 * @param eng_id[in]        - Engine for which the queue is initialized
 * @param qid[in]           - Queue id that needs to be initialized
 * @param count[in]         - Number of desc's to ack
 *
 * @return 0 on success.
 */
int ndl_dma_ack_completed_desc(int device_index, uint32_t eng_id, uint32_t qid, uint32_t count);

/** Copy data from buffer to mem_handle. Buffer has dma desc
 *
 * @param mem_handle[in]        - Handle on which data needs to be copied in.
 * @param buffer[in]            - Buffer from which data needs to be copied. Buffer has dma desc
 * @param offset[in]            - Offset in the mem handle.
 * @param num_descs[in]         - Number of descriptors to copy
 * @param queue_type[in]        - From which queue copy descriptors.
 *
 * @return 0 on success.
 */
int ndl_dma_copy_descriptors(uint64_t mem_handle, void *buffer, uint64_t offset, uint32_t num_descs, enum neuron_dma_queue_type queue_type);

/** Reset given NCs within a device.
 *
 * @param device_index[in]  - Device to reset.
 * @param nc_map[in]        - NCs to reset (-1 to reset entire device)
 * @param request_id[out]   - ID for this reset request
 *
 * @return 0 on success.
 */
int ndl_reset_ncs(int device_index, int nc_map, uint32_t *request_id);

/** Register the callback to NRT to warn/nudge users when hitting soft incompatibility
 *
 * @param callback  - the call back function
 * @return int - 0 on success, otherwise on failure
 */
int ndl_register_soft_incompat_callback(void (*callback)(const char *));

/** Waits for readiness of given NCs within a device.
 *
 * @param device_index[in]  - Device index.
 * @param request_id[in]    - ID for the reset request to wait on
 * @param result[out]       - Buffer to store the result.
 *                            If the device is ready then this would be set to 1.
 *
 * @return 0 on success.
 *
 */
int ndl_ready_ncs(int device_index, uint32_t request_id, uint8_t *result);

/** Get info on all the apps that are currently using the device, caller needs to free returned info (*info)
 *
 * @param device_index[in]  - Device index.
 * @param info[out] - Pointer to a pointer which will hold app data, needs to be deallocated by caller
 * @param size[out] - Number of entries in neuron_app_info
 *
 * @return 0   - on success
 */
int ndl_get_all_apps_info(ndl_device_t *device, struct neuron_app_info **info, size_t *count, uint16_t apps_info_flags);

/** Increment a semaphore in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param semaphore_index[in]   - Semaphore which needs to be incremented.
 * @param value[in]             - Value to decrement.
 *
 * @return 0 on success
 */
int ndl_nc_semaphore_increment(ndl_device_t *device, int nc_index, uint32_t semaphore_index, uint32_t value);

/** Decrement a semaphore in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param semaphore_index[in]   - Semaphore which needs to be decremented.
 * @param value[in]             - Value to increment.
 *
 * @return 0 on success
 */
int ndl_nc_semaphore_decrement(ndl_device_t *device, int nc_index, uint32_t semaphore_index, uint32_t value);

/** Get semaphore value in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param semaphore_index[in]   - Semaphore index.
 * @param value[out]            - Buffer where read value would be stored.
 *
 * @return 0 on success
 */
int ndl_nc_semaphore_read(ndl_device_t *device, int nc_index, uint32_t semaphore_index, uint32_t *value);


/** Write given value into the semaphore in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param semaphore_index[in]   - Semaphore index.
 * @param value[in]             - Value to write.
 *
 * @return 0 on success
 */
int ndl_nc_semaphore_write(ndl_device_t *device, int nc_index, uint32_t semaphore_index, uint32_t value);


/** Get event value in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param semaphore_index[in]   - Semaphore index.
 * @param value[out]            - Buffer where read value would be stored.
 *
 * @return 0 on success
 */
int ndl_nc_event_get(ndl_device_t *device, int nc_index, uint32_t event_index, uint32_t *value);


/** Set a event in Neuron Core.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron Core index
 * @param event_index[in]       - Event index.
 * @param value[in]             - Value to write.
 *
 * @return 0 on success
 */
int ndl_nc_event_set(ndl_device_t *device, int nc_index, uint32_t event_index, uint32_t value);


/** Configure notification queue
 *
 * Neuron device has multiple of neuron cores and TOP_SPs. If nq_dev_type is
 * NQ_DEV_TYPE_NEURON_CORE, nq_dev_index conveys neuron core index. In case of
 * NQ_DEV_TYPE_NEURON_TOPSP, nq_dev_index means TOP_SP index.
 *
 * @param device[in]            - Device
 * @param nq_dev_id[in]         - Notification device index
 * @param nq_dev_type[in]       - Notification device type
 * @param nq_type[in]           - Notification queue type
 * @param engine_index[in]      - Engine index
 * @param size[in]              - Size in bytes
 * @param on_host_memory[in]    - If true, NQ is created on host memory
 * @param dram_channel          - If NQ is created on device, DRAM channel to use
 * @param dram_region           - If NQ is created on device, DRAM region to use
 * @param force_alloc_mem       - If true, force allocate new memory (and delete already allocated memory, if any)
 * @param context[out]          - Resulting NQ context.
 *
 * @return 0 on success.
 */
int ndl_notification_init(ndl_device_t *device, int nq_dev_id, ndl_nq_dev_t nq_dev_type, uint8_t nq_type, uint8_t engine_index,
                          uint32_t size, bool on_host_memory, uint32_t dram_channel, uint32_t dram_region,
                          uint64_t *notification_context);

/** Configure notification queue with option to force re-allocate/re-size
 *
 * Neuron device has multiple of neuron cores and TOP_SPs. If nq_dev_type is
 * NQ_DEV_TYPE_NEURON_CORE, nq_dev_index conveys neuron core index. In case of
 * NQ_DEV_TYPE_NEURON_TOPSP, nq_dev_index means TOP_SP index.
 *
 * @param device[in]            - Device
 * @param nq_dev_id[in]         - Notification device index
 * @param nq_dev_type[in]       - Notification device type
 * @param nq_type[in]           - Notification queue type
 * @param engine_index[in]      - Engine index
 * @param size[in]              - Size in bytes
 * @param on_host_memory[in]    - If true, NQ is created on host memory
 * @param dram_channel          - If NQ is created on device, DRAM channel to use
 * @param dram_region           - If NQ is created on device, DRAM region to use
 * @param force_alloc_mem       - If true, force allocate new memory (and delete already allocated memory, if any)
 * @param context[out]          - Resulting NQ context.
 *
 * @return 0 on success.
 */
int ndl_notification_init_with_realloc(ndl_device_t *device, int nq_dev_id, ndl_nq_dev_t nq_dev_type, uint8_t nq_type, uint8_t engine_index,
                          uint32_t size, bool on_host_memory, uint32_t dram_channel, uint32_t dram_region, bool force_alloc_mem,
                          uint64_t *notification_context);

/** Returns mem_handle associated with the NQ
 *
 * @param notification_context[in]  - Notification context
 * @param mem_handle[out]           - Notification's memory handle would be stored here.
 *
 * @return 0 on success, 1 on failure
 */
int ndl_notification_get_mem_handle(uint64_t notification_context, uint64_t *mem_handle);

/** Returns size associated with the NQ
 *
 * @param notification_context[in]  - Notification context
 * @param size[out]           - Notification's size would be stored here.
 *
 * @return 0 on success, 1 on failure
 */

int ndl_notification_get_size(uint64_t notification_context, uint32_t *size);

/** Maps NQ to virtual address.
 *
 * @param notification_context[in]  - Notification context.
 * @param va [out]                  - Virtual address where the mapping is done.
 * @return 0 on success
 */
int ndl_notification_map(uint64_t notification_context, void **va);

/** Stops and destroys already configured notification queue.
 *
 * @param notification_context[in] - Notification context.
 *
 * @return 0 on success.
 */
int ndl_notification_destroy(uint64_t notification_context);

/** Makes neuron ds available for use and returns a valid pointer in **data and a valid size in *size
 *
 * @param device[in]            - Device
 * @param pid[in]               - PID for this NDS (if 0 it allocates a new one)
 * @param data[out]             - Will contain a valid pointer to the datastore
 * @param size[out]             - Will contain a valid size for the datastore
 *
 * @return 0 on success.
 */
int ndl_nds_open(ndl_device_t *device, int32_t pid, void **data, size_t *size);

/** Decreases ref count for the given pid
 *
 * @param device                - Device
 * @param pid                   - PID owning the datastore
 * @param data                  - Pointer to datastore raw data (returned by ndl_nds_open)
 * @param size                  - Size of datastore (returned by ndl_nds_open)
 *
 * @return 0 on success.
 */
int ndl_nds_close(ndl_device_t *device, int32_t pid, void *data, size_t size);

/** Enter inference critical section.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron core index
 * @param uuid[in]              - UUID of the model expected to be loaded
 *
 * This function would fail if the UUID is different or PID
 * which loaded the UUID is different.
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_reader_enter(ndl_device_t *device, int nc_index, struct neuron_uuid uuid);

/** Exit inference critical section.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron core index
 * @param uuid[in]              - UUID of the model expected to be loaded
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_reader_exit(ndl_device_t *device, int nc_index, struct neuron_uuid uuid);

/** Enter model load critical section.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron core index
 * @param uuid[in]              - UUID of the model to be loaded
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_writer_enter(ndl_device_t *device, int nc_index, struct neuron_uuid uuid);

/** Exit model load critical section and enter inference critical section.
 *
 * @param device[in]            - Device
 * @param nc_index[in]          - Neuron core index
 * @param uuid[in]              - UUID of the loaded model
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_writer_downgrade(ndl_device_t *device, int nc_index, struct neuron_uuid uuid);

/** Find given number of free NCs and mark them as used.
 *
 * @param nc_count[in]          - Number of free neuron cores needed.
 * @param start_nc[in]          - From where to start the free core search.
 * @param end_nc[in]            - Last NC where to stop the free core search.
 * @param max_nc_available[out] - Maximum number of free cores available.
 * @param bitmap[out]           - Bitmap of marked neuron core indexes.
 * @param size[in]              - size of the bitmap in bytes
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_nc_range_mark(uint32_t nc_count, uint32_t start_nc, uint32_t end_nc,
                           uint32_t *max_nc_available, uint64_t *bitmap, size_t size);

/** Unmark neuron cores as free.
 *
 * @param bitmap[in]           - Bitmap of marked neuron core indexes.
 * @param size[in]             - size of the bitmap in bytes
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_crwl_nc_range_unmark(uint64_t *bitmap, size_t size);

/** Gets the info for the copy buffer for copying data to/from device
 *
 * To dma data in and out of the device, app needs a host dram buffer allocated
 * by the driver. Allocating this every-time is expensive especially if we want
 * a bigger copy size. To avoid this performance penalty, applications can use
 * this preallocated buffer.
 *
 * @param device[in]    - Device
 * @param nc_id[in]     - nc id the copy buffer is from
 * @param cpy_buf[out]  - Pointer to copy buffer
 *
 * @return 0 on success
 */
int ndl_get_copy_buf(ndl_device_t *device, uint32_t nc_id, ndl_copy_buf_t **cpy_buf);

/** Set the neuron core init state
 * Initially the state is set to started and then app intializes the neuron core. Then
 * it sets the state to completed. If any other app tries to set the state to started when it
 * is already started then this routine will block until the init is done or timeout
 *
 * @param device[in]            - Device
 * @param state[in]             - State that will be state
 * @param new_state[out]        - State after the set is done
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_nc_init_set_state(ndl_device_t *device, uint32_t nc_id, uint32_t state, uint32_t *new_state);

/** Gets the state of model start. If this is the first model that will be loaded in the nc.
 *
 * @param device[in]            - Device
 * @param nc_id[in]             - nc id
 * @param started_count[out]    - number of times model started in that nc
 *
 * @return 0 on success, -1 on failure.
 */
int ndl_nc_model_started_count(ndl_device_t *device, uint32_t nc_id, uint64_t *started_count);

/** Gets the architecture & revision of the board
 *
 * @param architecture[out]          - Architecture of the board
 * @param revision[out]              - Revision of the board
 *
 * @return 0 on success
 */
int ndl_get_board_info(uint32_t *architecture, uint32_t *revision);

/** Gets BDF for a device - only for devices opened by the calling process - DEPRECATED don't use
 *
 * @param bus_num[out]               - Bus number for this device
 * @param pci_slot[out]              - PCI slot for this device
 * @param dev_func[out]              - Device function for this device
 *
 * @return 0 on success
 */
int ndl_get_device_bdf(int device_index, uint32_t *bus_num, uint8_t *pci_slot, uint8_t *dev_func);

/**
 * @brief Get the anonymous file-descriptor of dma-buf associated with
 * a Neuron device memory region if it was registered for EFA peer direct
 *
 * @param addr[in]        - Device buffer virtual address
 * @param size[in]        - Device buffer size (in bytes)
 * @param fd[out]         - dma-buf fd
 *
 * @return 0 on success
 */
int ndl_get_dmabuf_fd(uint64_t addr, uint64_t size, int* fd);

/** Gets BDF for a device
 *
 * @param device_index[in]           - Neuron device index
 * @param domain[out]                - PCIe domain for the device
 * @param bus_num[out]               - Bus number for the device
 * @param pci_slot[out]              - PCI slot for the device
 * @param dev_func[out]              - Device function for the device
 *
 * @return 0 on success
 */
int ndl_get_device_bdf_ext(int device_index, uint32_t *domain, uint32_t *bus_num, uint8_t *pci_slot, uint8_t *dev_func);

/** retrieve offset/size where to mmap around a physical address
 * 
 * @param device[in]             - Neuron device
 * @param pa[in]                 - physical address in device mem to retrieve mc mmap info for
 * @param mmap_offset[out]       - mmap offset
 * @param mem_handle[out]        - The handle for the given physical address.
 *                                 Set to 0 when using backwards compatible interface with old drivers.
 * @param size[out]              - size
 *
 */
int ndl_mem_get_mc_mmap_info(ndl_device_t *device, uint64_t pa, uint64_t *mmap_offset, uint64_t *size, uint64_t *mem_handle);

/** mmap a bar region into user address
 *
 * @param device[in]          - Neuron device
 * @param block[in]           - block type containing the resource
 * @param block_id[in]        - id of the block if is more than one block
 * @param resource[in]        - resource the caller wants to mmap
 * @param va[out]             - virual address of the resource
 * @param size[out]           - size of the resource
 *
 */
int ndl_mmap_bar_region( ndl_device_t *device, enum neuron_dm_block_type block, uint32_t block_id, enum neuron_dm_resource_type resource, 
                        void ** va, uint64_t * size);

/** Close all cached FDs
 *
 */
void ndl_device_cached_fd_close_all(void);

/** Log an error message to kernel messages/serial console
 * 
 * @param str[in]             - The error message
 * @param size[in]            - The size of the error message including null terminator
 * @param action[in]          - Additional action to perform
 * 
 * @return On success: 0
 *         On failure: -1 and:
 *           * errno == EFAULT when size is too large
 *           * errno == EBADMSG when str is not null terminated
 */
int ndl_printk(char *str, uint32_t size, uint32_t action);

/** get the host device id for an open device (for containers)
 * 
 * @param device[in]           - Neuron device
 * @param host_device_id[out]  - host device id 
 * 
 */
int ndl_get_host_device_id(ndl_device_t *device, uint32_t *host_device_id);

/** return device id to routing id mapping table along with number of entries in the table
 *
 * @param count[in/out]             - [in] size of map in entries.  [out] # entries returned
 * @param host_did_to_rid_map[out]  - map of host device id to routing ids 
 *
 */
int ndl_get_host_device_id_to_rid_map(uint32_t *count, uint32_t *host_did_to_rid_map);

int ndl_dump_device_allocation_info(ndl_device_t *device, uint32_t hbm_index, struct neuron_ioctl_mem_chunk_info *data, uint32_t *num_entries);

/** ask the driver to dump neuron core process info 
 *
 * @param nc_id[in]             - [in] neuron core to dump process info for
 * @param filter_log_owner[in]  - [in] only dump log entries for the owner pid of the neuron core
 * @param log_dump_limit[in]    - [in] max number of log entries to dump
 *
 */
int ndl_dump_nc_pid_info(uint32_t nc_id, bool filter_log_owner, uint32_t log_dump_limit);

/** write a value to entire HBM accessible to Neuron (so excludes firmware carveout)
 *
 *  @param hbm_index     - HBM to write to
 *  @param init_val      - value to write
 */
int ndl_hbm_scrub_start(ndl_device_t *device, uint32_t nc_id, uint32_t hbm_index, uint32_t axi_port, uint32_t init_val);
int ndl_hbm_scrub_wait(ndl_device_t *device, uint32_t nc_id, uint32_t hbm_index);

/** Gets the tpb mapping.
 *
 * @param map[out]              - Location to store the mapping information
 * @param max_num_entries[in]   - Maximum number of entries we can store in `map`
 * @param mapping_version[in]   - Flavor of mapping to get from the driver
 *
 * @return 0 on success
 */
int ndl_get_logical_to_physical_nc_map(struct neuron_ioctl_nc_map *map, uint32_t max_num_entries, enum neuron_ioctl_nc_mapping_type mapping_version);

/** return pod information
 *
 * @param pod_type[out] - type of pod
 * @param pod_sz[out]   - size of the pod
 *
 */
int ndl_pod_info(uint32_t * pod_type, uint32_t * pod_sz);

/** return pod election state
 *
 * @param state[out] - election state
 *
 */
int ndl_pod_election_state(uint32_t * state);

/** return pod mapping information.
 *
 * @param node_id[out]          - node id of the pod node.  -1 if the node is not part of a configured pod
 *
 */
int ndl_pod_mapping_info(int * node_id);


/** return pod status
 *
 * @param pod_id[out]           - pod id.  Only valid it the pod is configured as a pod
 * @param state[out]            - state of the pod election 
 * @param pod_type[out]         - type of pod 
 * @param pod_sz[out]           - size of the pod.  0 if the node is not part of a pod
 * @param node_id[out]          - node id of the pod node.  -1 if the node is not part of a configured pod
 * @param mode[out]             - current operating mode
 * @param modes_supported[out]  - supported operating modes
 *
 */
int ndl_pod_status(uint8_t *pod_id, uint32_t *state, uint32_t *pod_type, uint32_t *pod_sz, int *node_id, 
                   enum neuron_ultraserver_mode *mode, uint32_t *modes_supported);


/** control pod election state
 *
 * @param ctrl[in]           - control request.  (enum neuron_pod_ctrl_req)
 * @param mode[in]           - requested operating mode
 * @param timeout[in]        - timeout for control operation
 * @param state[out]         - state of the pod election 
 *
 */
int ndl_pod_ctrl(uint32_t ctrl, enum neuron_ultraserver_mode mode, uint32_t timeout, uint32_t *state);

int ndl_alloc_contiguous_scratchpad(ndl_device_t *device, uint64_t size, uint32_t hbm_index, uint32_t nc_id, uint64_t *mem_handle);

/** Similar to ndl_memory_map - only difference is that a contiguous scratchpad var may span multiple contiguous memchunks. So size of memory mapping is different from just the size of the first contiguous memchunk.
 *
 * @param mem_handle[in]     - Handle to map.
 * @param va[out]            - Resulting virtual address.
 * @param size[in]           - Size to map 
 *
 * @return 0 on success
 */

int ndl_memory_map_contiguous_scratchpad(uint64_t mem_handle, void **va, uint64_t size);

/** Set performance profile
 *
 * @param device[in]        - Device handle.
 * @param profile[in]       - Performance profile to set.
 *
 * @return 0 on success.
 */
int ndl_set_performance_profile(ndl_device_t *device, uint32_t profile);

bool ndl_feature_supported(int nd_fd, uint64_t feature);

/** dynamically allocate h2t queues (rings)
 *
 * @param device[in]                - Neuron device
 * @param nc_id[in]                 - neuron core to allocate h2t queues for
 * @param copy_queue_cnt[in]        - number of h2t copy queues to allocate
 * @param service_queue_cnt[in]     - number of service queues to allocate
 * @param copy_queue_bmap[out]      - bitmap of the allocated copy queues
 * @param servic_equeue_bmap[out]   - bitmap of the allocated service queues
 * @param copy_default_queue[out]   - default h2t copy queue
 *
 */
int ndl_h2t_dma_queue_alloc(ndl_device_t *device,  uint32_t nc_id, uint32_t copy_queue_cnt, uint32_t service_queue_cnt,
                            uint32_t *copy_queue_bmap, uint32_t *service_queue_bmap, uint32_t *copy_default_queue);

/** free dynamically allocated h2t queues
 *
 * @param device[in]           - Neuron device
 * @param nc_id[in]            - [in] neuron core to free queues for
 * @param queue_bmap[in]       - [in] bitmap of queues to free
 *
 */
int ndl_h2t_dma_queue_free(ndl_device_t *device,  uint32_t nc_id, uint32_t queue_bmap);

/** control metrics posting behavior
 *
 * @param device[in]            - Neuron device
 * @param mode[in]              - how to modify posting behavior (enable or disable periodic posting)
 */
int ndl_metrics_ctrl(ndl_device_t *device, enum neuron_metrics_mode mode);

/**
 * arbitrary size bitmap support
 *
 */
#define NBM_NR_BITS(t)     (sizeof(t)*8)
#define NBM_NR_ENT(nr,t)   (((nr)+NBM_NR_BITS(t)-1) / NBM_NR_BITS(t))

static inline uint32_t nbitmap_test_bit(uint32_t nr, uint64_t *addr)
{
    return (addr[nr/NBM_NR_BITS(*addr)] & (1ull << (nr % NBM_NR_BITS(*addr)))) != 0ull;
}

static inline void nbitmap_set_bit(uint32_t nr, uint64_t *addr)
{
    addr[nr/NBM_NR_BITS(*addr)] |= (1ull << (nr % NBM_NR_BITS(*addr)));
}

static inline uint32_t nbitmap_ffs1(uint32_t nr, uint64_t *addr)
{
    int i;
    for (i=0; i < NBM_NR_ENT(nr, *addr); i++) {
        uint32_t x = __builtin_ffsl(addr[i]);
        if (x) 
            return i * NBM_NR_BITS(*addr) + x;
    }
    return 0;
}

static inline uint32_t nbitmap_popcount(uint32_t nr, uint64_t *addr)
{
    int i;
    uint32_t cnt = 0;
    for (i=0; i < NBM_NR_ENT(nr, *addr); i++) {
        cnt += __builtin_popcountll(addr[i]);
    }
    return cnt;
}
static inline void nbitmap_clr_bit(uint32_t nr, uint64_t *addr)
{
    addr[nr/NBM_NR_BITS(*addr)] &= ~(1ull << (nr % NBM_NR_BITS(*addr)));
}

#ifdef __cplusplus
}
#endif
