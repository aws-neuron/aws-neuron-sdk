/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>
#include <neuron/ndl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Main NDS object handle
typedef void *nds_obj_handle_t;

// NDS object types
#define OBJECT_TYPE_MODEL_NODE_INFO     (0)
#define OBJECT_TYPE_PROCESS_INFO        (1)
#define OBJECT_TYPE_PROCESS_INFO_EXT    (2)

// Model-related structs
#define MODEL_MEM_USAGE_LOCATION_COUNT 2

/*
 * Number of slots for mem_usage_type in Neuron Datastore (also used by tools)
 *
 * In the current version of the neuron datastore's format, there are only 12 slots for storing
 * memory usage type, so we aggregate them using the same logic as for the 'per NC' memory tracker.
 * Monitor always aggregated them even further by adding them together, so we aren't breaking any feature.
 *
 * For usage types definiton, go to "inc/tdrv/dma_mem_usage_type.h"
 *
 */
enum {
    NDS_DMA_MEM_USAGE_SLOT_CODE,
    NDS_DMA_MEM_USAGE_SLOT_TENSORS,
    NDS_DMA_MEM_USAGE_SLOT_CONSTANTS,
    NDS_DMA_MEM_USAGE_SLOT_SCRATCHPAD,
    NDS_DMA_MEM_USAGE_SLOT_MISC,
    NDS_DMA_MEM_USAGE_SLOT_COUNT = 12 // do not change
};

// Aggregated data for all chunks of the same type/location
typedef struct nds_mem_usage_info {
    uint64_t total_size;        // Total size
    uint32_t chunk_count;       // Number chunks that make up the total size
} nds_mem_usage_info_t;

// Loaded model node information
typedef struct nds_model_node_info {
    uint32_t model_id;           // parent model id
    uint32_t model_node_id;      // node id
    char name[256];              // model name
    char uuid[16];               // uuid
    uint8_t nc_index;            // nc index
    uint8_t sg_index;            // subgraph index
} nds_model_node_info_t;

// Loaded model node memory usage information
typedef struct nds_model_node_mem_usage_info {
    // MODEL_MEM_USAGE_LOCATION_COUNT per each usage type
    nds_mem_usage_info_t model_mem_usage[MODEL_MEM_USAGE_LOCATION_COUNT][NDS_DMA_MEM_USAGE_SLOT_COUNT];
} nds_model_node_mem_usage_info_t;

// Version information
typedef struct nds_version_info {
    uint8_t major;
    uint8_t minor;
    uint32_t build;
} nds_version_info_t;

// Process information-related struct
typedef struct nds_process_info {
    int8_t  framework_type;
    char    tag[32];
    nds_version_info_t framework_version;
    nds_version_info_t fal_version;
    nds_version_info_t runtime_version;
} nds_process_info_t;

// Extended process information
typedef struct nds_process_info_ext {
    char tag[256];
} nds_process_info_ext_t;

typedef struct nds_instance nds_instance_t;
typedef struct ndl_device ndl_device_t;


// Feature bitmap's bit index information
typedef enum feature_bitmap_bit_index {
    BIT_INDEX_TEST_FEATURE = 0,
    BIT_INDEX_MULTICORE_FEATURE = 1,

    BIT_INDEX_COUNT = BIT_INDEX_MULTICORE_FEATURE + 1
} feature_bitmap_bit_index_t;


/** Opens NDS for the given pid. If pid == 0, it acquires it for the current PID
 *  and it's opened in read-write mode. If pid != 0, it acquires it for the provided PID
 *  and it's opened as read-only.
 *
 * @param device[in]            - ndl_device used to open this NDS
 * @pid pid[in]                 - pid for which to open the NDS, if 0 - it's opened as r/w for the current process
 * @inst[out]                   - address of a pointer which will contain the instance handle
 *
 * @return non zero in case of error
 */
int nds_open(ndl_device_t *device, pid_t pid, nds_instance_t **inst);

/** Releases the NDS instance and frees the data associated with it (mandatory for readers)
 *
 * @param inst[in]              - NDS instance to close
 *
 * @return non zero in case of error, the pointer gets deleted regardless
 */
int nds_close(nds_instance_t *inst);

/* --------------------------------------------
 * NDS Neuroncore Counters
 * --------------------------------------------
 */

/** Increments a simple per-nc counter
 *
 * @param inst[in]              - NDS instance
 * @param nc_index[in]          - Neuroncore index
 * @param counter_index[in]     - Counter index
 * @param increment[in]         - Amount to increment
 *
 * @return 0 on success.
 */
int nds_increment_nc_counter(nds_instance_t *inst, int nc_index, uint32_t counter_index, uint64_t increment);

/** Decrements a simple per-nc counter
 *
 * @param inst[in]              - NDS instance
 * @param nc_index[in]          - Neuroncore index
 * @param counter_index[in]     - Counter index
 * @param increment[in]         - Amount to increment
 *
 * @return 0 on success.
 */
int nds_decrement_nc_counter(nds_instance_t *inst, int nc_index, uint32_t counter_index, uint64_t decrement);

/** Gets a simple per-nc counter
 *
 * @param inst[in]              - NDS instance
 * @param nc_index[in]          - Neuroncore index
 * @param counter_index[in]     - Counter index
 * @param value[out]            - Counter value
 *
 * @return 0 on success.
 */
int nds_get_nc_counter(nds_instance_t *inst, int nc_index, uint32_t counter_index, uint64_t *value);

/** Sets a simple per-nc counter
 *
 * @param inst[in]              - NDS instance
 * @param nc_index[in]          - Neuroncore index
 * @param counter_index[in]     - Counter index
 * @param value[in]             - Value to set the counter to
 *
 * @return 0 on success.
 */
int nds_set_nc_counter(nds_instance_t *inst, int nc_index, uint32_t counter_index, uint64_t *value);

/* --------------------------------------------
 * NDS Neuron Device Counters
 * --------------------------------------------
 */

/** Increments a simple per-nd counter - may overflow
 *
 * @param inst[in]              - NDS instance
 * @param counter_index[in]     - Counter index
 * @param increment[in]         - Amount to increment
 *
 * @return 0 on success.
 */
int nds_increment_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t increment);

/** Decrements a simple per-nd counter - may overflow
 *
 * @param inst[in]              - NDS instance
 * @param counter_index[in]     - Counter index
 * @param decrement[in]         - Amount to decrement
 *
 * @return 0 on success.
 */
int nds_decrement_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t decrement);

/** Bitwise inclusive OR operation on counter
 * 
 * @param inst[in]              - NDS instance
 * @param counter_index[in]     - Counter index
 * @param 1ull << bit_index     - bit mask on the feature bitmap
 * 
 * @return 0 on success.
 */
int nds_or_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t bit_index);

/** Gets a simple per-nd counter
 *
 * @param inst[in]              - NDS instance
 * @param counter_index[in]     - Counter index
 * @param value[out]            - Counter value
 *
 * @return 0 on success.
 */
int nds_get_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t *value);

/** Sets a simple per-nd counter
 *
 * @param inst[in]              - NDS instance
 * @param counter_index[in]     - Counter index
 * @param value[in]             - Value to set the counter to
 *
 * @return 0 on success.
 */
int nds_set_nd_counter(nds_instance_t *inst, uint32_t counter_index, uint64_t *value);

/* --------------------------------------------
 * NDS objects
 * --------------------------------------------
 */

/** Writes an NDS object to the NDS memory
 *
 * @param obj[in]               - NDS object handle
 *
 * @return 0 on success.
 */
int nds_obj_commit(nds_obj_handle_t obj);

/** Creates a new NDS object with the given type
 *
 * @param inst[in]              - NDS instance
 * @param type[in]              - type of object to create
 *
 * @return handle for newly created object
 */
nds_obj_handle_t nds_obj_new(nds_instance_t *inst, int type);

/** Deletes a NDS object from NDS (and local memory)
 *
 * @param obj[in]               - NDS object handle
 *
 * @return 0 on success.
 */
int nds_obj_delete(nds_obj_handle_t obj);

/** Casts this NDS object to a mode_node_info_t which can be used for r/w
 *
 * @param obj[in]               - NDS object handle
 *
 * @return non-NULL on success.
 */
nds_model_node_info_t *nds_obj_handle_to_model_node_info(nds_obj_handle_t obj);

/** Casts this NDS object to a nds_model_node_mem_usage_info_t which can be used for r/w
 *
 * @param obj[in]               - NDS object handle
 *
 * @return non-NULL on success.
 */
nds_model_node_mem_usage_info_t *nds_obj_handle_to_model_node_mem_usage(nds_obj_handle_t obj);

/** Reads all model info data and returns it as an array (needs to be deleted by caller)
 *
 * @param inst[in]              - NDS instance
 * @param models[out]           - Pointer where to write the address of an array of length count containing object handles
 * @param count[out]            - Number of models loaded (present in the models array)
 *
 * @return non-NULL on success.
 */
int nds_read_all_model_nodes(nds_instance_t *inst, nds_obj_handle_t **models, size_t *count);

/** Casts this NDS object to a nds_process_info_t which can be used for r/w
 *
 * @param obj[in]               - NDS object handle
 *
 * @return non-NULL on success.
 */
nds_process_info_t *nds_obj_handle_to_process_info(nds_obj_handle_t obj);

/** Casts this NDS object to a nds_process_info_ext_t which can be used for r/w
 *
 * @param obj[in]               - NDS object handle
 *
 * @return non-NULL on success.
 */
nds_process_info_ext_t *nds_obj_handle_to_process_info_ext(nds_obj_handle_t obj);

/** Reads process info and returns a nds_obj_handle
 *
 * @param inst[in]              - NDS instance
 *
 * @return non-NULL on success.
 */
nds_obj_handle_t nds_read_process_info(nds_instance_t *inst);

/** Reads extended process info and returns a nds_obj_handle
 *
 * @param inst[in]              - NDS instance
 *
 * @return non-NULL on success.
 */
nds_obj_handle_t nds_read_process_info_ext(nds_instance_t *inst);

#ifdef __cplusplus
}
#endif
