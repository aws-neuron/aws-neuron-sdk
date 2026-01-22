/*
 * Shared tensor batch operation between runtime and driver.
 */

#ifndef NEURON_DRIVER_SHARED_TENSOR_BATCH_OP_H
#define NEURON_DRIVER_SHARED_TENSOR_BATCH_OP_H

#ifdef __KERNEL__
#include <linux/types.h>
typedef __u64 nrt_tensor_batch_offset_t;
typedef __u64 nrt_tensor_batch_size_t;
#else
#include <stdint.h>
typedef uint64_t nrt_tensor_batch_offset_t;
typedef uint64_t nrt_tensor_batch_size_t;
#endif

typedef struct nrt_tensor_batch_op {
    nrt_tensor_batch_offset_t offset;
    nrt_tensor_batch_size_t size;
    void *buffer;
} nrt_tensor_batch_op_t;

#endif  // NEURON_DRIVER_SHARED_TENSOR_BATCH_OP_H
