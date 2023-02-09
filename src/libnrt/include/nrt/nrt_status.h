/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: if making changes here please also keep
// KaenaTools: KaenaTools: pkg/rt/rt.go in sync

typedef enum {
    NRT_SUCCESS = 0,
    NRT_FAILURE = 1,                        // non specific failure, don't use if there is more descriptive type
    NRT_INVALID = 2,                        // e.g. invalid NEFF, bad instruction, bad DMA descriptor, input tensor name/size does not match the model, etc.

                                            // TODO invalid_handle is no longer useful because handles are not passed in nrt API
                                            // remove
    NRT_INVALID_HANDLE = 3,                 // make this one explicit instead of using more generic INVALID_INPUT because it could be a common caller mistake
    NRT_RESOURCE = 4,                       // failed to allocate a resource for requested operation

                                            // TODO separate exec timeout from others
    NRT_TIMEOUT = 5,                        // operation timed out
    NRT_HW_ERROR = 6,                       // Hardware failure
    NRT_QUEUE_FULL = 7,                     // not enough space in the execution input queue
    NRT_LOAD_NOT_ENOUGH_NC = 9,             // Failed to allocate enough NCs for loading a NEFF
    NRT_UNSUPPORTED_NEFF_VERSION = 10,      // Unsupported version of NEFF

    // Unique retcodes to help the caller identify when nrt apis are called outside the scope of nrt_init() and nrt_close()
    NRT_UNINITIALIZED = 13,
    NRT_CLOSED = 14,

    NRT_EXEC_BAD_INPUT = 1002,              // invalid input has been submitted to exec()
    NRT_EXEC_COMPLETED_WITH_NUM_ERR = 1003, // execution was completed with numerical errors (produced NaN)
    NRT_EXEC_COMPLETED_WITH_ERR = 1004,     // execution was completed with other errors,
                                            // either logical - event double clear, or physical - parity error
    NRT_EXEC_NC_BUSY = 1005,                // the neuron core is locked (in use) by another model/process
    NRT_COLL_PENDING = 1100,                // collective operation is still pending
} NRT_STATUS;

#ifdef __cplusplus
}
#endif
