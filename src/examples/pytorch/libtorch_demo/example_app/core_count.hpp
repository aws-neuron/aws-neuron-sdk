#pragma once

/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
 
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NRT_SUCCESS = 0,
    NRT_FAILURE = 1,                        
    NRT_INVALID = 2,                        
    NRT_INVALID_HANDLE = 3,                 
    NRT_RESOURCE = 4,                      
    NRT_TIMEOUT = 5,                        
    NRT_HW_ERROR = 6,                      
    NRT_QUEUE_FULL = 7,                    
    NRT_LOAD_NOT_ENOUGH_NC = 9,             
    NRT_UNSUPPORTED_NEFF_VERSION = 10,    
    NRT_FAIL_HOST_MEM_ALLOC = 11,           
    NRT_EXEC_BAD_INPUT = 1002,              
    NRT_EXEC_COMPLETED_WITH_NUM_ERR = 1003, 
    NRT_EXEC_COMPLETED_WITH_ERR = 1004,     
    NRT_EXEC_NC_BUSY = 1005,                
    NRT_COLL_PENDING = 1100,                
} NRT_STATUS;

NRT_STATUS nrt_get_total_nc_count(uint32_t *nc_count);

#ifdef __cplusplus
}
#endif