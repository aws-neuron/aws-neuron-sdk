/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define RT_VERSION_DETAIL_LEN 128
#define GIT_HASH_LEN 64

typedef struct nrt_version {
    uint64_t rt_major;
    uint64_t rt_minor;
    uint64_t rt_patch;
    uint64_t rt_maintenance;
    char rt_detail[RT_VERSION_DETAIL_LEN];
    char git_hash[GIT_HASH_LEN];
} nrt_version_t;

/** Get the NRT library version
 *
 * @param ver[out]          - Pointer to nrt version struct
 * @param size[in]          - Length of the data needed to be filled in the nrt_version_struct
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_version(nrt_version_t *ver, size_t size);

#ifdef __cplusplus
}
#endif
