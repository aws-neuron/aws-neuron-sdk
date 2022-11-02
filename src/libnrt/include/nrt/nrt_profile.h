/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <nrt/nrt.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Enable profiling for a model
 *
 * @param model[in]     - model to profile
 * @param filename[in]  - file to save profile information to
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_profile_start(nrt_model_t *model, const char *filename);

/** Collect results and disable profiling for a model
 *
 * @param filename[in] - file that contains profile information from nrt_profile_start
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_profile_stop(const char *filename);

#ifdef __cplusplus
}
#endif
