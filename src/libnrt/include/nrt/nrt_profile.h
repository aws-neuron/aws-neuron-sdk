/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include "nrt/nrt.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Enable profiling for a model
 *
 * @param model[in]     - model to profile
 * @param filename[in]  - output filename that will be used with nrt_profile_stop()
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_start(nrt_model_t *model, const char *filename);

/** Collect results and disable profiling for a model
 *
 * @param filename[in] - output filename to save the NTFF profile to
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_stop(const char *filename);

/** Options for continuous device profiling.
 *
 * Opaque struct used to preserve compatibility and enforce proper usage.
 * Use nrt_profile_continuous_options_set_* functions set options.
 * Default options:
 * - output_dir: "./output"
 *
 * Usage:
 *   nrt_profile_continuous_options_t *options;
 *   nrt_profile_continuous_options_allocate(&options);
 *   nrt_profile_continuous_options_set_output_dir(options, "./output");
 */
typedef struct nrt_profile_continuous_options nrt_profile_continuous_options_t;

/** Allocate memory for the nrt_profile_continuous_options_t struct and set all options to defaults.
 *
 * @param options[in] - pointer to a pointer to nrt_profile_continuous_options_t struct
 */
NRT_STATUS nrt_profile_continuous_options_allocate(nrt_profile_continuous_options_t **options);

/** Free up memory allocated for the options struct needed for continuous device profiling.
 *
 * @param options[in] - pointer to a nrt_profile_continuous_options struct
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_continuous_options_free(nrt_profile_continuous_options_t *options);

/** Sets the output directory for results of continuous device profiling.
 *
 * The filename is set automatically.
 *
 * @param[in,out] options Pointer to the options struct.
 * @param[in] output_dir Path to the output directory.
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_continuous_options_set_output_dir(nrt_profile_continuous_options_t *options, const char *output_dir);

/** @brief Start continuous device profiling.
 *
 * When continuous device profiling is started, profiling is enabled for every model but notifications
 * will only be serialized to disk when the user calls nrt_profile_continuous_save(). This gives
 * the user control over which profiles are saved to disk. When a profile is not saved, the overhead
 * of trace serialization and disk write is avoided. Continuous profiling is ideal for scenarios where you
 * only want to save profiles for specific executions. In this mode you do not need to call
 * nrt_profile_start() and nrt_profile_stop() because they are called internally. Continuous profiling
 * will not start if inspect device profiling is already enabled or async execution is enabled.
 *
 * @param options[in] - options to control continuous device profiling
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_continuous_start(nrt_profile_continuous_options_t *options);

/** Save NTFF profile to disk for the latest model executed on requested NeuronCore.
 *
 * Output directory will be set according to the options passed into this function. The filenames of
 * NTFFs within the output directory are chosen automatically to avoid conflicts. Calling save does
 * not stop continuous profiling.
 *
 * @param vnc[in]      - (start) NeuronCore id to collect profile for
 * @param options[in]  - options to control continuous device profiling
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_continuous_save(uint32_t vnc, nrt_profile_continuous_options_t *options);

/** Stops continuous device profiling.
 *
 * Calling stop does not save a profile.
 *
 * @return NRT_SUCCESS on success.
 */
NRT_STATUS nrt_profile_continuous_stop();

/* Begin tracing/profiling
 *
 * Users of this API must set options through environment variables:
 * 
 * - NEURON_RT_INSPECT_ENABLE: Set to 1 to enable system and device profiles.
 *   For control over which profile types are captured, use NEURON_RT_INSPECT_SYSTEM_PROFILE 
 *   and NEURON_RT_INSPECT_DEVICE_PROFILE.
 * - NEURON_RT_INSPECT_OUTPUT_DIR: The directory where captured profile data will be saved to.
 *   Defaults to ./output.
 * - NEURON_RT_INSPECT_SYSTEM_PROFILE: Set to 0 to disable the capture of system profiles. 
 *   Defaults to 1 when NEURON_RT_INSPECT_ENABLE is set to 1.
 * - NEURON_RT_INSPECT_DEVICE_PROFILE: Set to 0 to disable the capture of device profiles.
 *   Defaults to 1 when NEURON_RT_INSPECT_ENABLE is set to 1.
 * - NEURON_RT_INSPECT_ON_FAIL: Set to 1 to enable dumping of device profiles in case of an error 
 *   during graph execution. Defaults to 0.
 * 
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_begin();


/* Stop tracing/profiling and dump profile data.
 * Does nothing if `duration` is given to nrt_inspect_begin() and already elapsed
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_stop();


/** @brief Options for nrt_inspect_begin_with_options API.
 *
 * Opaque struct used to preserve compatibility and enforce proper usage.
 * Use nrt_inspect_config_set_* functions to set options or 
 * nrt_inspect_config_set_defaults to set use default options.
 *
 * Example Usage:
 *  nrt_inspect_config_t *options;
 *  nrt_inspect_config_allocate(&options);
 *  nrt_inspect_config_set_output_dir(options, "./output");
 */
typedef struct nrt_inspect_config nrt_inspect_config_t;


/** Allocate memory for the options structure which is needed to
 * start profiling using nrt_inspect_begin_with_options. This will set all options to defaults
 * 
 * @param options[out] - pointer to a pointer to options nrt_inspect_config struct
 * 
 */
NRT_STATUS nrt_inspect_config_allocate(nrt_inspect_config_t **options);

/** @brief all fields of the nrt_inspect_config structure to their default values.
 * 
 * Default behavior after calling this function:
 * - Session ID: 1
 * - Output directory: "./output" (when not explicitly set)
 * - Activity types: All activity types enabled (system_profile, device_profile, host_memory, cpu_util)
 * - System trace: All NeuronCores and event types enabled for capture
 * - Inspect mode: Disabled (profiles not captured automatically)
 * - Inspect on failure: Disabled (profiles not captured on execution failures)
 * 
 * @param options[in,out] - Pointer to an nrt_inspect_config structure.
 * 
 * @return NRT_SUCCESS on success
 * 
 * @note These default values set here are NOT influenced by the environment variables. 
 * If you are using the environment variables to set the values you do not need to use this method 
 * or any of the nrt_inspect_config_set_* functions.
 */
NRT_STATUS nrt_inspect_config_set_defaults(nrt_inspect_config_t *options);

/** Free up memory allocated for the options structure which is needed to
 * start profiling using nrt_inspect_begin_with_options
 * 
 * @param options[in] - pointer to an options nrt_inspect_config struct
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_config_free(nrt_inspect_config_t *options);

/**
 * @brief Sets the session ID for the nrt_inspect_config_t which is needed to
 * start profiling using nrt_inspect_begin_with_options
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] session_id Session ID to set.
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_config_set_session_id(nrt_inspect_config_t *options, int session_id);

/**
 * @brief Sets the output directory for results of 
 * profiling using nrt_inspect_begin_with_options
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] output_dir Path to the output directory. Must be a valid non-empty string 
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters, NRT_RESOURCE for memory allocation failure.
 * 
 * @note The function makes an internal copy of the string, so the caller
 *       does not need to keep the original string alive.
 * @note Call nrt_inspect_config_free() to properly clean up allocated memory.
 */
NRT_STATUS nrt_inspect_config_set_output_dir(nrt_inspect_config_t *options, const char *output_dir);

/**
 * @brief Sets max number of system trace events that can be stored across all ring buffers
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] sys_trace_max_events_per_nc Max number of system trace events that can be stored across all ring buffers.
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_config_set_sys_trace_max_events_per_nc(nrt_inspect_config_t *options, uint64_t sys_trace_max_events_per_nc);

/**
 * @brief Sets system trace capture enabled for a specific NeuronCore
 * ring buffers won't be allocated for disabled NeuronCores 
 * 
 * @param[in,out] options Pointer to the options structure.
 * @param[in] nc_idx Index of the NeuronCore.
 * @param[in] enabled Boolean value to enable or disable system trace capture.
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_inspect_config_set_capture_enabled_for_nc(nrt_inspect_config_t *options, uint32_t nc_idx, bool enabled);

/** 
 * @brief Sets system trace capture enabled for a specific event type
 * can save memory and reduce output size
 * @param[in,out] options Pointer to the options structure.
 * @param[in] event_type Valid event types.
 * @param[in] enabled Capture enabled flag.
 * @return NRT_SUCCESS on success
 * 
 * @note Event type must be a string from the list of supported event types. To get the list of supported event types, 
 * use nrt_sys_trace_get_event_types in the nrt_sys_trace.h header file.
 */
NRT_STATUS nrt_inspect_config_set_capture_enabled_for_event_type_string(nrt_inspect_config_t *options, const char *event_type, bool enabled);

/**
 * @brief Enable both system and device profiling for normal execution
 * 
 * When disabled (default), no profiles are captured during normal execution.
 * This flag controls whether profiles are captured automatically for each execution.
 * Note: If both enable_inspect and enable_inspect_on_fail are false, no profiling occurs.
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] enable_inspect Boolean value to enable or disable inspect profiling.
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters.
 */
NRT_STATUS nrt_inspect_config_set_enable_inspect(nrt_inspect_config_t *options, bool enable_inspect);

/**
 * @brief Enable dumping of device profiles in case of execution failures
 * 
 * When enabled, device profiles will be captured and saved when graph execution fails.
 * This is disabled by default. If both enable_inspect and enable_inspect_on_fail are false,
 * no profiling occurs at all.
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] enable_inspect_on_fail Boolean value to enable or disable inspect on failure.
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters.
 */
NRT_STATUS nrt_inspect_config_set_enable_inspect_on_fail(nrt_inspect_config_t *options, bool enable_inspect_on_fail);

 /**
 * Begin tracing/profiling with configurable options
 *
 * Parameters:
 * @param[in] options - A pointer to an nrt_inspect_config struct containing configuration options
 *                     for profiling. Use nrt_inspect_config_set_* functions to set options.
 *                     If NULL is passed, default options will be used.
 * @return NRT_SUCCESS on success
 * 
 * @note This API ignores all the NEURON_RT_INSPECT_* environment variables.
 * If you are using the environment variables to set the values you do not need to use this method 
 * or any of the nrt_inspect_config_set_* functions. Use nrt_inspect_begin() instead.
 */
NRT_STATUS nrt_inspect_begin_with_options(nrt_inspect_config_t *options);

/**
 * @brief Returns all available activity type strings
 *
 * This function allocates and returns an array of all supported activity type
 * strings. The caller is responsible for freeing both the individual strings
 * and the array itself, or can use nrt_inspect_config_free_activity_types().
 *
 * @param[out] activity_types Pointer to store the allocated array of activity type strings.
 * @param[out] count Pointer to store the number of activity types returned.
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters, 
 *         NRT_RESOURCE for memory allocation failure.
 */
NRT_STATUS nrt_inspect_config_get_all_activity_types(const char ***activity_types, size_t *count);

/**
 * @brief Returns the currently enabled activity type strings
 *
 * This function examines the enabled_activities bitmask in the configuration
 * and returns an array of strings for only the currently enabled activity types.
 * The caller is responsible for freeing both the individual strings and the array itself,
 * or can use nrt_inspect_config_free_activity_types().
 *
 * @param[in] options Pointer to the options structure.
 * @param[out] activity_types Pointer to store the allocated array of enabled activity type strings.
 * @param[out] count Pointer to store the number of enabled activity types returned.
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters, 
 *         NRT_RESOURCE for memory allocation failure.
 */
NRT_STATUS nrt_inspect_config_get_enabled_activity_types(nrt_inspect_config_t *options, const char ***activity_types, size_t *count);

/**
 * @brief Free the activity types array allocated by nrt_inspect_config_get_all_activity_types
 * or nrt_inspect_config_get_enabled_activity_types.
 * This function properly frees both the array and all individual strings.
 * 
 * @param[in] activity_types Pointer to the activity types array to be freed.
 * @param[in] count Number of activity types in the array.
 */
void nrt_inspect_config_free_activity_types(const char **activity_types, size_t count);

/**
 * @brief Sets or clears a specific activity type in the configuration
 *
 * This function enables or disables a specific activity type by name. It converts
 * the activity type string to the corresponding enum value and updates the
 * enabled_activities bitmask accordingly.
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] activity_type String name of the activity type. Valid values are:
 *                         "system_profile", "device_profile", "host_memory", 
 *                         "cpu_util", "all"
 * @param[in] enabled True to enable the activity, false to disable it.
 * @return NRT_SUCCESS on success, NRT_INVALID for invalid parameters or unknown activity type.
 */
NRT_STATUS nrt_inspect_config_set_activity(nrt_inspect_config_t *options, const char *activity_type, bool enabled);


#ifdef __cplusplus
}
#endif
