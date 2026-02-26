.. _api_nrt_profile_h:

nrt_profile.h
=============

Neuron Runtime Profiling API - Tools for profiling model execution and device performance.

**Source**: `src/libnrt/include/nrt/nrt_profile.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h>`_

Functions
---------

nrt_profile_start
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_profile_start(nrt_model_t *model, const char *filename);

Enable profiling for a model.

**Parameters:**

* ``model`` [in] - model to profile
* ``filename`` [in] - output filename that will be used with nrt_profile_stop()

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_profile.h:18 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L18>`_

nrt_profile_stop
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_profile_stop(const char *filename);

Collect results and disable profiling for a model.

**Parameters:**

* ``filename`` [in] - output filename to save the NTFF profile to

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_profile.h:26 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L26>`_

nrt_profile_continuous_start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_profile_continuous_start(nrt_profile_continuous_options_t *options);

Start continuous device profiling.

When continuous device profiling is started, profiling is enabled for every model but notifications will only be serialized to disk when the user calls nrt_profile_continuous_save().

**Parameters:**

* ``options`` [in] - options to control continuous device profiling

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_profile.h:77 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L77>`_

nrt_profile_continuous_save
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_profile_continuous_save(uint32_t vnc, nrt_profile_continuous_options_t *options);

Save NTFF profile to disk for the latest model executed on requested NeuronCore.

**Parameters:**

* ``vnc`` [in] - (start) NeuronCore id to collect profile for
* ``options`` [in] - options to control continuous device profiling

**Returns:** NRT_SUCCESS on success.

**Source**: `nrt_profile.h:91 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L91>`_

nrt_inspect_begin
^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_inspect_begin();

Begin tracing/profiling.

Users of this API must set options through environment variables (NEURON_RT_INSPECT_ENABLE, NEURON_RT_INSPECT_OUTPUT_DIR, etc.).

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_profile.h:118 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L118>`_

nrt_inspect_stop
^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_inspect_stop();

Stop tracing/profiling and dump profile data.

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_profile.h:126 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L126>`_

nrt_inspect_begin_with_options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_inspect_begin_with_options(nrt_inspect_config_t *options);

Begin tracing/profiling with configurable options.

**Parameters:**

* ``options`` [in] - A pointer to an nrt_inspect_config struct containing configuration options for profiling. If NULL is passed, default options will be used.

**Returns:** NRT_SUCCESS on success

**Note:** This API ignores all the NEURON_RT_INSPECT_* environment variables.

**Source**: `nrt_profile.h:237 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L237>`_

nrt_inspect_config_allocate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_inspect_config_allocate(nrt_inspect_config_t **options);

Allocate memory for the options structure which is needed to start profiling using nrt_inspect_begin_with_options.

**Parameters:**

* ``options`` [out] - pointer to a pointer to options nrt_inspect_config struct

**Returns:** NRT_SUCCESS on success

**Source**: `nrt_profile.h:149 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L149>`_

nrt_inspect_config_set_output_dir
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   NRT_STATUS nrt_inspect_config_set_output_dir(nrt_inspect_config_t *options, const char *output_dir);

Sets the output directory for results of profiling using nrt_inspect_begin_with_options.

**Parameters:**

* ``options`` [in,out] - Pointer to the options structure.
* ``output_dir`` [in] - Path to the output directory. Must be a valid non-empty string

**Returns:** NRT_SUCCESS on success, NRT_INVALID for invalid parameters, NRT_RESOURCE for memory allocation failure.

**Source**: `nrt_profile.h:180 <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_profile.h#L180>`_
