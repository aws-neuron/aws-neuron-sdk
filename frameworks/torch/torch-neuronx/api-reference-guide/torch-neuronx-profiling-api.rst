.. _torch-neuronx-profiling-api:

PyTorch Neuron (``torch-neuronx``) Profiling API
================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

The profiler provides a method to generate a context manager to capture
trace events at the operator or runtime level.

.. py:function:: torch_neuronx.experimental.profiler.profile(port=9012,ms_duration=60000,neuron_tensorboard_plugin_dir="logs/plugins/neuron",profile_type="operator",auto_start=True,delete_working=True)

   The :func:`torch_neuronx.experimental.profiler.profile` method retuns a ``profile`` context manager object. This object
   doesn't need to be used directly, as default options are set to auto capture events based on the ``profile_type``.

   The context manager will wrap around the entire model
   and training/inference loop. The context-manager is 
   backwards-compatible with the torch_xla.debug.profiler``

   *Required Arguments*

   None

   *Optional Keyword Arguments*

   :keyword int port: Port to run the profiling GRPC server on. Default is 9012.
   :keyword int ms_duration: This defines how long the profiler will capture the
      HLO artifacts from the model to view in the profiler. The unit is in
      milliseconds. The default value is 60000 ms, or 1 minute.
   :keyword str neuron_tensorboard_plugin_dir: The directory the neuron tensorboard plugin will file write to.
      This will be ``logs/plugins/neuron`` by default/
   :keyword str profile_type: There is “trace” and “operator”. “trace”
      is the Torch Runtime Trace Level, while “operator” is the Model
      Operator Trace Level. Default is "operator"
   :keyword bool auto_start: If set to true, the profiler will start profiling immediately.
      If set to false, the profiler can be set to start at a later condition.
      Refer to ``profile.start()`` for more details. Default is ``True``.
   :keyword bool delete_working: If set to False turns off the deletion of temporary files. Default True.

   :returns: The traced :class:`profile`

   :rtype: ~profile

.. py:function:: torch_neuronx.experimental.profiler.profile.start()

   The :func:`torch_neuronx.experimental.profiler.profile.start` method starts the profiler if not started (i.e when ``auto_start=False``).
   This function does not take in any parameters, nor return anything.

    *Required Arguments*

   None

    *Optional Keyword Arguments*

   None

   :returns: None