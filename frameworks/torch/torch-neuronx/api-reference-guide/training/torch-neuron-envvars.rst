.. _pytorch-neuronx-envvars:

PyTorch Neuron Environment Variables (``torch-neuronx``)
========================================================

Environment variables allow modifications to PyTorch Neuron behavior
without requiring code change to user script. It is recommended to set
them in code or just before invoking the python process, such as
``NEURON_FRAMEWORK_DEBUG=1 python3 <script>`` to avoid inadvertently
changing behavior for other scripts. Environment variables specific to
PyTorch Neuron are (beta ones are noted):

``NEURON_CC_FLAGS``:

-  Compiler options. Full compiler options are described in the :ref:`mixed-precision-casting-options`.
   Additional options for the Neuron
   Persistent Cache can be found in the :ref:`Neuron Persistent Cache <neuron-caching>` guide.

``NEURON_FRAMEWORK_DEBUG`` **[Beta]**:

-  Enable dumping of XLA graphs in both HLO format (intermediate representation) and text form for debugging.

``NEURON_EXTRACT_GRAPHS_ONLY`` **[Beta]**:

-  Dump the XLA graphs in HLO format (intermediate representation) and execute empty stubs with zero outputs
   in order to allow multiple XLA graphs to be traced through a trial execution.
   Used automatically for ahead-of-time
   graph extraction for parallel compilation in :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`
   tool. This environment variable can be checked in the training script
   to prevent checking of bad outputs during trial run.

``NEURON_NUM_RECENT_MODELS_TO_KEEP`` **[Beta]**:

-  Keep only N number of graphs loaded in Neuron runtime for each
   process, where N is the value this environment variable is set to.
   Default is to keep all graphs loaded by a process.

``NEURON_COMPILE_CACHE_URL``

-  Set the :ref:`Neuron Persistent Cache <neuron-caching>` URL or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
   If starts with ``s3://``, it will use AWS S3 as cache backend. Otherwise it will use
   local disk cache. Default is ``/var/tmp/neuron-compile-cache``.
   If this is specified together with ``cache_dir=<cache_url>`` option via ``NEURON_CC_FLAGS``, the ``--cache_dir`` option takes precedence.

``NEURON_PARALLEL_COMPILE_MAX_RETRIES`` **[Beta]**:

-  Set the maximum number of retries when using :ref:`Neuron Persistent Cache <neuron-caching>` or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
   If set to N, the tool will try compilation N more time(s) if the first graph compilation failed.
   Example: Set NEURON_PARALLEL_COMPILE_MAX_RETRIES=1 when precompiling on 
   trn1.2xlarge where there's limited host memory and CPU resources.
   Default is 0.

``NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE`` **[Beta]**:

- When using :ref:`Neuron Persistent Cache <neuron-caching>` or :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>` , if you want to ignore the error in training script
  and compile the accumulated HLO graphs, you can do so by setting this environment variable.
  Example: If NEURON_IGNORE_TRAINING_SCRIPT_ERROR_AND_COMPILE=1 is set when using ``neuron_parallel_compile``,
  a crash in the training script would be ignored and the graphs collected upto the crash would be
  compiled.

``NEURON_DUMP_HLO_SNAPSHOT`` **[Beta]**:

- Dump the inputs, outputs, and graph in HLO format of a graph execution in a snapshot file. This
  variable can be set to ``1``, ``ON_NRT_ERROR``, ``ON_NRT_ERROR_CPU``, ``ON_NRT_ERROR_HYBRID`` to
  dump snapshots at every iteration using CPU memory, or dump only on errors automatically using
  device, host, and both device and host memory respectively.

``NEURON_NC0_ONLY_SNAPSHOT`` **[Beta]**:

- Dump only the snapshot associated with Neuron Core 0 when ``NEURON_NC0_ONLY_SNAPSHOT=1`` and 
  the ``NEURON_DUMP_HLO_SNAPSHOT`` flag is set.

``NEURON_FUSE_SOFTMAX`` **[Beta]**:

- Enable custom lowering for Softmax operation to enable compiler optimizations.

``NEURON_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING`` **[Beta]**:

- When set to 1, mark all parameter transfers as static to enable runtime optimizations for torch.nn modules that are wrapped as done in Megatron-LM. This setting is not needed if torch.nn modules are not wrapped.

``BUCKET_CAP_MB`` **[PyTorch XLA]**:

- If there are many parameters, such as in BERT training, small allreduce sizes can limit performance. To improve performance, you can try increasing the bucket size using ``BUCKET_CAP_MB`` environment variable, which is set to 50MB by default. For example, BERT pretraining on multiple instances can see improved performance with ``BUCKET_CAP_MB=512``.

``XLA_USE_BF16`` **[PyTorch XLA]**:

- When ``XLA_USE_BF16=1``, PyTorch Neuron will automatically map both torch.float and torch.double tensors
  to bfloat16 tensors and turn on Stochastic Rounding mode. This can both reduce memory footprint and improve performance.
  Example: to enable bfloat16 autocasting and stochastic rounding, set XLA_USE_BF16=1 only, as
  stochastic rounding mode is on by default when XLA_USE_BF16=1. If you would like to preserve some tensors in float32, see ``XLA_DOWNCAST_BF16`` below.

``XLA_DOWNCAST_BF16`` **[PyTorch XLA]**:

- When ``XLA_DOWNCAST_BF16=1``, PyTorch Neuron will automatically map torch.float tensors to bfloat16 tensors, torch.double tensors
  to float32 tensors and turn on Stochastic Rounding mode. This can both reduce memory footprint and improve performance, while preserving some tensors in float32.
  Example: to enable float to bfloat16 and double to float autocasting and stochastic rounding, set XLA_DOWNCAST_BF16=1 only, as
  stochastic rounding mode is on by default when XLA_DOWNCAST_BF16=1. If you want to cast both torch.float and torch.double to bfloat16, please see ``XLA_USE_BF16`` above.

``NEURON_RT_STOCHASTIC_ROUNDING_EN`` **[Neuron Runtime]**:

- When ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``, PyTorch Neuron will use stochastic rounding instead of
  round-nearest-even for all internal rounding operations when casting from FP32 to a reduced precision data type (FP16, BF16, FP8, TF32).
  This feature has been shown to improve
  training convergence for reduced precision training jobs, such as when bfloat16 autocasting is
  enabled. This is set to 1 by default by PyTorch Neuron when XLA_USE_BF16=1 or XLA_DOWNCAST_BF16=1. To switch to round-nearest-even mode, please set ``NEURON_RT_STOCHASTIC_ROUNDING_EN=0``.

``NEURON_RT_STOCHASTIC_ROUNDING_SEED`` **[Neuron Runtime]**:

- Sets the seed for the
  random number generator used in stochastic rounding (see previous section). If this environment variable is not set, the seed is set to 0 by default. Please set ``NEURON_RT_STOCHASTIC_ROUNDING_SEED`` to a fixed value to ensure reproducibility between runs.

``NEURON_RT_VISIBLE_CORES`` **[Neuron Runtime]**:

  Integer range of specific NeuronCores needed by the process (for example, 0-3 specifies NeuronCores 0, 1, 2, and 3).
  You this environment variable when using torchrun to limit the launched processs to specific consecutive NeuronCores. To ensure best performance, the multi-core jobs requiring N NeuronCores for collective communication must be placed at the NeuronCore ID that starts at a multiple of N, where N is the world size limited to 1, 2, 8, 32. For example, a process using 2 NeuronCores can be mapped to 2 free NeuronCores starting at NeuronCore id 0, 2, 4, 6, etc, and a process using 8 NeuronCores can be mapped to 8 free NeuronCores starting at NeuronCore id 0, 8, 16, 24.

Additional Neuron runtime environment variables are described in `runtime
configuration
documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-runtime/nrt-configurable-parameters.html>`__.

Additional XLA runtime environment variables are described in `PyTorch-XLA troubleshooting guide
<https://github.com/pytorch/xla/blob/v1.10.0/TROUBLESHOOTING.md#user-content-environment-variables>`__.
