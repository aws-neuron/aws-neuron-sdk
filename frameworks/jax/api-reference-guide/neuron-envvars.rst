.. _jax-neuronx-envvars:

JAX NeuronX Environment Variables
======================================

Environment variables allow modifications to JAX NeuronX behavior
without requiring code change to user script. It is recommended to set
them in code or just before invoking the python process, such as
``NEURON_RT_VISIBLE_CORES=8 python3 <script>`` to avoid inadvertently
changing behavior for other scripts. Environment variables specific to
JAX Neuronx are:

``NEURON_CC_FLAGS``

-  Compiler options. Full compiler options are described in the :ref:`mixed-precision-casting-options`.

``XLA_FLAGS``

- When set to ``"--xla_dump_hlo_snapshots --xla_dump_to=<dir>"``, this environmental variable enables dumping snapshots in ``<dir>`` directory. See :ref:`torch-neuronx-snapshotting` section for more information. The snapshotting interface for JAX and Pytorch are identical.
- When set to ``"--xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=<dir> --xla_dump_hlo_pass_re='.*'"``, this environmental variable enables dumping HLOs in proto and text formats after each XLA pass. The dumped ``*.hlo.pb`` files are in HloProto format.

``NEURON_FORCE_PJRT_PLUGIN_REGISTRATION``

- When ``NEURON_FORCE_PJRT_PLUGIN_REGISTRATION=1``, the Neuron PJRT plugin will be registered in JAX regardless of the instance type.

``NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU``

-  When ``NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1``, the Neuron PJRT plugin will compile and execute "trivial" computations on CPU instead of Neuron cores. A "trivial" computation is defined as an HLO program that does not contain any collective-compute instructions. The HLO program will be compiled by the XLA CPU compiler and outputs of the computation will be allocated on Neuron cores. The following HLO instructions are considered as collective-compute instructions.

    - ``all-gather``
    - ``all-gather-done``
    - ``all-gather-start``
    - ``all-reduce-done``
    - ``all-reduce-start``
    - ``all-to-all``
    - ``collective-permute``
    - ``partition-id``
    - ``replica-id``
    - ``recv``
    - ``recv-done``
    - ``reduce-scatter``
    - ``send``
    - ``send-done``

``NEURON_PJRT_PROCESSES_NUM_DEVICES``

- Should be set to a comma-separated list stating the number of NeuronCores used by each worker process. It is used to construct a global device array with its size equal to the sum of the list. This gets reported to the XLA PJRT runtime when requested. Must be set for multi-process executions. It can be used in conjunction with ``NEURON_RT_VISIBLE_CORES`` to expose a limited number of NeuronCores to each worker process. If ``NEURON_RT_VISIBLE_CORES`` is not set, it should be set to available number of NeuronCores on the host. ``NEURON_PJRT_PROCESSES_NUM_DEVICES`` must be less than or equal to ``NEURON_RT_VISIBLE_CORES``.

``NEURON_PJRT_PROCESS_INDEX``

- An integer stating the index (or rank) of the current worker process. This is required for multi-process environments where all workers need to know information on all participating processes. Must be set for multi-process executions. The value should be between ``0`` and ``NEURON_PJRT_PROCESS_INDEX - 1``.

``NEURON_RT_STOCHASTIC_ROUNDING_EN`` **[Neuron Runtime]**

- When ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1``, JAX Neuron will use stochastic rounding instead of
  round-nearest-even for all internal rounding operations when casting from FP32 to a reduced precision data type (FP16, BF16, FP8, TF32).
  This feature has been shown to improve
  training convergence for reduced precision training jobs. 
  To switch to round-nearest-even mode, set ``NEURON_RT_STOCHASTIC_ROUNDING_EN=0``.

``NEURON_RT_STOCHASTIC_ROUNDING_SEED`` **[Neuron Runtime]**

- Sets the seed for the random number generator used in stochastic rounding (see previous section). If this environment variable is not set, the seed is set to 0 by default. Please set ``NEURON_RT_STOCHASTIC_ROUNDING_SEED`` to a fixed value to ensure reproducibility between runs.

``NEURON_RT_VISIBLE_CORES`` **[Neuron Runtime]**

- Integer range of specific NeuronCores needed by the process (for example, 0-3 specifies NeuronCores 0, 1, 2, and 3). Use this environment variable when launching processes to limit the launched process to specific consecutive NeuronCores.

Additional Neuron runtime environment variables are described in :ref:`nrt-configuration`.
