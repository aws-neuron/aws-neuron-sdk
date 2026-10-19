.. meta::
   :description: Conceptual overview of TorchTitan running on AWS Neuron with TorchNeuron. Covers the support matrix, architecture, supported models, and distributed strategies.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, PyTorch, FSDP, DTensor, Llama, DeepSeek, GPT-OSS, pre-training
   :date-modified: 2026-04-28

.. _about-torchtitan-on-neuron:

==================================
What is TorchTitan Neuron?
==================================

This topic covers TorchTitan Neuron and how it applies to pre-training with
the AWS Neuron SDK. TorchTitan is a PyTorch-native library for pre-training
large models. On Trainium it runs on **TorchNeuron**, the native PyTorch
backend for Neuron, so you can use TorchTitan's standard model definitions,
parallelism strategies, and ``torch.compile`` integration without changing
your model code.

TorchTitan expresses every parallelism strategy with standard PyTorch building
blocks: ``DeviceMesh``, DTensor, ``fully_shard`` (FSDP2), and
``torch.distributed`` collectives. TorchNeuron registers a ``neuron`` device and
a ``neuron`` process-group backend that implement these APIs. When you add
``--compile.enable --compile.backend neuron``, TorchTitan's compiled regions
are lowered by the Neuron compiler. Large-model pre-training on Trainium
therefore stays inside the PyTorch-native stack, with no separate framework or
graph-tracing layer to adopt.

Applies to
----------

This concept is applicable to:

- Pre-training workloads on Trn2 and Trn3 instances using TorchTitan.
- Distributed training using TorchTitan's built-in parallelism strategies on
  top of TorchNeuron's ``torch.distributed`` support.

Support matrix
--------------
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Supported version or value
   * - Neuron SDK
     - 2.33.0 (TorchNeuron public beta)
   * - PyTorch
     - 2.14.0
   * - ``torch-neuronx``
     - 2.14.3.0 package family
   * - Python
     - 3.12, 3.13, 3.14 (Linux x86_64)
   * - TorchTitan
     - Use `<https://github.com/aws-neuron/torchtitan-neuron>`_, the Neuron fork of upstream ``pytorch/torchtitan`` commit `7cdf0aa9 <https://github.com/pytorch/torchtitan/commit/7cdf0aa9>`_
   * - Instance types
     - Trn2 and Trn3. The TorchTitan tutorials use a single ``trn2.48xlarge``
       with 64 logical NeuronCores (one process per NeuronCore).
   * - Execution modes
     - Eager, and ``torch.compile`` with ``--compile.backend neuron``
       (static shapes only)

.. list-table:: Parallelism support
   :header-rows: 1
   :widths: 30 15 55

   * - Strategy
     - Status
     - Notes
   * - FSDP2 (``fully_shard``)
     - Supported
     - Default TorchTitan data-parallel sharding
       (``--parallelism.data-parallel-shard-degree``). Uses Neuron
       ``all_gather`` and ``reduce_scatter``.
   * - SimpleFSDP
     - Supported with ``jit`` backend
     - Compiler-friendly FSDP for ``torch.compile``. Available for Llama3 and
       DeepSeek-V3.
   * - Tensor parallel (TP)
     - Supported
     - DTensor ``ColwiseParallel``, ``RowwiseParallel``, and
       ``SequenceParallel``. The tutorials use TP=4.
   * - Context parallel (CP)
     - Supported
     - Ring attention for sequence-sharded SDPA, forward and backward.
   * - Expert parallel (EP)
     - Supported
     - Uneven ``all_to_all_single`` token dispatch through a native NKI
       all-to-all-v collective.
   * - Pipeline parallel (PP)
     - Supported by TorchNeuron
     - The Neuron backend implements point-to-point ``send`` and ``recv``.
       Not covered by the TorchTitan tutorials.
   * - DDP
     - Supported
     - Standard ``DistributedDataParallel``.
   * - Activation checkpointing
     - Supported
     - ``--activation-checkpoint.mode full`` and selective, including the
       per-op policy.


How TorchTitan works on Neuron
------------------------------
TorchTitan is a layer on top of the native PyTorch backend. It does **not** use
PyTorch/XLA, and it doesn't trace your model into a separate graph framework.
Model code, parallelism plans, the optimizer, and checkpointing are ordinary
PyTorch running on the ``neuron`` device.

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │ TorchTitan                                                   │
   │   models (Llama3, Qwen3, DeepSeek-V3, GPT-OSS, ...)          │
   │   parallelize: FSDP2 / SimpleFSDP, TP, CP, EP, AC            │
   │   train loop, optimizer, checkpoint, metrics                 │
   ├──────────────────────────────────────────────────────────────┤
   │ PyTorch 2.14                                                 │
   │   DeviceMesh · DTensor · fully_shard · torch.distributed     │
   │   torch.compile (Dynamo + AOTAutograd)                       │
   ├──────────────────────────────────────────────────────────────┤
   │ TorchNeuron (torch-neuronx)                                  │
   │   "neuron" device (PrivateUse1)   "neuron" process group     │
   │   eager ATen dispatch             DTensor sharding rules     │
   │   torch.compile backend="neuron"  NKI kernels (SDPA, ...)    │
   ├──────────────────────────────────────────────────────────────┤
   │ Neuron compiler (neuronx-cc) · Neuron runtime · collectives  │
   ├──────────────────────────────────────────────────────────────┤
   │ Trainium (Trn2 / Trn3) NeuronCores                           │
   └──────────────────────────────────────────────────────────────┘

A training run goes through four layers:

- **Process and device setup.** ``torchrun`` starts one process per
  NeuronCore. Set ``NEURON_RT_NUM_CORES`` and ``--nproc_per_node`` to the
  same value. TorchNeuron registers the ``neuron`` backend for
  ``init_process_group`` and assigns each rank its NeuronCore from
  ``LOCAL_RANK``.
- **Parallelism.** TorchTitan builds a ``DeviceMesh`` and applies its
  parallelism plan with DTensor and ``fully_shard``. TorchNeuron records every
  mesh's replica groups when the mesh is created, which SPMD compilation
  depends on. It also supplies sharding rules for attention and linear layers
  so that DTensor operations stay sharded on the device.
- **Execution.** Without ``--compile.enable``, every operation runs in eager
  mode. Each ATen op is compiled to a Neuron executable on first use and then
  cached. With ``--compile.enable --compile.backend neuron``, TorchTitan wraps
  part of the model in ``torch.compile``, and those regions are compiled by the
  Neuron compiler instead:

  - **Stock models with FSDP2:** each ``TransformerBlock`` is compiled
    separately with ``fullgraph=True``. FSDP2's parameter all-gather and
    gradient reduce-scatter run as eager collectives between the compiled
    blocks.
  - **SimpleFSDP:** the whole model is compiled as one ``fullgraph=True``
    graph. The FSDP collectives are inside the graph, so the compiler can
    schedule communication together with compute. This is why SimpleFSDP is
    the recommended FSDP choice with ``torch.compile``.

- **Caching.** Compiled executables are stored in a persistent cache keyed by
  the workload, the target, the logical NeuronCore (LNC) configuration, the
  compiler version, and the compiler options. The first step of a run can take
  several minutes to compile. Later steps, and later runs with the same
  configuration, reuse the cache.


Supported models
----------------

TorchTitan's model families run on TorchNeuron through the same model
definitions used on other hardware. The following configurations have
TorchNeuron tutorials:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Model
     - Workload
     - Validated configuration
   * - Llama3-8B
     - Pre-training
     - TP4 × FSDP16 on one ``trn2.48xlarge``. Eager, ``torch.compile``, and
       SimpleFSDP + ``torch.compile``.
   * - Qwen3-32B (dense)
     - Pre-training
     - TP4 × FSDP16, full activation checkpointing. Eager and ``torch.compile``.
   * - DeepSeek-V3
     - Pre-training
     - Mixture of experts (MoE) with multi-head latent attention.
       SimpleFSDP + ``torch.compile``.
   * - GPT-OSS
     - Pre-training
     - MoE with grouped experts.
   * - Qwen3-8B
     - Inference
     - Hugging Face ``transformers`` model with DTensor tensor parallelism and
       ``torch.compile(backend="neuron")``.

Other TorchTitan models (for example Llama4 and the Qwen3 MoE variants) use the
same building blocks. They are expected to run, but they don't have a
validated TorchNeuron configuration yet.


Distributed strategies
----------------------

TorchTitan composes its parallelism strategies on a multi-dimensional
``DeviceMesh``. Each rank of the mesh is one process, and each process owns one
logical NeuronCore. On a ``trn2.48xlarge`` with the default LNC=2, that is 64
ranks per node: 16 Trainium2 devices with 4 logical cores each.

**FSDP2 and SimpleFSDP.** Both shard parameters, gradients, and optimizer state
across the data-parallel shard dimension. FSDP2 (``fully_shard``) uses runtime
hooks around each module and works in eager mode and with per-block
compilation. SimpleFSDP expresses sharding as DTensor parametrizations, so the
all-gather and reduce-scatter are traced into the compiled graph. Use
SimpleFSDP when you combine FSDP with ``torch.compile``.

**Tensor parallelism (DTensor).** TP shards attention heads and MLP
projections. TorchNeuron provides DTensor sharding strategies for scaled
dot-product attention (replicated, head-sharded, batch-sharded, and
sequence-sharded) and for linear forward and backward, so a TP plan stays
sharded through attention.

**Context parallelism.** Sequence-sharded attention runs as ring attention. By
default, TorchNeuron uses a fused NKI ring-attention kernel that drives the
whole ring inside one kernel, and falls back to PyTorch's templated ring
otherwise.

**Expert parallelism.** MoE token dispatch uses ``all_to_all_single`` with
uneven split sizes. TorchNeuron implements this as a native NKI all-to-all-v
collective that sends exactly the requested per-rank counts without dynamic
shapes. Set ``NEURON_USE_NKI_ALLTOALLV=0`` to use the padding-based fallback.

**Mapping onto NeuronCores.** Collective cost depends on where the ranks of a
group sit. Ranks on the same Trainium device communicate fastest, then ranks
on the same node, then ranks across nodes. Put the most communication-heavy
dimension on the innermost mesh dimension:

- TP communicates in every layer, so keep each TP group on one device. For
  example, TP=4 matches the 4 logical NeuronCores of a Trainium2 device at
  LNC=2.
- FSDP communicates once per layer or per step and can span devices and nodes.
  ``--parallelism.data-parallel-shard-degree -1`` uses all remaining ranks.

.. code-block:: text

   trn2.48xlarge, 64 ranks, TP=4 × FSDP=16

   device 0             device 1                   device 15
   ┌─────────────────┐  ┌─────────────────┐        ┌─────────────────┐
   │ r0  r1  r2  r3  │  │ r4  r5  r6  r7  │  ...   │ r60 r61 r62 r63 │
   └─────────────────┘  └─────────────────┘        └─────────────────┘
       TP group 0           TP group 1                 TP group 15
   └────────────── FSDP shards across the 16 TP groups ──────────────┘

The LNC setting changes the number of ranks per node and is part of the
compilation cache key, so changing it triggers a recompile.


.. _torchtitan-neuron-limitations:

Known limitations
-----------------

The following limitations are specific to running TorchTitan on Neuron. For
general TorchNeuron constraints (operator coverage, data types, and
``torch.compile`` parameters), see the TorchNeuron API compatibility guide.

- **Static shapes only.** ``torch.compile`` on Neuron requires static shapes.
  Keep ``--training.seq-len`` and ``--training.local-batch-size`` fixed for a
  run. Each new input shape triggers a new compilation.
- **Long first step.** The first training step compiles every graph and can
  take several minutes. Throughput is steady after that. The persistent NEFF cache
  avoids paying this cost again on later runs.
- **Compile mode.** Only ``mode="default"`` is supported. TorchTitan's default
  is compatible.
- **Forced graph breaks under** ``fullgraph=True``. TorchTitan compiles with
  ``fullgraph=True``. TorchNeuron forces a graph break on operations Neuron
  can't lower in a compiled graph, and under ``fullgraph=True`` each of those
  becomes an error. This affects:

  - ``torch._grouped_mm`` / ``grouped_mm``, used by MoE grouped experts. It
    always forces a graph break; it runs on Neuron in eager mode.
  - ``sort`` / ``argsort`` on a dimension longer than 16,384, and ``topk``
    with ``k`` ≥ 16,384.

  For MoE models, keep the expert computation out of the compiled region, or
  run without ``--compile.enable``.
- **64-bit types.** ``float64`` and ``int64`` are computed as 32-bit types on
  the device. Keep training in ``bfloat16`` / ``float32``.
- **Pinned TorchTitan revision.** The tutorials are validated against
  `<https://github.com/aws-neuron/torchtitan-neuron>`_, the Neuron fork of upstream ``pytorch/torchtitan`` commit `7cdf0aa9 <https://github.com/pytorch/torchtitan/commit/7cdf0aa9>`_.
- **Profiling.** Neuron device profiling through TorchTitan's built-in profiler
  needs support for the PrivateUse1 profiler activity, which isn't in the pinned
  revision. The Llama3 tutorial passes ``--profiling.no-enable-profiling``. To profile,
  use ``torch.profiler`` with ``torch_neuronx.profiling.NeuronConfig`` and
  Neuron Explorer.

.. include:: /includes/torch/trn3-sizing-note.txt

Related concepts
----------------

- :doc:`/torch/about-pytorch-neuron` — What TorchNeuron is and how the native
  PyTorch backend works on Neuron.

Further reading
---------------

- :doc:`/training/torchtitan/get-started/setup-torchtitan-neuron` — Set up
  TorchTitan on a Neuron instance.
- :doc:`/training/torchtitan/tutorials/index` — TorchTitan training tutorials
  for Llama, DeepSeek, and GPT-OSS.
- :doc:`/training/torchtitan/best-practices` — Enablement and performance
  guidance.
