.. meta::
   :description: Learn about TorchNeuron, the native PyTorch backend for AWS Trainium. Covers eager mode, Adaptive Eager Execution, torch.compile, distributed training, NKI integration, and MoE support.
   :keywords: TorchNeuron, PyTorch, native backend, PrivateUse1, eager mode, Adaptive Eager Execution, torch.compile, distributed training, NKI, MoE, AWS Neuron, Trainium
   :date-modified: 2026-10-14

.. _about-pytorch-neuron:

====================
What is TorchNeuron?
====================

.. admonition:: Beta Release
   :class: note

   TorchNeuron is in public beta for the Neuron 2.33 release. It is supported on Trainium 2 (trn2)
   and Trainium 3 (trn3) instances. We welcome your feedback.

TorchNeuron is the open-source native PyTorch backend for AWS Trainium. It registers Trainium as a
standard PyTorch device, so a model written in standard PyTorch runs on it by setting the device to
``neuron``; code written against ``torch.accelerator`` needs no change. This topic covers
TorchNeuron's architecture, execution modes, and how it fits into the PyTorch ecosystem for
training and inference workloads.

TorchNeuron integrates through PyTorch's standard `PrivateUse1 <https://docs.pytorch.org/tutorials/advanced/privateuseone.html>`_ device backend mechanism and uses
standard PyTorch APIs. It supports eager mode execution, ``torch.compile`` for JIT compilation, and standard distributed APIs, and it supports
custom kernel development through the :ref:`Neuron Kernel Interface (NKI) <neuron-nki>`.

.. figure:: /images/torchneuron/torchneuron-portability-path.png
   :alt: A standard PyTorch model runs on AWS Trainium by setting the device to neuron, through the TorchNeuron PrivateUse1 backend.
   :align: center

   PyTorch model path to Trainium

Applies to
----------

This concept is applicable to:

- Training workloads on Trn2 and Trn3 instances
- Inference workloads on Trn2 and Trn3 instances

How TorchNeuron works
---------------------

TorchNeuron registers as a native device backend through PyTorch's ``PrivateUse1`` mechanism.
PyTorch's dispatcher routes tensor operations to the Neuron backend, which provides optimized
implementations of ``ATen`` operators (core tensor operations) and distributed communication
operators. These primitives execute directly on Trainium hardware. Forward and backward passes
are validated against PyTorch's upstream ``ATen`` test suites, and the results are published with
each release.

.. figure:: /images/torchneuron/torchneuron-stack.png
   :alt: TorchNeuron software stack: the PyTorch frontend and dispatcher, the Neuron PrivateUse1 and TorchDynamo backends, the Neuron Compiler, the Neuron Runtime, and AWS Trainium.
   :align: center

   The TorchNeuron software stack

TorchNeuron (``torch-neuronx`` for PyTorch 2.14) autoloads when running on Neuron hardware — no
explicit ``import torch_neuronx`` is needed. This uses PyTorch's
autoloading feature for out-of-tree extensions, which allows custom backends to register their
device automatically.

TorchNeuron supports two execution modes:

- **Eager mode**: Operations dispatch and execute immediately through the PyTorch dispatcher.
  Each operator compiles to its own NEFF (Neuron Executable File Format) as it runs. You can
  optionally enable Adaptive Eager Execution, which fuses queued operators for better performance.
- **torch.compile**: A custom TorchDynamo backend (``backend="neuron"``) captures FX graphs and
  compiles each graph to a single NEFF. The compilation pipeline is covered in
  the :ref:`torch.compile on Neuron <about-torchneuron-torch-compile>` section below.

For performance tuning, TorchNeuron provides three levels of control: run in eager mode with no
code change; apply ``torch.compile(backend="neuron")`` for graph-level optimization; and use NKI
kernels for fine-grained control over specific operators.

.. figure:: /images/torchneuron/torchneuron-optimization-ladder.png
   :alt: Three levels of control: eager mode, torch.compile with the neuron backend, and NKI kernels.
   :align: center
   :width: 420px

   Levels of performance optimization on Trainium

Training and inference
----------------------

TorchNeuron's eager and ``torch.compile`` backends serve both training and inference, independent
of the workload — the same package provides both backends, and neither is specific to training or
inference.

For serving, a model trained with TorchNeuron runs on the same ``torch.compile`` backend, whether
through the vLLM Neuron and vLLM Omni Neuron plugins or a custom native PyTorch inference stack.
See :ref:`Inference <about-torchneuron-vllm>` below.

Model and operator coverage
---------------------------

Many PyTorch model architectures run on TorchNeuron natively — including language,
vision, video, multimodal, and state-space models — because it implements the ``ATen`` operators
these models are built from. For the per-API support status, see the
`torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__. Hugging Face ``transformers``, ``diffusers``,
PEFT, and ``accelerate`` run on it with no Neuron-specific path. Coverage and fallback behavior
differ by execution path:

- In eager mode, an operator without a Neuron implementation runs on CPU and emits a
  warning, so the model still runs. See the `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__ for
  coverage details and how to detect fallbacks.
- Under ``torch.compile``, TorchDynamo inserts a graph break for constructs that it cannot
  capture. Each captured segment compiles independently. A captured operation that the Neuron
  backend cannot lower causes compilation to fail by default. See
  :doc:`/torch/compile/reference-torch-compile` for supported and unsupported surfaces.

Data types
----------

Set precision with ``torch.dtype`` as in standard PyTorch. TorchNeuron supports ``bfloat16``,
``float16``, ``float32``, complex types, and the integer and boolean types natively, and
``bfloat16`` is the training baseline. ``torch.autocast(device_type="neuron")`` is supported, so mixed-precision code
works as written. ``float64`` runs as ``float32``. ``int64`` runs natively for supported operations. FP8 and MXFP8 require Trainium 3.
For per-operator data type coverage, see the `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__.

Profiling
---------

Profile a model with the standard ``torch.profiler`` API. Neuron Explorer reads the capture and
shows the PyTorch calls with the underlying ATen operators and the corresponding device activity.
See :doc:`/tools/neuron-explorer/how-to-profile-pytorch`.

Eager mode execution
---------------------

In eager mode, operations are dispatched and execute immediately upon invocation, following
PyTorch's standard "define-by-run" paradigm. Each operation is dispatched one at a time through
PyTorch's dispatcher to the Neuron backend, which lowers it to StableHLO, compiles it to a NEFF,
and enqueues it for asynchronous execution on the device. A shape the compiler has not seen before
compiles on first use; after that the operator is served from the cache. Python control flow,
prints, and breakpoints work as they do on any other PyTorch device.

Adaptive Eager Execution
^^^^^^^^^^^^^^^^^^^^^^^^^

TorchNeuron implements *Adaptive Eager Execution* to improve performance and preserve
PyTorch stream order. Adaptive Eager Execution applies optimizations such as operator fusion.
Fusion can change intermediate precision and cause small numerical differences.

Neuron operations are enqueued asynchronously. The Python interpreter can issue later operations
while earlier operations run. TorchNeuron analyzes queued operations and fuses eligible sequences.
These fused operations run as one operator call and preserve the original execution order.

**Debugging and tensor inspection:**

Whenever you print a tensor, Neuron synchronizes at that operation and performs a device-to-host
copy. In the context of Adaptive Eager Execution, printing operations may determine fusion
boundaries — if printing occurs after an operation that would normally be fused with subsequent
operations, fusion will not happen at that point to ensure the requested tensor value is available.

Adaptive Eager Execution is off by default. To turn it on, set
``TORCH_NEURONX_ENABLE_CONCATENATION=1`` before you start your script. See :doc:`/torch/troubleshooting`.

**ATen operator implementations:**

ATen implementations and kernels are part of the Neuron backend for eager mode.

.. _about-torchneuron-torch-compile:
.. _about-torch-compile:

torch.compile on Neuron
------------------------

TorchNeuron provides native ``torch.compile`` support through a custom
`TorchDynamo <https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html>`__
backend for AWS Trainium, selected with ``torch.compile(model, backend="neuron")``. It lets you
JIT-compile some or all of your PyTorch code to improve performance on Trainium. Models and
input tensors are placed on the ``neuron`` device. For a quickstart, see
:doc:`/torch/get-started/quickstart-build-first-model`; for the exact supported and unsupported
parameters, see :doc:`/torch/compile/reference-torch-compile`.

How torch.compile works on Neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you apply ``torch.compile(backend="neuron")`` to a model or function, the following stages
execute:

#. **TorchDynamo** captures Python bytecode and extracts PyTorch operations into an FX graph.
#. **AOT Autograd** generates separate forward and backward graphs.
#. **Neuron backend** lowers the FX graphs to **StableHLO** MLIR (through a vendored
   ``torch_mlir`` importer). The lowered graph may include NKI kernels passed as custom ops.
#. **Neuron Compiler** (``neuronx-cc``) applies hardware-specific optimizations and produces the
   NEFF that executes on the hardware.

The Neuron backend provides native ``torch.compile`` support without using TorchInductor.

**Compile-only (device-free) compilation.** You can compile a graph to a NEFF without executing
it on a Neuron device by setting ``TORCH_NEURONX_COMPILE_ONLY=1`` or using
``torch_neuronx.compiler.compile_only()``. In this mode, operations compile but never run and a
Neuron runtime-initialization failure is non-fatal. On a host without a Neuron device, set
``NEURON_RT_FAKE_INSTANCE_TYPE`` to the target instance type.

Graph capture
^^^^^^^^^^^^^

The Neuron backend targets **full-graph** compilation; use ``fullgraph=True`` for best
performance. When TorchDynamo encounters a construct it does not capture (for example,
data-dependent control flow), it inserts a *graph break* and compiles each segment independently.
The model still runs correctly, but graph breaks reduce optimization opportunities because the
compiler has less context for each segment. Integrating NKI kernels into the compiled graph avoids graph breaks.
For the full parameter matrix and the operations that commonly cause graph breaks, see
:doc:`/torch/compile/reference-torch-compile`.

Caching and recompilation
^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch.compile()`` compiles during the first call for a new graph specialization. This call
traces, lowers, and compiles the graph. Later calls reuse the artifact when the graph and
compilation inputs match. Changes to shapes, dtypes, graph guards, compiler options, or the
lowered graph can cause another compilation. Compiled NEFFs are stored in a content-addressed,
on-disk persistent cache. The cache key uses the lowered HLO bytes, the compiler configuration,
and the compiler version. Point ``TORCH_NEURONX_NEFF_CACHE_DIR`` at a shared mount, such as Amazon
FSx, to reuse compiled NEFFs across instances. A second on-disk cache holds the lowered StableHLO
in ``TORCH_NEURONX_HLO_CACHE_DIR``. Eager mode and ``torch.compile`` share this cache. The PyTorch
Dynamo precompile cache is optional. See :doc:`/torch/compile/reference-torch-compile` for the
cache settings.

When to use torch.compile
^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose your execution mode based on your workflow:

* **Eager mode** for debugging, rapid iteration, prototyping, and inspecting intermediate tensor
  values. With Adaptive Eager Execution enabled, eager mode also delivers good performance without
  compilation overhead.
* **torch.compile** for production workloads, large models, and throughput optimization, where
  compilation overhead is amortized over many iterations.
* **Mixed mode** when combining eager execution with ``torch.compile`` on specific components —
  for example, running most of your model in eager mode but applying ``torch.compile`` to
  performance-critical components such as FlexAttention.

.. _about-torchneuron-vllm:

Inference
---------

For inference, you have two paths on Neuron, both built on the same native ``torch.compile``
lowering that TorchNeuron uses. Use the vLLM Neuron or vLLM Omni Neuron plugin when it supports
your model, or run custom native PyTorch inference without a plugin when you build your own
serving stack. A model trained with TorchNeuron serves on the same backend either way.

The plugins add the serving layer on top of the compile backend — an OpenAI-compatible API
server, continuous batching, a paged KV cache, and parallelism across NeuronCores — and call
``torch.compile`` for you when the server starts. For a custom stack, you call ``torch.compile``
yourself and implement the serving layer.

For an overview of the inference paths on Neuron, see
:ref:`Inference on AWS Neuron <inference-home>`. For running native PyTorch models for inference
without a plugin, see :ref:`Native PyTorch inference on Neuron <inference-native-pytorch>`.

Distributed training support
----------------------------

TorchNeuron supports PyTorch distributed APIs through the ``torch.distributed`` module for
collective communications across sharded models. Higher-level distributed training tools such as
FSDP and DTensor are implemented using these ``torch.distributed`` primitives.

The Trainium backend supports the following ``torch.distributed`` APIs and techniques:

* Fully Sharded Data Parallel (FSDP v1, v2, and SimpleFSDP) and Hybrid Sharded Data Parallel (HSDP)
* Distributed Tensor (DTensor) and ``DeviceMesh``
* Distributed Data Parallel (DDP)
* Tensor Parallelism (TP)
* Pipeline Parallelism (PP)

On Trainium, the unit of distribution is the NeuronCore, the compute unit of a Trainium chip. Configure sharding strategies based on available NeuronCores per Trainium chip,
which depends on model and workload requirements. For some parallelism strategies like Tensor
Parallelism, you need to specify how many NeuronCores are used for sharding. For other strategies
like FSDP, no configuration changes are needed.

**Activation checkpointing:** TorchNeuron supports activation checkpointing.

**Mixed precision:** Both the ``torch.autocast`` API and passing a mixed precision policy directly
to FSDP are supported. Automatic datatype conversion follows PyTorch CUDA conventions.

Use ``torch.distributed.pipelining`` to split a model into stages and run a
pipeline schedule such as ``ScheduleGPipe`` or ``Schedule1F1B``. Configure
EFA/RDMA for stage-to-stage communication, including on a single node. See
:ref:`Pipeline parallelism on Neuron <torchneuron-distributed-pipeline>` for
setup and an example.

NKI integration
----------------

TorchNeuron integrates with the Neuron Kernel Interface (NKI), which lets you write, optimize, and
run custom operators. NKI's definition and registration patterns are similar to Triton's.

NKI provides fine-grained control beyond Adaptive Eager Execution and ``torch.compile``.
Developers can call performance-critical NKI kernels within training code to replace sequences
of standard PyTorch operations. NKI kernels function in both eager and ``torch.compile`` modes:

* Immediate execution and debugging capabilities in eager mode for rapid iteration
* Graph-level optimizations with ``torch.compile`` for production deployment

NKI kernels integrate with native PyTorch code through the ``@nki.jit`` decorator and ``@nki_op``
for custom op registration. Training models that include NKI kernels requires a backward version
of the custom op, implemented using the
`register_autograd() <https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd>`__
function.

**When to use NKI:**

* For performance optimization requiring low-level hardware control
* For novel research requiring operations not yet expressible in standard PyTorch

.. note::

   ``@nki.jit`` kernels run without ``import torch_neuronx``, because ``import torch`` autoloads
   TorchNeuron. Import ``torch_neuronx`` only to call its own APIs, such as ``nki_op`` or
   ``wrap_nki``.

For the ``@nki_op`` and ``wrap_nki`` patterns, see :doc:`/torch/guides/how-to-work-with-nki`. For
details on NKI, see the :doc:`NKI documentation </nki/index>`.

MoE support
------------

TorchNeuron supports native PyTorch Mixture of Experts (MoE) operations, including
``torch._grouped_mm`` and MoE dispatch and combine operations. The supported MoE models are
DeepSeek, GPT-OSS, and Qwen3. To train MoE models on Neuron, use
:doc:`TorchTitan Neuron </training/torchtitan/index>`.

For how these operations behave under ``torch.compile``, see
:ref:`Graph breaks <torchneuron-graph-breaks>`.

Open source and development
----------------------------

TorchNeuron is open source under the Apache 2.0 license at
`aws-neuron/torch-neuronx <https://github.com/aws-neuron/torch-neuronx>`_ and follows a
GitHub-first development model; contributions are accepted through the repository's contributing
guide. It is an out-of-tree PyTorch backend, and the ``torch-neuronx`` wheel installs with ``pip``
from the Neuron pip repository (see :doc:`/torch/get-started/quickstart-torch-neuron`).

.. note::

   The XLA-based ``torch-neuronx`` library (versions before 2.14) remains available as a separate
   version lineage of the same package. TorchNeuron is the native PyTorch backend and the
   recommended path for new workloads on PyTorch 2.14.
