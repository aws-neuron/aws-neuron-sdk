.. _neuron-2-33-0-torchneuron:

.. meta::
   :description: Release notes for TorchNeuron (torch-neuronx), the native PyTorch backend for AWS Trainium, in AWS Neuron SDK 2.33.0.
   :keywords: TorchNeuron, torch-neuronx, PyTorch 2.13, PyTorch 2.14, release notes, Trainium, torch.compile, torch.distributed, NKI
   :date-modified: 10/19/2026

Release Notes for Neuron Component: TorchNeuron
===============================================

The release notes for ``torch-neuronx``, the native PyTorch backend for AWS Trainium.

.. _torchneuron-2-33-0-rn:

torch-neuronx 2.13.3.0 and 2.14.3.0 (Neuron 2.33.0 Release)
-----------------------------------------------------------

Date of release: 10/19/2026

Neuron 2.33.0 introduces the native ``torch-neuronx`` backend as a public beta.
It adds the ``neuron`` device through the PyTorch ``PrivateUse1`` extension.
Move the model and its inputs to ``neuron`` to run them on Trainium2 or Trainium3.
For an overview, see :doc:`/torch/about-pytorch-neuron`.

.. note::

   Install the ``torch-neuronx`` package family that matches the PyTorch version.
   For example, install ``torch-neuronx==2.14.*`` with PyTorch 2.14.

Support matrix
~~~~~~~~~~~~~~~~~~

TorchNeuron supports the following operating systems on Linux x86_64.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Ubuntu version
     - Support status
     - Supported instances
   * - Ubuntu 22.04
     - End of support
     - Not applicable
   * - Ubuntu 24.04
     - Supported
     - Trn2 and Trn3
   * - Ubuntu 26.04
     - Supported
     - Trn2 and Trn3

Neuron no longer publishes new Ubuntu 22.04 DLAMIs or container images.
Use Ubuntu 24.04 or Ubuntu 26.04 for a new environment.

Each package family supports all Python versions in its row.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - ``torch-neuronx`` package family
     - PyTorch version
     - Python versions
   * - ``2.13.3.0.*``
     - 2.13.0
     - 3.12, 3.13, and 3.14
   * - ``2.14.3.0.*``
     - 2.14.0
     - 3.12, 3.13, and 3.14

Each PyTorch and Python combination has a release wheel.

To install ``torch-neuronx`` and the matching dependencies, see
:doc:`Install and verify torch-neuronx
</torch/get-started/quickstart-torch-neuron>`.

New features
~~~~~~~~~~~~

* **Neuron device for PyTorch**: Move models and tensors with ``.to("neuron")``.
  Run supported ``torch.nn``, ``torch.autograd``, and ``torch.optim`` code.
  ``torch.accelerator`` discovers the Neuron device.
  In eager mode, an unsupported operation can run on the CPU.
  ``torch_neuronx.get_fallback_ops()`` lists recorded fallback operations.
  See the `torch-neuronx operator compatibility table <https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md>`__.

* **Data types**: Supported operations can use ``bfloat16``, ``float16``,
  ``float32``, ``int64``, complex types, other integer types, and ``bool``.
  ``torch.autocast`` supports mixed-precision execution on the ``neuron`` device.
  ``float64`` runs as ``float32`` and produces a warning.

* **Eager execution**: Operations run when Python calls them.
  Use standard control flow, ``print``, and breakpoints during development.
  Each operation compiles on first use and is cached for later calls.

* **torch.compile**: ``torch.compile(model, backend="neuron")`` compiles a
  model or selected regions. It also compiles supported backward graphs.
  Code outside a compiled region runs in eager mode.
  The backend supports ``mode="default"`` and static shapes.
  Set ``dynamic=False``.
  See :doc:`About torch.compile on Neuron
  </torch/compile/about-torch-compile>`.

* **torch.distributed**: This release provides a ``neuron`` c10d backend.
  Initialize it with ``init_process_group(backend="neuron")``.
  The backend supports ``all_reduce``, ``all_gather``, ``reduce_scatter``,
  ``all_to_all``, ``broadcast``, ``reduce``, ``gather``, ``scatter``, and
  ``barrier``.
  Point-to-point support includes ``send``, ``recv``, and ``batch_isend_irecv``.
  These operations support the following PyTorch parallelism APIs:

  * Data parallelism with ``DistributedDataParallel`` (DDP)
  * Fully sharded data parallelism with ``fully_shard`` (FSDP2)
  * Tensor and sequence parallelism with DTensor and ``DeviceMesh``
  * Pipeline parallelism with ``torch.distributed.pipelining``
  * Context parallelism with PyTorch's experimental ``torch.distributed.tensor.experimental.context_parallel()`` API for scaled dot-product attention

  You can combine supported APIs on one ``DeviceMesh``.
  Training runs on one node or many nodes.
  Collectives use Trainium collective cores within a node and EFA between nodes.
  See :doc:`How to use torch.distributed on Neuron
  </torch/guides/torch-distributed>`.

* **Streams and events**: ``torch.neuron.Stream`` and ``torch.neuron.Event``
  support device-work ordering and synchronization.
  Use ``wait_stream()`` or ``wait_event()`` to order dependent work.
  Use ``query()`` to check completion.
  Use ``synchronize()`` to wait for completion on the CPU.
  Streams are distinct by default, but they do not guarantee concurrent execution.
  Set ``TORCH_NEURONX_ENABLE_STREAMS=0`` before Python starts to use one stream.

* **Profiling and memory**: ``torch.profiler`` captures framework, runtime,
  and device traces for Neuron Explorer.
  A default Neuron session starts without additional configuration.
  Pass a ``NeuronConfig`` to select the captured information.
  ``torch_neuronx.memory_stats()``, ``memory_summary()``, ``empty_cache()``,
  and memory snapshots report and manage device memory.
  See the :doc:`profiling guide
  </tools/neuron-explorer/how-to-profile-workload>` and the
  :doc:`memory guide </torch/reference/guidelines-memory-management>`.

* **Compilation cache**: TorchNeuron caches compiled artifacts on disk.
  Processes and distributed ranks can reuse matching artifacts.
  ``torch_neuronx.compiler.compile_only()`` builds the cache without execution.
  See :doc:`About torch.compile on Neuron
  </torch/compile/about-torch-compile>`.

* **Custom kernels with NKI**: ``torch_neuronx.nki_op`` registers an NKI kernel
  as a PyTorch custom operation.
  The operation can run in eager mode and inside ``torch.compile``.
  Add a backward pass with ``torch.library.register_autograd``.
  Use profiling to identify operations that limit performance.
  Use NKI to implement or tune those operations.
  See :doc:`How to work with NKI kernels
  </torch/guides/how-to-work-with-nki>`.

Experimental features
~~~~~~~~~~~~~~~~~~~~~

* **Operation concatenation**: Combines eligible eager operations to reduce
  per-operation overhead.
  It is disabled by default.
  Set ``TORCH_NEURONX_ENABLE_CONCATENATION=1`` before Python starts.

Breaking changes
~~~~~~~~~~~~~~~~

* **Moving from the XLA-based torch-neuronx**: The native backend uses the
  ``neuron`` device instead of ``torch_xla`` and ``xm.xla_device()``.
  Update code written for the XLA-based library.

Known issues and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Dynamic shapes**: Native dynamic shape support is in development.
  In this release, ``torch.compile`` requires ``dynamic=False``, and each new input
  shape can require another compilation. Pad or bucket inputs to reduce shape variation.
* **First-run compilation**: The first operation or graph run includes
  compilation time when the cache has no matching artifact.
  Later runs can reuse the cache.
* **Unsupported APIs**: This release does not support ``torch.sparse``,
  ``torch.func``, ``torch.nested``, or ``torch.distributions``.
* **Limited torch.fft support**: Real-to-complex transforms, such as
  ``torch.fft.rfft``, run on the ``neuron`` device. Other ``torch.fft``
  transforms run on the CPU.
* **TorchAO and MXFP8**: This release does not include TorchAO integration or
  composite MXFP8 data type support.

Getting started
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch_neuronx

   model = torch.nn.Linear(16, 8).to("neuron").eval()
   inputs = torch.randn(4, 16, device="neuron")

   with torch.no_grad():
       eager_output = model(inputs)
       compiled_model = torch.compile(model, backend="neuron", dynamic=False)
       compiled_output = compiled_model(inputs)

   print(tuple(eager_output.cpu().shape))
   print(tuple(compiled_output.cpu().shape))

Both calls return a tensor with shape ``(4, 8)``.
