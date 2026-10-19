.. meta::
   :description: How to optimize a PyTorch model for throughput on AWS Trainium with torch.compile(backend="neuron"). Covers full-graph compilation, graph breaks and recompilation, graph partitioning, NKI kernels, numerical validation, NEFF cache reuse, and measuring compilation and step time with TorchNeuron.
   :keywords: TorchNeuron, torch.compile, backend neuron, fullgraph, NEFF cache, optimize, throughput, best practices, graph breaks, recompilation, mixed mode, NKI, AWS Neuron, Trainium
   :date-modified: 2026-10-14

.. _torch-compile-optimize-howto:

===============================================
How to optimize a model with torch.compile
===============================================

Task overview
-------------

This topic discusses how to apply ``torch.compile(backend="neuron")`` to a working
eager-mode model on AWS Trainium and tune it for throughput.

Prerequisites
-------------

- **A working eager-mode model on Neuron:** See
  :ref:`torch-get-started`. Model and inputs are on the ``neuron`` device.
- **Concepts:** Skim :ref:`torch-compile-deep-dive` for pipeline and caching context.

Instructions
------------

.. _torch-compile-optimize-step-1:

**1:** Wrap your model or callable with the Neuron backend

Apply the backend with the functional form:

.. code-block:: python

   import torch

   compiled = torch.compile(model, backend="neuron", fullgraph=True)

Or with the decorator form on a function:

.. code-block:: python

   @torch.compile(backend="neuron", fullgraph=True)
   def compiled_forward(x):
       return model(x)

Pass ``backend="neuron"`` and remove CUDA-only settings such as CUDA graphs and
``mode="max-autotune"``. The backend supports ``mode="default"`` only; other modes
emit a warning and fall back to default. Dynamic shapes are not supported, so leave
``dynamic`` unset or pass ``dynamic=False``. Use ``fullgraph=True`` so graph breaks
are raised rather than silently splitting the model.

.. _torch-compile-optimize-step-2:

**2:** Warm up, then time steady-state iterations

The first call with a given input signature triggers compilation, which can take
several minutes for large models. Later calls with the same shapes and dtypes reuse the
compiled NEFF. Time a whole training iteration, not only the compiled forward pass,
because host work and the optimizer do not appear in a timer that covers the forward
and backward passes alone. Run a few warmup iterations, then time the steady-state
iterations:

.. code-block:: python

   import time
   import torch_neuronx

   compiled = torch.compile(model, backend="neuron", fullgraph=True)

   def train_step(batch):                     # one full training iteration
       optimizer.zero_grad()
       loss = loss_fn(compiled(batch["input"]), batch["target"])
       loss.backward()
       optimizer.step()
       return loss

   for batch in warmup_batches:               # warm up: triggers and caches compilation
       train_step(batch)

   torch_neuronx.synchronize()
   start = time.perf_counter()
   for batch in timed_batches:
       train_step(batch)
   torch_neuronx.synchronize()
   iteration_ms = (time.perf_counter() - start) * 1000 / len(timed_batches)

Synchronize the device before you read the clock. Neuron operations run
asynchronously, so a timer without a synchronize measures only the time to enqueue
the work.

.. _torch-compile-optimize-step-3:

**3:** Reuse compiled NEFFs across runs

Compiled NEFFs are cached on disk (``TORCH_NEURONX_NEFF_CACHE_DIR``, default
``/tmp/neff_cache``). To reuse them across runs and across instances, point the cache
directory at a shared mount such as Amazon FSx, and keep the lock and scratch
directory on a local disk:

.. code-block:: bash

   export TORCH_NEURONX_NEFF_CACHE_DIR=/fsx/neuron/neff_cache
   export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/tmp/local_cache

``TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`` (default ``/tmp/local_cache``) holds the
backend's file locks and scratch files only. The cached NEFFs themselves stay in
``TORCH_NEURONX_NEFF_CACHE_DIR``. Keep the lock directory on a local POSIX
filesystem, because file locks are not reliable on NFS or Lustre.

The cache key includes the compiler version,
the target platform, the logical NeuronCore configuration, and the compiler flags
(``NEURON_CC_FLAGS``). A change to any of these makes a new key, so pin
``torch-neuronx`` and the compiler in production. See :ref:`torch-compile-ref` for
all cache-directory variables.

.. _torch-compile-optimize-step-4:

**4:** Find graph breaks and recompilations

Graph breaks and recompilations are the most common reasons a compiled model runs
slower than expected. Use the standard PyTorch logging to list them:

.. code-block:: bash

   TORCH_LOGS="graph_breaks,recompiles" python train.py

Each graph-break entry names the source line and the reason, for example a
``print()`` call, a ``.item()`` call, or a Python builtin that Dynamo cannot trace.
Each recompilation entry names the guard that failed, for example a changed input
shape or a changed Python attribute. Fix these before you tune anything else. See
:ref:`best practices 1 <torch-compile-bp-1>`, :ref:`2 <torch-compile-bp-2>`,
and :ref:`3 <torch-compile-bp-3>`.

.. _torch-compile-optimize-step-5:

**5:** Measure compilation

Use the backend metrics API to see how many graphs compiled and how long each stage
took:

.. code-block:: python

   from torch_neuronx.neuron_dynamo_backend import (
       get_dynamo_metrics,
       reset_dynamo_metrics,
   )

   reset_dynamo_metrics()
   compiled = torch.compile(model, backend="neuron", fullgraph=True)
   compiled(*example_inputs)          # first call triggers compilation

   for m in get_dynamo_metrics():
       print(m.graph_name, m.graph_node_count, m.torch_neuronx_compile_us)

Each entry is one compiled graph, named ``{model_name}_{aot_id}_{graph_type}_{nth}``.
For a training step, each compiled region produces a forward graph and a backward
graph. More graphs than you expect means that a graph break split a region. To export
the metrics as CSV at process exit, set
``TORCH_NEURONX_ENABLED_METRIC_TABLES=graph_stats`` (and optionally
``TORCH_NEURONX_METRICS_DIR``).

Confirm your work
-----------------

Confirm that the model compiled once and that later calls reuse the compiled graph.
Reset the metrics, run the model twice, and check the compiled-graph count each time:

.. code-block:: python

   reset_dynamo_metrics()
   compiled(*example_inputs)                       # first run
   first = len(get_dynamo_metrics())

   reset_dynamo_metrics()
   compiled(*example_inputs)                       # second run, same shapes
   second = len(get_dynamo_metrics())

   assert first >= 1 and second == 0, (first, second)

The first run compiles at least one graph. The second run with the same input shapes
compiles nothing, which confirms that the compiled graph was reused. If the second
count is not zero, the inputs changed shape or dtype between the calls.

To confirm on-disk cache reuse across processes, run the script a second time. The
second process starts much faster because it loads the NEFFs from
``TORCH_NEURONX_NEFF_CACHE_DIR`` instead of compiling them again.

Common issues
-------------

Uh oh! Did you encounter an error or other issue while working through this
task? Here are some commonly encountered issues and how to address them.

.. rubric:: The model silently runs slower than expected

- **Possible solution**: The backend may have fallen back to eager on a compilation
  failure. See the fallback section of :ref:`torch-compile-deep-dive`,
  :ref:`best practice 10 <torch-compile-bp-10>`, and :ref:`torch-troubleshooting`.

.. rubric:: The model recompiles every iteration

- **Possible solution**: Input shapes are changing. See
  :ref:`best practice 2 <torch-compile-bp-2>`.

.. rubric:: The model recompiles on the second or third iteration only

- **Possible solution**: Python state that the compiled code reads changed after the
  first call. See :ref:`best practice 3 <torch-compile-bp-3>`.

.. rubric:: A small code change makes one region much slower

- **Possible solution**: The change added a graph break inside the region. Find it
  with :ref:`step 4 <torch-compile-optimize-step-4>`. See
  :ref:`best practice 1 <torch-compile-bp-1>`.

.. rubric:: The compiled model gives different results from eager mode

- **Possible solution**: Disable dropout and compare again. See
  :ref:`best practice 8 <torch-compile-bp-8>`.

.. rubric:: A larger batch size does not increase throughput

- **Possible solution**: Step time can grow linearly with batch size. See
  :ref:`best practice 6 <torch-compile-bp-6>`.

.. _torch-compile-best-practices:

Best practices
--------------

These best practices target fast steady-state throughput (after compilation),
effective cache reuse, and predictable compilation behavior from the Neuron
TorchDynamo backend.

In these practices, a *region* is a module or function that you pass to
``torch.compile``. A region without graph breaks compiles to one graph, and a graph
break splits it into more than one graph.

General principles
~~~~~~~~~~~~~~~~~~

- Validate correctness in eager mode first, then apply ``torch.compile`` for performance.
- Keep each region free of graph breaks (``fullgraph=True``) and give it fixed input shapes.
- Choose how many regions to compile by measurement, not by assumption. See :ref:`best practice 5 <torch-compile-bp-5>`.
- Measure every change on the full training iteration. The sum of isolated measurements is not equal to the full-iteration result.
- Keep the default fail-loud behavior in production.

Compilation best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-1:

**1:** Minimize graph breaks

    * Follow this practice when: Using ``torch.compile(backend="neuron")`` on any model.

Each graph break splits a region into separately compiled graphs, which reduces
the scope the compiler can optimize across. Common causes include data-dependent
control flow, unsupported operators, and calls into code Dynamo cannot trace. Calls
that run only during tracing also break the graph, for example a ``print()`` or
logging call in a helper that builds a configuration object on its first call.
Refactor those paths, or move them out of the compiled region.

To read a scalar from the device, such as a loss value or a counter, read it outside
the compiled region. ``.item()`` inside the region breaks the graph and synchronizes
the host with the device. NKI kernels registered as PyTorch operators integrate into
the graph without forcing a break.

TorchNeuron also forces a graph break on operations that Neuron cannot lower in a
compiled graph. With ``fullgraph=True``, each of these raises an error instead:

- ``torch.sort`` and ``torch.argsort`` when the sorted dimension is longer than 16,384.
- ``torch.topk`` when ``k`` is 16,384 or more. The limit applies to ``k``, not to the
  size of the dimension you select from.
- ``torch.nn.functional.grouped_mm`` and ``torch._grouped_mm``, always.
- Any other operation that has no Neuron lowering, always. TorchNeuron builds this
  list when it is imported, from the installed PyTorch, torch-mlir, and Neuron
  decompositions.

Keep sorts and top-k selections under these limits, and move the other operations out
of the hot compiled region. For other operations that commonly cause graph breaks, see
:ref:`torchneuron-graph-breaks`.

.. _torch-compile-bp-2:

**2:** Keep input shapes static to avoid recompilation

    * Follow this practice when: Running compiled models with variable batch or sequence sizes.

Each new input shape or dtype triggers a new compilation, which stalls the step that
sees it. Limit the shapes the model sees:

- Use a fixed batch size. For a ``DataLoader``, set ``drop_last=True`` so the last
  batch is not smaller than the others.
- Pad variable-length inputs to a small set of bucket sizes, and pass a mask for the
  padding. Each bucket compiles once.

.. code-block:: python

   import torch.nn.functional as F

   BUCKETS = (128, 256, 512, 1024)

   def pad_to_bucket(input_ids):              # input_ids: [batch, seq_len]
       seq_len = input_ids.shape[-1]
       target = next(b for b in BUCKETS if b >= seq_len)
       return F.pad(input_ids, (0, target - seq_len))

Run each bucket once during warmup, so that no compilation occurs in the timed steps.
The graph count from ``get_dynamo_metrics()`` should match the number of buckets.

.. _torch-compile-bp-3:

**3:** Initialize lazy state before the first compiled call

    * Follow this practice when: The model or its custom operators build caches, kernel configurations, or index tables on first use.

Dynamo guards on the Python state that compiled code reads. If that state is built
lazily during the first call, the first trace sees one value and later calls see
another, so the region recompiles. If the code that builds the state prints, logs, or
calls an untraceable function, the region also breaks. Build the state before the
first compiled call:

.. code-block:: python

   # Build caches, kernel configs, and static index tables in eager mode first.
   # prepare_static_state is a placeholder for your model's own setup method.
   model.prepare_static_state(example_inputs)

   compiled = torch.compile(model, backend="neuron", fullgraph=True)

The same applies to data-independent tensors such as gather indices, masks, and
geometry tables that depend only on input shapes. Compute them once, move them to the
``neuron`` device, and pass them to the compiled region, instead of recomputing them
in every iteration.

Graph partitioning best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-4:

**4:** Compile large, stable regions separately

    * Follow this practice when: Training a model with several large components, such as a backbone, task heads, and a loss, or when only some components benefit from compilation.

Compile each large component as its own region with ``fullgraph=True``. Each region
produces its own NEFFs, so you can change one component without a full
recompilation, and you can measure each region separately. Include the loss
computation in a compiled region where possible. A loss computed in eager mode runs
as many small operations and can be a large part of the step time.

.. code-block:: python

   model.backbone = torch.compile(model.backbone, backend="neuron", fullgraph=True)
   for i, head in enumerate(model.heads):
       model.heads[i] = torch.compile(head, backend="neuron", fullgraph=True)
   loss_fn = torch.compile(loss_fn, backend="neuron", fullgraph=True)

Components that you do not compile stay in eager mode. Compiled and eager code share
the same ``neuron`` tensors, so no copy occurs at the boundary. Keep each compiled
region large enough to justify its compilation time, and time it against its eager
version before you keep it.

.. _torch-compile-bp-5:

**5:** Measure graph boundaries instead of assuming one graph is fastest

    * Follow this practice when: A compiled region is large, or the region mixes NKI kernels with compiler-generated code.

One large graph gives the compiler the most scope, but it is not always the fastest
option. A graph boundary in the right place can reduce step time, and a boundary in
the wrong place can increase it. If a large region is slower than you expect,
compile its parts as separate regions and compare the step time. Keep the
partitioning that is faster, and record why it is there so a later change does not
remove it by mistake.

A common example is the backward pass of a data-parallel model. Both partitionings
let the all-reduce of a layer's weight gradients start as soon as those gradients are
available, so both can overlap communication with the backward pass of the next
layer. What differs is where that overlap can happen. When the whole model compiles
as one backward graph, the overlap is confined to that graph. With smaller graphs, or
with FSDP v2 and host-side collectives, a collective can also overlap with work
outside the backward pass, such as the optimizer, which gives the scheduler more
overlap budget. Measure both partitionings on the full iteration before you choose.

.. _torch-compile-bp-6:

**6:** Measure throughput per sample when you change the batch size

    * Follow this practice when: Increasing the batch size to raise throughput.

A larger batch size increases throughput only if step time grows more slowly than
the batch size. For workloads that are limited by data movement, such as gather-heavy
or scatter-heavy models, step time can grow linearly with batch size. Compute samples
per second for each batch size, and check device memory use with
``torch_neuronx.memory_summary()``. If throughput per sample does not improve, keep
the smaller batch size and use data parallelism across NeuronCores instead. When you
run one process per NeuronCore, compare configurations by the sum of the per-process
rates, not by the rate of one process.

Custom kernel best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-7:

**7:** Replace profiled hotspots with NKI kernels, one site at a time

    * Follow this practice when: A profile shows a few operations that dominate the step, such as large gathers, scatter-adds, or segment reductions.

Profile first, and use real inputs. A profile captured with random inputs
does not show the real cost of data-dependent operations such as gathers and scatters,
because random indices give a different memory access pattern. Profile the full step
with ``torch.profiler`` and view it in Neuron Explorer. See
:doc:`/torch/guides/memory-profiling-guide`.

For an operation that dominates the profile, write an NKI kernel and register it with
``torch_neuronx.nki_op`` (and an autograd backward for training). A kernel registered
as a PyTorch operator stays inside the compiled graph. See
:ref:`torchneuron-how-to-work-with-nki`.

An NKI kernel is not faster at every call site. For an operation that appears at many
sites, such as a convolution, benchmark each site's shape separately against the
compiler-generated version, and enable the kernel only where it is faster. Keep an
environment variable or flag that turns each kernel off, so you can compare
performance and numerics at any time.

Numerical validation best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-8:

**8:** Validate each compiled region against eager mode

    * Follow this practice when: Compiling a model for the first time, or after any performance change.

Compare the compiled outputs, losses, and gradients against eager mode on the same
inputs after every optimization. To make the comparison deterministic, disable
dropout and other random operations, because random numbers generated in a compiled
graph do not match the eager random stream. If the results differ, compile smaller
regions until you find the region that diverges, and compare it against an eager
reference, or against a CPU reference in ``float64``.

Check gradients as well as forward outputs, and check for values that are finite but
wrong, not only for ``NaN`` and ``Inf``. Expect small differences from ``bfloat16``
and from reductions that run in a different order. In models with discrete steps,
such as Hungarian matching or ``topk``, a small difference can change a decision and
cause a large change in gradients. For these models, compare losses and the
gradients of the affected components, or hold the matching or ``topk`` result
constant during the comparison.

Caching best practices
~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-9:

**9:** Compile once, then start the rest of the fleet

    * Follow this practice when: Running the same model on many instances.

Set up the shared NEFF cache as in :ref:`step 3 <torch-compile-optimize-step-3>`.
Compile on one instance first, then start the other instances, so that they load the
NEFFs instead of compiling them again.
Keep the compiler version, ``NEURON_CC_FLAGS``, and the logical NeuronCore
configuration identical across the fleet, because a difference in any of them makes
a new cache key.

Fallback and reliability best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _torch-compile-bp-10:

**10:** Keep the default fail-loud behavior in production

    * Follow this practice when: Promoting a compiled workload to production or CI.

By default (``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1``) the backend re-raises
compilation and execution failures instead of silently falling back to eager
execution. Keep this default in production and CI so a compilation regression
surfaces as an error rather than a silent slowdown. Re-enable fallback with
``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=0`` only where degrading to eager
execution is preferable to failing, and expect reduced performance when a region
falls back.

Related information
-------------------

- :ref:`torch-compile-deep-dive` - How torch.compile works on Neuron
- :ref:`torch-compile-ref` - Environment variables and helper API
- :ref:`best-practices-pytorch-neuron` - Broader PyTorch-on-Neuron best practices
- :ref:`torchneuron-how-to-work-with-nki` - Integrate NKI kernels with torch-neuronx
- :doc:`/torch/guides/memory-profiling-guide` - Profile memory and device activity
- :doc:`/torch/compile/reference-torch-compile` - Supported parameters and graph-break details
