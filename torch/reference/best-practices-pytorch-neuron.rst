.. meta::
   :description: Best practices for optimizing PyTorch workloads on AWS Trainium with TorchNeuron. Covers eager mode, torch.compile, distributed training, mixed precision, and NKI integration.
   :keywords: TorchNeuron, best practices, optimization, performance, PyTorch, Trainium, eager mode, torch.compile, FSDP, NKI, mixed precision
   :date-modified: 2026-10-15

.. _best-practices-pytorch-neuron:

==============================
Best practices for TorchNeuron
==============================

This topic covers best practices for running PyTorch workloads efficiently on AWS Trainium with
TorchNeuron. The practices apply to eager mode, torch.compile, distributed training, and custom
kernels on Neuron hardware.

Overview
--------

These practices help you maximize throughput, reduce compilation time, and avoid common performance
problems when running PyTorch on Trainium. They apply to both training and inference workloads on
TorchNeuron's native PyTorch backend.

This guidance targets the TorchNeuron public beta in Neuron SDK 2.33.0: PyTorch 2.14.0, ``torch-neuronx`` 2.14.3.0, Python 3.10–3.14, on Trainium2 and Trainium3.

TorchNeuron registers a ``neuron`` device with PyTorch through the PrivateUse1 mechanism. Supported ATen ops on a ``neuron`` tensor are lowered to StableHLO, compiled to a NEFF by ``neuronx-cc``, and cached. Unsupported ops fall back to CPU by default, which is convenient for bring-up but costs performance (see practice 1). ``torch.compile(backend="neuron")`` follows the same path, but compiles a whole FX graph at a time instead of one op at a time:

.. code-block:: text

   Eager:    ATen op -> NeuronDispatcher -> StableHLO -> neuronx-cc -> NEFF (cached) -> NRT
   Compiled: Dynamo -> AOTAutograd -> FX passes -> StableHLO -> neuronx-cc -> NEFF (cached) -> NRT


General principles
------------------

- Start with eager mode to validate correctness, then add ``torch.compile`` for performance.
- With TorchTitan, use SimpleFSDP for ``torch.compile`` workloads; it is more compiler-friendly than FSDP1 or FSDP2.
- Profile before optimizing — use Neuron Explorer to identify bottlenecks.
- Minimize graph breaks in ``torch.compile`` to maximize optimization opportunities.

Execution mode best practices
-------------------------------

**1:** Start development in eager mode, move to torch.compile for production

* Follow this practice when: Beginning a new workload on Trainium

**How to apply:**

1. Move the model and inputs to the device and run in eager mode first:

   .. code-block:: python

      import torch

      model = MyModel().to("neuron")
      x = torch.randn(8, 1024, device="neuron")
      ref = model(x)


2. Confirm that ops run on Neuron, not on the CPU. Unsupported ops fall back to CPU silently, which costs performance:

   .. code-block:: python

      torch_neuronx.clear_op_tracking()
      model(x)
      print("CPU fallback ops:", torch_neuronx.get_fallback_ops())


   To make fallbacks visible instead of silent, set ``TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1``. Execution errors are then raised, and only ops with no Neuron implementation fall back.

3. Switch to ``torch.compile`` using the supported settings and compare against the eager reference:

   .. code-block:: python

      compiled = torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)
      out = compiled(x)
      torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)


**Supported torch.compile parameters:**

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Support
     - Notes
   * - ``fullgraph``
     - Supported
     - ``True`` recommended for best performance
   * - ``dynamic``
     - ``False`` only
     - Dynamic shapes and symbolic ints are not supported
   * - ``mode``
     - ``"default"`` only
     - Other modes log a warning and run as ``"default"``; CUDA-graph modes give no speedup on Neuron
   * - ``options``
     - Not supported
     - Inductor tuning options do not configure Neuron; pass compiler flags through ``NEURON_CC_FLAGS`` instead
   * - ``disable``
     - Supported
     - Useful for A/B testing


**Tips:**

- Keep shapes static. Pad variable-length inputs to a fixed set of bucket sizes; each distinct shape triggers a separate compile. A ``SymInt`` in the FX graph, or a ``neuronx-cc`` compilation error that contains ``Unbounded dynamism is disabled``, means a dynamic shape got through.
- Prefer ``float32`` or ``bfloat16`` for floating point: ``float64`` is cast to ``float32`` internally. ``int64`` is supported. In eager mode, ops with native ``int64`` support run in 64 bits (``TORCH_NEURONX_INT64_MODE=native``, the default). Under ``torch.compile``, ``int64`` graph inputs are converted to ``int32``, so values outside the 32-bit range lose precision there.
- Keep the model and inputs on the same device. CPU inputs to a compiled graph are copied to the device automatically, which hides transfer costs. Set ``TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1`` to make this an error instead.
- Use contiguous tensors. Only contiguous tensors are supported under ``torch.compile``, and non-contiguous inputs are copied on every call (see :ref:`Additional recommendations <additional-recommendations>`).
- Compiled graphs **re-raise errors by default** rather than silently falling back to eager (``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION`` defaults to ``1``). Leave this default in place, because a silent fallback can make a broken graph look like it works.
- Compile only the hot regions. ``torch.compile`` can wrap a single function or submodule, through the decorator or ``nn.Module.compile()``, while the rest of the application stays eager.
- (Experimental) ``TORCH_NEURONX_MODULE_AUTO_COMPILE=1`` finds and compiles eligible leaf modules automatically, starting from ``Module.to("neuron")``. A leaf that hits a graph break, guard miss, or unprofitable graph stays in eager mode. Set ``TORCH_NEURONX_MODULE_AUTO_COMPILE_DEBUG=1`` to see which modules were promoted to compiled execution.

**2:** Use Adaptive Eager Execution for a balance of debuggability and performance

* Follow this practice when: Running in eager mode and not actively debugging

In plain eager mode every op is its own compile and launch, so per-op overhead dominates for small ops. Adaptive eager execution buffers consecutive eligible ops and compiles and executes them as one fused unit. Buffered ops are flushed by a timeout, by ``torch.neuron.synchronize()``, or by an op that cannot be fused. You keep the eager programming model and reduce the number of separate compiles and launches.

Adaptive eager execution is **experimental and off by default** in 2.33.0.

**How to apply:** set the variable before Python starts. It is read once, when the execution engine is created.

.. code-block:: bash

   export TORCH_NEURONX_ENABLE_CONCATENATION=1
   python train.py


Inspect what was fused:

.. code-block:: python

   import torch_neuronx

   print(torch_neuronx.get_concatenation_status())
   print(torch_neuronx.get_concatenated_ops())              # fused group -> original ops
   print(torch_neuronx.get_concatenated_op_execution_counts())


Tuning knobs (advanced): ``TORCH_NEURONX_CONCATENATION_BUFFER_SIZE_LIMIT`` and ``TORCH_NEURONX_CONCATENATION_QUEUE_LOW_THRESHOLD``. To profile the fusion itself, set ``TORCH_NEURONX_CONCATENATION_PROFILING=1`` or call ``torch_neuronx.metrics.set_concatenation_profiling_enabled(True)``.

**When to turn it off:** while debugging op-level errors or numerics, also set ``NEURON_LAUNCH_BLOCKING=1`` so that each op runs synchronously and errors point at the op that caused them. ``TORCH_NEURONX_ENABLE_STACK_TRACE=1`` adds Python stack traces per op, at extra cost.

**3:** Minimize graph breaks when using torch.compile

* Follow this practice when: Using torch.compile(backend="neuron")

Each graph break splits the model into separate NEFFs. The compiler cannot fuse or schedule across the split, and every boundary adds host round-trips and launch overhead. ``fullgraph=True`` turns any graph break into an error, so you find them early.

**Breaks that TorchNeuron inserts on its own.** When you import ``torch_neuronx``, it patches Dynamo to force a graph break on ops that Neuron cannot lower. With the default ``fullgraph=False``, those ops run in eager mode between compiled graphs. With ``fullgraph=True``, they raise ``Unsupported``. The list includes:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Op
     - Breaks when
   * - ``torch.sort``, ``torch.argsort``
     - The sorted dimension is longer than 16384
   * - ``torch.topk``
     - ``k`` ≥ 16384
   * - ``torch.nn.functional.grouped_mm``, ``torch._grouped_mm``
     - Always
   * - Ops with no Neuron lowering
     - Always (the list is collected at import from the installed torch, torch-mlir, and Neuron decompositions)


Keep sorts and top-k selections under these limits. Otherwise every step pays for a graph split and an eager round-trip.

**How to apply:**

1. Compile with ``fullgraph=True`` during development to find breaks. Use ``TORCH_LOGS="graph_breaks"`` to list where they happen and why.
2. Remove common break sources:

   - Host syncs inside the model: ``.item()``, ``.tolist()``, ``.cpu()``, ``print(tensor)``, or Python control flow that depends on tensor values.
   - Data-dependent output shapes, such as ``nonzero``, boolean mask indexing, or ``unique``. Rewrite them with fixed-shape alternatives like ``torch.where`` or masked arithmetic.
   - Object collectives such as ``dist.all_gather_object``. Inside compiled regions, use **functional collectives** (``torch.distributed._functional_collectives``), which trace into the graph.
   - Calls into untraceable extensions. All ``torch_neuronx._C`` bindings are wrapped with ``torch._dynamo.disable``, so call them outside the compiled region.

3. Write custom kernels with ``@nki.jit``. A kernel call on Neuron tensors is integrated into the compiled graph instead of causing a break. Wrap the call in ``@nki_op`` only when you need autograd or a named PyTorch operator (see :ref:`Additional recommendations <additional-recommendations>`).
4. If a break can't be avoided, place it at a natural boundary, such as between the forward pass and the optimizer step, rather than inside a hot layer.

**Inspecting what was compiled:**

.. code-block:: bash

   export TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
   export TORCH_NEURONX_DEBUG_DIR=./neuron_artifacts


Each graph writes ``neuron_dynamo_backend/rank<N>/<job_id>/{fx_graphs,stablehlo}/`` and ``neuronx-cc/<hash>/log-neuron-cc.txt`` under the debug directory. If there are more NEFFs than you expect, look for graph breaks or recompiles.

To debug numerics, ``TORCH_NEURONX_DYNAMO_TENSOR_DUMPS="inputs=all;outputs=0"`` saves each compiled graph's inputs and outputs as pickled NumPy arrays. You can then compare them against an eager or CPU run.

For how graph breaks work on Neuron, see the Graph breaks section of :ref:`torch-compile-deep-dive`. For a step-by-step workflow to find and remove them, see :ref:`torch-compile-optimize-howto`.

Distributed training best practices
-------------------------------------

**4:** Choose TorchTitan's SimpleFSDP for torch.compile workloads

* Follow this practice when: Combining distributed training with torch.compile

DDP and FSDP2 (``fully_shard``) are the supported distributed APIs in 2.33.0, and both work with eager execution. FSDP1 (``FullyShardedDataParallel``) also runs. However, FSDP's runtime hooks and parameter swapping are hard for the compiler to see through. SimpleFSDP, available in TorchTitan as the ``simple_fsdp`` module rather than as a ``torch.distributed`` API, expresses sharding as DTensor parametrizations with collectives in the traced graph. This lets the Neuron backend compile communication and compute together.

**How to apply (TorchTitan, Llama3-8B, trn2.48xlarge, TP4 × FSDP16):**

.. code-block:: bash

   export NEURON_RT_NUM_CORES=64
   torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES \
     --local-ranks-filter 0 --role rank --tee 3 \
     --rdzv_backend c10d --rdzv_endpoint "localhost:0" \
     -m torchtitan.train --module simple_fsdp.llama3 --config simple_fsdp_llama3_8b \
     --parallelism.tensor-parallel-degree 4 \
     --parallelism.data-parallel-shard-degree -1 \
     --training.global-batch-size 16 \
     --training.local-batch-size 1 \
     --compile.enable --compile.backend neuron


**Tips:**

- Use one Neuron device per process. Initialize with ``dist.init_process_group(backend="neuron")`` and call ``torch.neuron.set_device(local_rank)``.
- If you stay on FSDP2 or DDP, compile the per-layer blocks rather than the outer wrapper, so the FSDP hooks remain outside the compiled region.
- Inside compiled code, use functional collectives (``_functional_collectives.all_reduce``, ``all_gather_tensor``, ``reduce_scatter_tensor``), not the in-place ``dist.*`` APIs.
- Create meshes with ``DeviceMesh`` / ``init_device_mesh``. TorchNeuron hooks mesh creation to record every replica group, which SPMD compilation depends on. Process groups built by hand, outside a mesh, are not visible to it.

**5:** Configure sharding based on NeuronCore topology

* Follow this practice when: Setting up any distributed training on Trainium

Collective cost depends on where the ranks in a group sit physically. Ranks on the same chip communicate much faster than ranks on different chips, and those are faster again than ranks on different nodes. Place the most communication-heavy dimension on the closest cores.

**How to apply:**

1. **Put tensor parallelism on the innermost mesh dimension.** TP communicates on every layer, so keep each TP group within one Trainium device, for example TP4 on adjacent ranks. Data-parallel/FSDP communication happens once per layer or step and can span devices and nodes.

   .. code-block:: python

      from torch.distributed.device_mesh import init_device_mesh
      mesh = init_device_mesh("neuron", (world_size // 4, 4), mesh_dim_names=("dp_shard", "tp"))


2. **Match the rank count to the NeuronCore count.** Set ``NEURON_RT_NUM_CORES`` and ``--nproc_per_node`` to the same value. TorchNeuron sets ``NEURON_RT_VISIBLE_CORES`` for each rank from ``LOCAL_RANK``.
3. **Choose the Logical NeuronCore (LNC) configuration on purpose.** The LNC setting changes the number of visible cores and is part of the NEFF cache key, so switching it forces a recompile. See `Logical NeuronCore Configuration <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html>`_.
4. **Pin host threads to the device's NUMA node.** TorchNeuron does this automatically on first device use (``NEURON_NUMA_PIN=1``). On many-rank jobs, also bind memory from process start:

   .. code-block:: bash

      torchrun --no-python --nproc_per_node 64 ... neuron-numa-launch ./train.py --args


   Check it with ``torch_neuronx.numa.get_numa_status()``.

5. **Tune collective bucketing only after profiling.** The defaults (``COLLECTIVE_BUCKETSIZE_IN_MB=512``, split all-gather and split reduce-scatter enabled) are a reasonable starting point.


Mixed precision best practices
-------------------------------

**6:** Use torch.autocast for automatic mixed precision

* Follow this practice when: Training models where mixed precision improves throughput

Trainium's matmul engines run fastest in BF16. ``torch.autocast`` with ``device_type="neuron"`` runs matmul- and convolution-heavy ops in lower precision. Numerically sensitive ops such as reductions, softmax, and losses stay in FP32. You don't need to rewrite the model to get this.

**How to apply:**

.. code-block:: python

   model = MyModel().to("neuron")              # keep master weights in fp32
   opt = torch.optim.AdamW(model.parameters())
   compiled = torch.compile(model, backend="neuron", dynamic=False)

   with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
       loss = loss_fn(compiled(x), y)
   loss.backward()
   opt.step()


**Tips:**

- Prefer ``bfloat16``. It has the same exponent range as FP32, so it doesn't need a ``GradScaler``. Supported AMP dtypes are ``float32``, ``float16``, and ``bfloat16``, but not ``float64``.
- For FSDP, set the precision through the FSDP policy (``MixedPrecisionPolicy`` for FSDP2, ``MixedPrecision`` for FSDP1), not by casting the whole model.
- Validate accuracy against an FP32 eager run. Bitwise equality with CPU results is not expected; compare with tolerances.
- Don't combine framework autocast with a compiler ``--auto-cast`` flag in ``NEURON_CC_FLAGS`` unless you mean to. The compiler default is ``auto_cast="none"``.

Performance profiling best practices
--------------------------------------

**7:** Profile with Neuron Explorer before optimizing

* Follow this practice when: Investigating performance bottlenecks

TorchNeuron plugs into ``torch.profiler`` and produces three aligned traces: the PyTorch framework trace, the Neuron Runtime system trace, and the Neuron device trace. You can open them in Neuron Explorer to see where time goes: host dispatch, compilation, data movement, collectives, or device compute.

**How to apply:**

.. code-block:: python

   import os, torch
   from torch.profiler import ProfilerActivity, profile, schedule
   import torch_neuronx
   from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

   out_dir = "./profile_out"
   os.makedirs(out_dir, exist_ok=True)
   cfg = NeuronConfig(
       modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
       profile_output_dir=out_dir,
       correlate_device_time=True,   # adds NEURON device-time columns to key_averages()
   )
   exporter = NeuronProfiler(cfg)

   with profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
       experimental_config=cfg,
       schedule=schedule(wait=1, warmup=2, active=1),
       on_trace_ready=exporter.export_trace,   # writes next to the NRT traces
   ) as prof:
       for _ in range(4):
           train_step()
           torch.neuron.synchronize()
           prof.step()


**Tips:**

- **Warm up first.** The first iterations include compilation. Use a ``schedule`` with ``wait``/``warmup`` so only steady-state steps are profiled. ``precache_node_info=True`` (the default) collects per-NEFF metadata during the wait steps, which keeps profiling overhead out of the active step.
- **Limit capture on large jobs** with ``capture_enabled_for_nc="0"`` and ``max_events_per_nc``, so the trace stays a manageable size.
- **Call torch.neuron.synchronize() before stopping timers**, because execution is asynchronous.
- **Decide what to fix based on the trace:**

  - Many small NEFFs, or long gaps between them, point to graph breaks or eager overhead (see practices 2 and 3).
  - Collective-bound steps point to sharding layout or bucket sizes (see practice 5).
  - Compute-bound hot ops are candidates for NKI kernels.

- **Measure counters as well as time.** Set ``TORCH_NEURONX_ENABLED_METRIC_TABLES``, or use ``torch_neuronx.metrics.report()``, to track compile counts, cache hits, and fallbacks.
- **Profile memory as well.** ``torch_neuronx.memory_summary()`` and ``torch_neuronx.memory_stats()`` report current and peak allocation, allocation counts, and OOM counts. After recording memory history, ``NeuronProfiler.export_snapshot()`` writes a memory snapshot next to the runtime trace, so Neuron Explorer can load both together. ``torch_neuronx.empty_cache()`` releases cached blocks without freeing live tensors.


.. _additional-recommendations:

Additional recommendations
---------------------------

- **Pin package versions in production environments to avoid unexpected behavior from updates.** Pin ``torch``, ``torch-neuronx``, ``neuronx-cc``, ``nki``, and the Neuron runtime/collectives packages together. The compiler version is part of the NEFF cache key, so an upgrade also invalidates cached NEFFs.

- **Pass contiguous tensors to torch.compile regions.** The Neuron backend supports only contiguous tensor layouts in compiled graphs. If you pass a non-contiguous input, such as a transposed, permuted, sliced, or ``expand``\ ed view, the backend does not fail. Instead, it makes a contiguous copy before **every** call. That costs an extra device copy and allocation each step, and the cost doesn't show up in your model code.

  .. code-block:: python

     # Avoid: x.T is a non-contiguous view -> hidden copy on every call
     out = compiled(x.T)

     # Prefer: materialize the layout once, outside the hot loop
     x_t = x.T.contiguous()
     for _ in range(steps):
         out = compiled(x_t)


  - Store weights, KV caches, and other persistent buffers in the layout the graph uses, rather than permuting them on each call.
  - Check layouts at the graph boundary with ``t.is_contiguous()``. For ``channels_last`` and other non-default memory formats, convert to ``torch.contiguous_format`` before the compiled region.
  - **DTensor:** local shards (``dtensor.to_local()``) can be non-contiguous after redistribution, particularly in tensor-parallel attention. The TorchTitan tutorials patch the SDPA path with a ``force_local_contiguous`` workaround for this (`pytorch/pytorch#173041 <https://github.com/pytorch/pytorch/issues/173041>`_). Call ``.contiguous()`` on local shards before they enter a compiled region or a custom kernel.
  - Layout changes inside the graph, such as ``transpose``, ``permute``, and ``view``, are fine, because the compiler handles them. The requirement applies only to tensors that cross the graph boundary.

- **Use the TorchDynamo cache to avoid recompilation across runs.** TorchNeuron caches at two levels:

  - The persistent NEFF cache (on by default) reuses compiled NEFFs whenever the HLO and compiler config haven't changed. Point it at shared storage so all instances benefit:

    .. code-block:: bash

       export TORCH_NEURONX_NEFF_CACHE_DIR=/fsx/neff_cache        # shared (e.g. FSx)
       export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=/tmp/local_cache # must be local POSIX FS (locks)


  - The Dynamo precompile cache skips re-tracing ``torch.compile`` regions on warm starts:

    .. code-block:: bash

       export TORCH_CACHING_PRECOMPILE=1        # stored under TORCHINDUCTOR_CACHE_DIR


  - NEFF cache keys are built from the lowered HLO plus the compiler config (``TORCH_NEURONX_NEFF_KEY_FROM_HLO=1`` is the default). Python edits that don't change the lowered graph still hit the cache, and entries carry across sessions. Changing the target, the LNC setting, the compiler version, or compiler flags produces a new key.
  - Fill the cache ahead of time with ``torch_neuronx.compiler.compile_only()`` (or ``TORCH_NEURONX_COMPILE_ONLY=1``). This compiles NEFFs without running them. On a host without a Neuron device, also set ``NEURON_RT_FAKE_INSTANCE_TYPE``. The outputs are placeholders, so don't use this mode to check results.
  - If a compiler process crashes, it can leave a stale lock at ``{local_cache_dir}/{hash}/locks/compile.lock``, which blocks other processes for up to ``TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT`` seconds (default 1200). Delete it if that happens.

- **Consider NKI kernels for performance-critical operations that standard PyTorch operators cannot express efficiently.** Write the kernel with ``@nki.jit`` and call it on Neuron tensors. The call works in eager mode and inside ``torch.compile`` without a graph break. To train through the kernel, or to call it as a named PyTorch operator, wrap the call with ``@torch_neuronx.nki_op`` and register a backward pass:

  .. code-block:: python

     import nki, nki.language as nl, nki.isa as nisa
     from torch_neuronx import nki_op

     @nki.jit
     def add_kernel(x1, x2, y):
         a = nl.ndarray(x1.shape, x1.dtype, nl.sbuf)
         b = nl.ndarray(x2.shape, x2.dtype, nl.sbuf)
         c = nl.ndarray(y.shape, y.dtype, nl.sbuf)
         nisa.dma_copy(a, x1); nisa.dma_copy(b, x2)
         nisa.tensor_tensor(c, a, b, nl.add)
         nisa.dma_copy(y, c)
         return y

     @nki_op("mylib::nki_add", mutates_args={"y"})
     def nki_add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> None:
         add_kernel(x1, x2, y)


  - Declare every input the kernel writes in ``mutates_args``. An operator that writes into an input returns ``None``, because a ``torch.library`` custom operator cannot return a tensor that aliases an input.
  - For a backward pass, call ``register_autograd`` on an operator that does not write into its inputs, as in step 4 of :ref:`torchneuron-how-to-work-with-nki`. PyTorch does not attach a backward pass to an operator that writes into an input, such as the in-place ``nki_add`` above.
  - You do not need ``torch_neuronx.wrap_nki`` or the ``fake_fn`` argument of ``nki_op`` for kernels that you call on Neuron tensors. Both are for code that runs on ``meta`` tensors.

  For the full workflow, see :ref:`torchneuron-how-to-work-with-nki`.

  The built-in NKI flash-attention path for ``scaled_dot_product_attention`` is enabled by default when ``seqlen % 2048 == 0``. Set ``TORCH_NEURONX_ENABLE_NKI_SDPA=0`` to disable it. Pass any value that changes the kernel's control flow as a kernel argument; otherwise the NKI trace cache can return a stale kernel.

.. include:: /includes/torch/trn3-sizing-note.txt

Further reading
----------------

- :doc:`/torch/compile/reference-torch-compile` — ``torch.compile`` supported parameters and graph-break details
- :doc:`/torch/guides/torch-distributed` — Distributed training configuration
- :doc:`/torch/about-pytorch-neuron` — TorchNeuron architecture overview
- :doc:`/tools/neuron-explorer/index` — Neuron Explorer profiling tools
