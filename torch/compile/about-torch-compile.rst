.. meta::
   :description: How torch.compile works on AWS Neuron. Deep dive into the TorchNeuron TorchDynamo backend: FX capture, AOT Autograd, StableHLO lowering, neuronx-cc, NEFF caching, graph breaks, fallback, and compile-only mode.
   :keywords: TorchNeuron, torch.compile, TorchDynamo, AOT Autograd, StableHLO, neuronx-cc, NEFF, caching, graph breaks, compile-only, AWS Neuron, Trainium
   :date-modified: 2026-08-21

.. _torch-compile-deep-dive:

===================================
How torch.compile works on Neuron
===================================

This topic explains how ``torch.compile(backend="neuron")`` compiles PyTorch
models for AWS Trainium, from FX graph capture through NEFF execution. Familiarity
with :ref:`TorchNeuron <about-pytorch-neuron>` and PyTorch ``torch.compile`` is
assumed.

What you should know before reading
-----------------------------------

- **TorchNeuron backend:** See :ref:`about-pytorch-neuron` for the eager-mode and
  device-backend fundamentals.
- **torch.compile / TorchDynamo:** Familiarity with
  `TorchDynamo <https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html>`__
  and AOT Autograd.

Overview
--------

``torch.compile(model, backend="neuron")`` selects the native Neuron TorchDynamo
backend. Dynamo captures your PyTorch code into FX graphs, and the Neuron backend
lowers each graph through StableHLO to a NEFF that runs on Trainium. Compilation is
just-in-time: it happens on the first call with a given input signature, and later
calls with the same signature reuse the compiled NEFF.

The backend can compile the whole model or, when Dynamo inserts graph breaks, a set
of independently compiled subgraphs. It does **not** use TorchInductor — the
lowering path is StableHLO and the Neuron Compiler (``neuronx-cc``), not the
inductor code-generation stack you would use on a GPU.

The compilation pipeline
------------------------

A compiled call flows through four stages:

#. **TorchDynamo** traces Python bytecode and captures it into an FX graph, guarded
   by the input signature (shapes and dtypes).
#. **AOT Autograd** partitions the traced graph into separate forward and backward
   graphs, which are compiled as independent graphs by backend compiler. This lets a single compiled callable serve both training and
   inference.
#. **Neuron backend** runs its FX passes, then lowers the FX graph to **StableHLO**
   MLIR through Neuron's vendored ``torch_mlir`` importer. Composite operators (for
   example, scaled dot-product attention) are decomposed during this step, and the
   graph can embed NKI kernels as custom operators.
#. **Neuron Compiler** (``neuronx-cc``) applies hardware optimizations to the
   StableHLO and emits the NEFF that the Neuron Runtime executes on Trainium.

.. note::

   The Neuron backend provides native ``torch.compile`` support **without**
   TorchInductor. The two required changes from a CUDA ``torch.compile`` call are
   to pass ``backend="neuron"`` and to drop CUDA-only arguments (for example,
   ``mode="max-autotune"``).

Graph breaks
------------

Graph breaks are supported. When Dynamo cannot keep the whole model in one graph —
for example, because of a data-dependent branch or an unsupported construct — it
splits execution into multiple FX graphs. The Neuron backend compiles each graph
segment independently, so a break reduces the scope of optimization but does not
stop compilation.

Two behaviors are worth knowing:

- **Trivial subgraphs are short-circuited.** A segment that only forwards inputs to
  outputs (no compute) is passed through without a compile, so breaks around such
  segments cost nothing.
- **NKI kernels do not force a break.** A kernel registered as a PyTorch operator
  integrates into the surrounding graph and is compiled with it.

For the operations that commonly cause graph breaks, see
:ref:`torchneuron-graph-breaks`. To keep a model in a single
graph, compile with ``fullgraph=True`` so a break raises instead of silently
splitting.

Fallback execution
-------------------

The backend has two internal degradation paths: FX-to-StableHLO conversion failure
and NEFF execution failure. On either failure it can fall back to running the
original (uncompiled) FX graph.

.. important::

   Fallback is **disabled by default**. The environment variable
   ``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION`` defaults to ``True``, so the backend
   **re-raises** the underlying error instead of silently running the model in
   eager mode. This default exists because silent fallback masks real compilation
   and execution failures. Set ``TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=0`` to
   re-enable fallback when you would rather degrade to eager execution than fail.

Because the default surfaces errors, you normally do not need to change anything to
"fail loudly" — that is already the behavior. See :ref:`torch-compile-best-practices`
for when re-enabling fallback makes sense, and :ref:`torch-troubleshooting` for
diagnosing the failures it would otherwise hide.

Caching and recompilation
--------------------------

The backend caches work at three layers so that repeated compilations get faster and
compiled artifacts can be reused across runs and instances:

- **Persistent NEFF cache** — the compiled NEFF is stored on disk, content-addressed
  by a key derived from the lowered HLO bytes plus the compiler configuration and
  compiler version. Because the key comes from the IR and the compiler, a change to
  the model, the lowering, or the compiler produces a different key, so a stale
  entry is never reused. Point ``TORCH_NEURONX_NEFF_CACHE_DIR`` at a shared mount
  (for example, Amazon FSx) to reuse NEFFs across instances.
- **HLO cache** — lowered StableHLO is cached on disk
  (``TORCH_NEURONX_HLO_CACHE_DIR``), shared by the eager and ``torch.compile`` entry
  points.
- **Dynamo precompile cache (MegaCache)** — when ``TORCH_CACHING_PRECOMPILE=1``, the
  PyTorch Dynamo precompile cache stores the backend result of the trace. The Neuron
  cache entry embeds the StableHLO bytes directly, so a warm load re-invokes the
  compiler (hitting the persistent NEFF cache) without retracing or recompiling, and
  ``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts`` can move NEFFs
  between machines.

Because the backend forces ``dynamic=False``, each distinct input shape is a
separate compiled graph. Changing input shapes forces recompilation, so pad or
bucket inputs to a fixed set of shapes to maximize cache reuse. See
:ref:`torch-compile-ref` for the full list of cache-directory environment variables.

Compile-only (device-free) compilation
---------------------------------------

Compile-only mode builds the NEFF — including the backward graph — without executing
it on a device. It is meant for build and CI hosts that have the compiler toolchain
but no attached Trainium. Enable it with the ``TORCH_NEURONX_COMPILE_ONLY=1``
environment variable or the ``torch_neuronx.compiler.compile_only()`` context
manager. In this mode a Neuron runtime-initialization failure is non-fatal, and the
backend returns placeholder outputs carrying the compiled artifact.

Supported modes (summary)
-------------------------

The backend supports ``mode="default"`` only. The other ``torch.compile`` modes
(``reduce-overhead``, ``max-autotune``, ``max-autotune-no-cudagraphs``) emit a
warning and fall back to default. ``dynamic`` is forced to ``False`` regardless of
what you pass, because dynamic shapes are not supported. ``fullgraph=True`` is
recommended so that graph breaks are surfaced rather than silently splitting the
model. The full parameter matrix lives in the
:doc:`/torch/compile/reference-torch-compile`.

Related concepts
----------------

* :ref:`about-pytorch-neuron` - TorchNeuron architecture and execution modes
* :doc:`/torch/compile/reference-torch-compile` - Supported ``torch.compile`` parameters and graph-break details

Further reading
---------------

* :ref:`torch-compile-optimize-howto` - Optimize a model with ``torch.compile``
* :ref:`torch-compile-ref` - Backend environment variables and helper API
* :ref:`torch-compile-best-practices` - torch.compile best practices
