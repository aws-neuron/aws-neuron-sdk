.. meta::
   :description: Port GPU-compatible transformer models to the vLLM Neuron Trainium2 backend using the Autoport vLLM Neuron skill
   :keywords: autoport, vllm, vllm-neuron, model porting, Trainium2, trn2, GPU-compatible, transformer, tensor parallelism, expert parallelism, validation
   :date-modified: 2026-08-13

.. _neuron-framework-autoport-vllm-neuron:

Deep dive: Port GPU-compatible models to the vLLM Neuron backend with the Autoport skill
========================================================================================

**Why read this guide?** You are an ML engineer who needs to serve a GPU-compatible
transformer model on AWS Trainium2 through the vLLM Neuron backend. The model
architecture is not yet registered with vLLM Neuron, so you need to port it. The
Autoport vLLM Neuron skill handles this for you. It is an AI agent workflow that
researches the model architecture, generates a vLLM Neuron model implementation,
registers it, and validates accuracy against the GPU-compatible reference
implementation.

**How to use this guide:** Use this guide when you need to add a model to the
vLLM Neuron backend that is not already in the model registry. Jump to
:ref:`workflow steps <autoport-vllm-workflow>` if you already have your environment
ready.

This skill is distinct from the :ref:`NxD Inference Autoport skill
<neuron-framework-autoport>`. The NxD Inference skill targets the
``neuronx-distributed-inference`` library and compiles to NEFF directly. This skill
targets the **vLLM Neuron Trainium2 backend** and produces a model that plugs into
the vLLM model registry for online and offline serving. If you are not sure which
serving stack you want, see :ref:`nxdi-vllm-user-guide-v1`.

You need experience with PyTorch model development, transformer architectures, and
tensor parallelism to get the most out of this content.

Prerequisites
-------------

Before you start, make sure you understand the following topics.

- **vLLM Neuron backend.** How vLLM serves models on Trainium and how the
  vLLM Neuron model registry works. See :ref:`nxdi-vllm-user-guide-v1`.
- **PyTorch model architecture.** Transformer building blocks (attention, MLP,
  embeddings, normalization) and how source repositories organize model code.
- **Tensor parallelism concepts.** How models shard across NeuronCores, and the
  constraints that head counts and intermediate sizes place on the tensor parallel
  degree. See :doc:`/libraries/nxd-inference/app-notes/parallelism`.
- **Expert parallelism.** For Mixture of Experts (MoE) models, how experts
  distribute across parallel groups. See :ref:`moe-inference-deep-dive`.

Overview
--------

Adding a new model architecture to the vLLM Neuron backend usually takes multiple
days of engineering work: reading the source implementation, mapping every weight,
respecting Trainium hardware constraints, and debugging silent accuracy bugs. The
Autoport vLLM Neuron skill replaces that manual effort. An AI coding agent executes
the full porting process from architecture research through equivalence validation.

The skill works with dense transformer models (decoder only), models with grouped
query attention (GQA) and multi-query attention (MQA), Mixture of Experts (MoE)
models that fit within tensor parallelism, large MoE models that require expert
parallelism (EP), models with novel attention (sliding window, multi-latent
attention), and vision-language models.

The workflow has three phases and eleven steps.

**Phase A: Research and analysis.**

1. **Architecture research.** The agent fetches the source model config, reads the
   modeling source, inspects the checkpoint weight keys, and computes valid tensor
   parallel sizes.
2. **Architecture analysis.** The agent compares the model to the closest existing
   vLLM Neuron reference model and produces a summary of the changes needed.

**Phase B: Code generation.**

3. **Generate boilerplate.** The agent creates ``config.py``, ``factory.py``, and
   ``__init__.py`` from templates.
4. **Generate model.py.** The agent copies the reference model and modifies it
   section by section (normalization, RoPE, attention, MLP, MoE, decoder, backbone,
   LM head, weight loading).
5. **Register the model.** The agent adds the model to ``vllm_neuron/model/registry.py``.
6. **Generate example and docs.** The agent creates a run script, a results file,
   and a model README.
7. **Self-review.** The agent runs through a checklist before reporting completion.

**Phase C: Validation.**

8. **Smoke test.** The agent runs offline inference and checks for coherent output.
9. **Online serving and APC.** The agent serves the model through the vLLM OpenAI
   API server and runs a battery of completion, batch, streaming, and prefix caching
   tests.
10. **Logit validation.** The agent compares Neuron logits against a CPU reference
    implementation and reports sigma values across top-k levels.
11. **Deep equivalence validation.** The agent invokes the
    :ref:`Equivalence skill <neuron-framework-equivalence>` with the vLLM Neuron
    adapter for component-level and end-to-end verification.

.. _autoport-vllm-requirements:

Hardware and software requirements
-----------------------------------

* **Instance type.** ``trn2.48xlarge``. The vLLM Neuron backend targets Trainium2.
  Each Neuron device exposes roughly 24 GB of HBM shared between two NeuronCores in
  the default LNC=2 configuration (96 GB per device).
* **Neuron SDK.** Installed on the instance. The Amazon Linux 2023 Neuron DLAMI is a
  good starting point.
* **Python.** 3.10 or later.
* **vLLM Neuron package.** The ``vllm_neuron`` or ``private_vllm_neuron`` package
  must be present in your workspace. The skill stops if it cannot find one.
* **Transformers package.** Used to read the reference model config and source.
* **Model weights.** Downloaded from a model hub or available locally as
  ``.safetensors`` files.

.. note::
   The Autoport vLLM Neuron skill supports a dry run mode for environments without
   Trainium hardware. In dry run mode the agent performs architecture research and
   code generation but skips compilation, inference, and validation. See
   :ref:`autoport-vllm-dry-run`.

.. _autoport-vllm-parameters:

Porting parameters
------------------

The skill accepts the following arguments.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``model-name``
     - The snake_case name for the port (for example, ``yi``, ``deepseek_v2``).
       Becomes the directory name under ``vllm_neuron/model/``.
   * - ``hf-model-id``
     - The source model identifier (for example, ``01-ai/Yi-6B-Chat``).
   * - ``--review``
     - Optional flag. Pauses after the architecture analysis (Step 2) so you can
       confirm or override the agent's decisions before code generation.

The agent derives the PascalCase class prefix from ``model-name`` (for example,
``yi`` becomes ``Yi``, ``gpt_neox`` becomes ``GPTNeoX``). Model code goes in
``vllm_neuron/model/<model-name>/`` and the example script goes in
``examples/vllm_neuron/models/<model-name>/``.

.. _autoport-vllm-workflow:

Workflow
--------

Phase A: Research and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 1. Research the architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent gathers everything it needs before writing any files.

**Fetch the source model config.** The agent loads ``AutoConfig`` for the model and
records the ``architectures`` field (the exact string used for registry
registration), the layer dimensions (``hidden_size``, ``intermediate_size``,
``num_hidden_layers``), the head counts (``num_attention_heads``,
``num_key_value_heads``, ``head_dim``), the vocabulary and position limits, the
normalization epsilon, the RoPE parameters, the activation function, the
``tie_word_embeddings`` flag, the bias flags, and any model-specific fields such as
``sliding_window`` or ``num_local_experts``.

**Read the reference source.** The agent reads the modeling file to confirm exact
weight names, whether Q/K/V projections are fused or separate in the checkpoint,
which projections carry biases, and any non-standard behavior such as parallel
residual connections.

.. important::
   The agent verifies the normalization type by reading the ``forward()``
   implementation, not the class name. Some models name their class ``RMSNorm`` but
   implement full LayerNorm (with mean subtraction and bias). Using the wrong
   normalization produces output that passes smoke tests but fails accuracy
   validation.

**Read the porting guide.** The agent reads the canonical model bringup guide at
``doc/vllm_neuron/source/design/framework/model_bringup.md`` and the parallelism
design docs it links. The bringup guide designates GPT-OSS BF16 as the canonical
annotated reference, where ``# >>> PARALLELISM <<<`` blocks are infrastructure to
keep as-is and ``# <-- MODEL-SPECIFIC`` blocks are what change per architecture.

**Select a reference model.** The agent picks the closest existing vLLM Neuron model
as a copy source.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - If the model has...
     - Use reference
   * - Standard GQA + RoPE (most common)
     - ``llama3/``
   * - NoPE layers (per-layer RoPE omission)
     - ``smollm3/``
   * - QKV bias
     - ``llama3/`` (add bias params to QKV/O projections)
   * - Q/K per-head RMSNorm
     - ``qwen3/`` (dense) or ``qwen3_moe/`` (MoE)
   * - Mixture of Experts (fits in TP)
     - ``qwen3_moe/``
   * - Mixture of Experts (needs EP)
     - ``gpt_oss/``
   * - Large MoE with multi-latent attention
     - ``deepseek_v32/``
   * - Vision-language model
     - ``qwen3_vl/``

**Compute valid tensor parallel sizes.** The agent finds tensor parallel (TP) sizes
that satisfy all five rules.

1. ``num_attention_heads % tp_size == 0``
2. ``(num_attention_heads / tp_size)`` is **even** (NKI decode megakernel constraint)
3. ``num_key_value_heads % tp_size == 0`` or ``tp_size % num_key_value_heads == 0``
   (GQA replication)
4. ``intermediate_size % tp_size == 0``
5. ``vocab_size % tp_size == 0`` (or use padding)

The agent then applies the memory constraint. With TP=N, each rank holds roughly
``model_size_bytes / N`` of weights plus KV cache. If per-rank weight memory exceeds
about 20 GB, the agent increases the TP size. It recommends the smallest valid TP
size of 2 or greater that also fits the model in memory.

**Inspect the checkpoint weight keys.** The agent opens the ``.safetensors`` files
to confirm which layers carry biases, the exact key naming convention, and weight
shapes. This is essential for building the ``load_weights`` mapping correctly.

**Evaluate expert parallelism (MoE models only).** If the model is MoE and no single
TP size satisfies all five rules while fitting in memory, the model needs Expert
Parallelism. EP uses two-level parallelism: a TP sub-group for attention and dense
layers, and an EP group for distributing experts. The agent computes the EP degree
from the world size (64 on a full ``trn2.48xlarge``) and verifies that
``num_local_experts`` divides evenly across EP groups.

.. important::
   **Unified collective group for EP.** The Neuron Distributed Graph Engine (DGE)
   cannot handle two different collective group sizes in the same NEFF. All
   collectives (attention, MLP, MoE, embedding, LM head, sampler) must use a single
   group, the full world group. Weight sizing uses the sub-group, but collectives go
   through the unified group. Violating this causes ``NEFF Warmup failed with
   status 1006``.

Step 2. Analyze the architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent presents a summary comparing the target model to the reference, section by
section (normalization, RoPE, attention, MLP, MoE, EP, decoder, backbone, LM head),
with the change needed for each and any known pitfalls.

If you passed the ``--review`` flag, the agent pauses here and asks you to confirm or
override its decisions. Otherwise, the agent prints the analysis and proceeds
automatically.

Phase B: Code generation
^^^^^^^^^^^^^^^^^^^^^^^^^

The defining rule of code generation is **search before implementing**. Before
writing any module (normalization, RoPE, attention, MLP), the agent searches for an
existing implementation in priority order: shared functional ops in
``vllm_neuron/functional/``, shared neural network modules in ``vllm_neuron/nn/``,
other ported models in ``vllm_neuron/model/*/model.py``, pre-built NKI kernels in the
installed ``neuronxcc`` package, and finally production NKI kernels. The agent
implements from scratch only when nothing matches.

Step 3. Generate boilerplate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent generates three files from the skill's templates, filled with values from
Step 1.

.. code-block:: text

   vllm_neuron/model/<model-name>/config.py     # from config.py.template
   vllm_neuron/model/<model-name>/factory.py    # from factory.py.template
   vllm_neuron/model/<model-name>/__init__.py   # from init.py.template

Step 4. Generate model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent copies the selected reference model's ``model.py`` and modifies it section
by section: normalization (RMSNorm vs LayerNorm, eps, biases), RoPE (variant and
scaling), attention (bias, sliding window, GQA/MQA head counts), MLP (activation and
gating), MoE (router precision, expert loop), decoder (norm placement and residual
pattern), backbone (vocab padding, position embeddings), and LM head (tied vs
untied, bias). It then builds the complete checkpoint key mapping so that every
``nn.Parameter`` has a corresponding ``load_weights`` entry, and renames all classes
with the model's PascalCase prefix.

The agent observes four hardware constraints that pass on the CPU simulator but fail
on real Trainium hardware.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constraint
     - Why it matters
   * - **No meta tensors in __init__**
     - vLLM creates models on ``torch.device("meta")`` first, then loads weights. Any
       tensor computation in ``__init__`` (such as RoPE ``inv_freq``) creates meta
       tensors that later fail with "Cannot copy out of meta tensor". The agent
       computes ``inv_freq`` lazily in ``forward()``.
   * - **No .to(device) in forward**
     - On the CPU simulator this is a no-op, but on hardware ``torch.compile`` turns
       it into a cross-device copy that raises ``NotImplementedError: unimplemented
       _copy_from``. Tensors passed into ``forward()`` are already on the correct
       device.
   * - **No .item() in forward**
     - Scalar extraction breaks ``torch.compile`` with "Unsupported Tensor.item()
       call". The agent uses static values or tensor operations instead.
   * - **2D bias shapes for NKI decode**
     - The ``NF.attention_decode`` megakernel requires bias tensors to be 2D
       ``[1, size]``, not 1D. The agent unsqueezes bias tensors before passing them.

For MoE models, the agent keeps router gate weights in float32, not bf16, because the
softmax over experts is extremely sensitive to precision. For MoE models with EP, the
agent implements the unified ``sp_group`` pattern: ``tp_group`` (the sub-group) sizes
weights, while ``sp_group`` (the full world) carries every collective. After loading
weights, it scales replicated layers (``o_proj``, dense ``down_proj``) by
``1/ep_degree`` but does not scale MoE expert weights, and it filters expert weight
loading by EP rank.

Step 5. Register the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent edits ``vllm_neuron/model/registry.py`` to import the new model and add it
to ``get_models()``.

.. important::
   The source architecture string must **exactly** match the ``architectures`` field
   from ``config.json``, including capitalization. For example, a config with
   ``"architectures": ["PhiMoEForCausalLM"]`` requires the registry entry
   ``("PhiMoEForCausalLM", PhiMoEForCausalLM)``. ``"PhimoeForCausalLM"`` would fail to
   load the model.

Step 6. Generate example and docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent creates the example run script
(``examples/vllm_neuron/models/<model-name>/run.py``), a results file, and the model
README (``vllm_neuron/model/<model-name>/README.md``). The README includes an
architecture parameter table, the key differences from the reference model, and a
feature status table (TP/SP/DP/EP, Eagle3, FP8 KV cache) with status markers and
notes.

Step 7. Self-review
~~~~~~~~~~~~~~~~~~~~~

Before reporting completion, the agent runs through a checklist: every
``nn.Parameter`` has a ``load_weights`` mapping, bias shapes are 2D for the decode
megakernel, no ``.to(device)`` or ``.item()`` in the forward path, RoPE ``inv_freq``
computed lazily, class names consistent across all files, the registry uses the exact
source architecture string, shared modules imported rather than reimplemented, MoE router
weights in float32, and normalization type matching the reference source. EP models get an
additional checklist covering the unified collective group, weight scaling, and
expert rank filtering.

Phase C: Validation
^^^^^^^^^^^^^^^^^^^^

Step 8. Smoke test
~~~~~~~~~~~~~~~~~~~

The agent runs the example script for offline inference.

.. code-block:: bash

   # Standard (non-EP) models
   NEURON_SKIP_EFA_AFFINITY=1 python examples/vllm_neuron/models/<model-name>/run.py

   # EP models (requires contiguous collective topology)
   NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1 python examples/vllm_neuron/models/<model-name>/run.py

``NEURON_SKIP_EFA_AFFINITY=1`` is needed on trn2 instances where the PCI topology
does not match the hardcoded mapping. It is safe for single-node TP. The agent checks
for successful weight loading (no missing key errors), successful compilation (no NKI
shape mismatch errors), and reasonable generated text (not garbage or degenerate
repetition), then records the results.

Step 9. Online serving and APC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent starts the vLLM OpenAI API server and runs a test battery: a basic
completion, a batch of four prompts, a streaming request, and a counting
continuation. It measures latency and throughput over ten requests, then restarts
with ``--enable-prefix-caching`` (APC) and re-runs the tests.

.. code-block:: bash

   NEURON_SKIP_EFA_AFFINITY=1 python3 -m vllm.entrypoints.openai.api_server \
       --model <hf-model-id> \
       --tensor-parallel-size <tp-size> \
       --max-model-len 256

Step 10. Validate logits
~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent generates the canonical logit validation test and runs it. The test
compares Neuron output logits against a CPU reference implementation at several top-k
levels (5, 50, 1000, all) and reports the difference as sigma values.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Sigma
     - Interpretation
   * - < 3
     - Excellent. Within normal numerical noise.
   * - 3-5
     - Acceptable. Minor differences, usually from bf16 quantization.
   * - 5-10
     - Investigate. May indicate a real issue, but can be model-specific.
   * - > 10
     - Likely a bug in the port.

.. note::
   A logit validation failure is not a hard blocker. The agent reports the sigma
   values and which top-k levels failed, explains what they mean, and continues to
   completion. Some models naturally have higher variance at certain top-k levels,
   and you decide whether the accuracy is acceptable for your use case.

Step 11. Validate deep equivalence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent always runs the full equivalence pipeline for rigorous component-level and
end-to-end verification. It invokes the :ref:`Equivalence skill
<neuron-framework-equivalence>` with the **vLLM Neuron adapter** (``--target-stack
vllm_neuron``), which handles distributed init, ``from_configs()`` instantiation,
weight mapping with transpositions and QKV fusion, attention metadata construction,
and the ``vllm.LLM`` API.

The equivalence pipeline maps each submodule between the GPU-compatible reference
implementation and the port, runs component-level R-ratio tests, performs fault localization and
debugging if it finds failures, runs a teacher-forced end-to-end comparison (R-ratio,
cosine similarity, KL divergence), and generates an equivalence report.

- All R < 1.2 and the end-to-end test passes: the port is verified at the component
  level.
- Component failures: the equivalence report includes patches showing what is wrong.
  The agent uses these as a guide to fix the model code.

.. _autoport-vllm-dry-run:

Dry run mode
------------

When Trainium hardware is not available, the agent operates in dry run mode. It skips
the agent-level prerequisites (package import checks, the ``neuron-ls`` NeuronCore
check), activates the provided virtual environment, and resolves source paths by
filesystem lookup rather than by importing packages. Phase A (research) and Phase B
(code generation) run normally. The agent skips Phase C (compilation, inference, and
validation) because no hardware is present.

Invoke dry run mode by specifying ``dry-run`` when calling the skill.

.. _autoport-vllm-file-organization:

File organization
-----------------

The skill produces files in these locations.

.. code-block:: text

   vllm_neuron/model/<model-name>/
   ├── __init__.py          # Exports the factory ForCausalLM class
   ├── config.py            # Model config class
   ├── factory.py           # Model factory
   ├── model.py             # The ported model implementation (final product)
   └── README.md            # Architecture table and feature status

   vllm_neuron/model/registry.py            # Modified: new model registered

   examples/vllm_neuron/models/<model-name>/
   ├── run.py               # Offline inference example
   └── results/results.md   # Offline and online serving results

   test/vllm_neuron/model/<model-name>/bf16/e2e/
   └── test_logits.py       # Logit validation test

.. _autoport-vllm-debugging:

Common issues
-------------

Inference and compilation failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symptom
     - Likely cause and fix
   * - ``Cannot copy out of meta tensor``
     - RoPE or another tensor computed in ``__init__``. Move it to ``forward()`` and
       compute it lazily.
   * - ``Bias shape must be [1, I], got (N,)``
     - 1D bias passed to the NKI decode megakernel. Add ``.unsqueeze(0)`` before
       passing to ``NF.attention_decode``.
   * - ``unimplemented _copy_from xla:0 neuron:N``
     - A ``.to(device)`` call in the forward path. Remove it; tensors are already on
       device.
   * - ``Unsupported Tensor.item()``
     - A ``.item()`` call in the forward path. Use static values or tensor ops.
   * - ``Checkpoint key(s) not found``
     - Weight mapping mismatch. Check ``load_weights`` mappings against the actual
       checkpoint keys.
   * - ``size mismatch for lm_head.bias``
     - lm_head bias not mapped correctly. Use a separate ``nn.Parameter`` and an
       explicit mapping.
   * - ``nrt_tensor_allocate status=4``
     - HBM out of memory. Increase the TP size.
   * - ``No EFA device found``
     - PCI topology mismatch. Set ``NEURON_SKIP_EFA_AFFINITY=1``.
   * - Degenerate output (``the the the...``)
     - Decode path bug (bias, KV cache, or norm). Test prefill only
       (``max_tokens=1``), then debug decode.
   * - Registry ``AttributeError: no attribute 'from_configs'``
     - Architecture string mismatch in the registry. Verify the exact source
       architecture string from ``config.json``.

Expert parallelism failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These apply only to MoE models that use expert parallelism.

- ``NEFF Warmup failed with status 1006`` or a DGE scatter/gather out-of-bound error.
  Mixed collective group sizes in one NEFF. Unify all collectives to one group
  (``sp_group``).
- OOM during weight loading. Expert weights are not filtered by EP rank. Map only the
  local expert range.
- Garbage output but no crash. Check the ``o_proj`` and ``down_proj`` scaling
  (missing ``div_(ep_degree)``), and confirm the router gate is replicated, not
  EP-sharded.
- Compilation OOM or timeout. With many experts in a loop, the compiler unrolls all
  iterations. Switch to the ``NF.moe_cte`` (prefill) and ``NF.moe_block_tkg``
  (decode) kernels.

.. note::
   Do not clear ``/var/tmp/neuron-compile-cache`` as a pre-flight step. It is a
   shared system directory that other processes may depend on. Clear it only
   reactively if you hit an ``[NLA001]`` JSON parse error or a ``FileNotFoundError``
   on neff output paths.

Related resources
-----------------

- :ref:`neuron-framework-autoport` for the NxD Inference Autoport skill (the
  NEFF-targeting variant of this workflow).
- :ref:`neuron-framework-equivalence` for the Equivalence skill that Step 11 invokes.
- :ref:`nxdi-vllm-user-guide-v1` for deploying and serving models with vLLM on Neuron.
- :ref:`moe-inference-deep-dive` for Mixture of Experts support on Neuron.
- `Neuron Agentic Development <https://github.com/aws-neuron/neuron-agentic-development>`_
  for the skill source and templates.
