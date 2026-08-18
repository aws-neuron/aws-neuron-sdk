.. meta::
   :description: Use the Neuron Framework Equivalence skill to validate functional and numerical equivalence between a HuggingFace reference model and a NxD Inference ported implementation on AWS Trainium.
   :keywords: equivalence, model validation, NxD Inference, vLLM, vLLM Neuron, NeuronX, Trainium, R-ratio, tensor comparison, accuracy, porting verification
   :date-modified: 2026-08-13

.. _neuron-framework-equivalence:

Deep dive: Validate model ports with the Equivalence skill
==========================================================

**Why read this guide?** This guide is intended for ML engineers who need to verify
that a ported NxD Inference model produces numerically correct output compared to its
HuggingFace reference. It explains the Equivalence skill — an AI agent-driven workflow
that progressively validates a model port through eight stages of structural analysis,
component-level testing, fault localization, debugging, and end-to-end accuracy verification.

**How to use this guide:** If you are porting a model from scratch, start with the
:ref:`Autoport skill <neuron-framework-autoport>` first. Use this guide after you have a
completed port and need to verify its correctness. Skip to the
:ref:`workflow stages <equiv-workflow>` if you already understand the environment setup
and R-ratio methodology.

This topic explores the Equivalence skill in depth, covering structural scaffolding,
the 3-tensor R-ratio method, component-level testing, fault localization, patching,
end-to-end comparison, and downstream evaluation. You need experience with PyTorch
model development, the NxD Inference library structure, and basic numerical analysis
to fully understand this content.

Prerequisites
-------------

Before you start, you must be familiar with the following:

- **NxD Inference library overview:** How to build and deploy models
  using NxD Inference. See :doc:`../index`.
- **PyTorch model architecture:** Transformer building blocks (attention, MLP,
  embeddings) and how HuggingFace models are structured.
- **Neuron compilation model:** How ``torch-neuronx`` traces Python code into
  HLO and compiles it to NEFF for NeuronCores. See :ref:`nxdi-feature-guide`.
- **Tensor parallelism concepts:** How models are sharded across NeuronCores.
  See :doc:`/libraries/nxd-inference/app-notes/parallelism`.
- **Model porting workflow:** How models are ported to NxD Inference.
  See :ref:`neuron-framework-autoport`.

Overview
--------

The Equivalence skill validates functional and numerical equivalence between a source
(reference) neural network implementation and a target (ported) implementation. It does
**not** perform the actual porting work — it verifies that an existing port is correct
through progressive stages of testing, localization, and debugging.

The skill is designed for workflows where a model has been migrated between:

- **Frameworks:** HuggingFace to NxD Inference
- **Hardware targets:** CPU to Neuron (Trainium)
- **Precision regimes:** FP32 to BF16, FP32 to MXFP4/INT8
- **Execution modes:** single TP degree to multi-TP degree sharding

It works with dense transformer models (decoder-only, encoder-decoder), Mixture of Experts
(MoE) models, models with novel attention mechanisms (sliding window, grouped query attention,
multi-latent attention), models requiring weight dequantization (MXFP4, INT8), and
cross-framework ports with precision regime changes.

Scope: causal-LM models only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   This pipeline assumes a model with a sampler and a token-generation output. Teacher-forced
   token/logit comparison, the ``TARGET_CAUSAL_CLASS`` input, and the R-ratio on logits all
   depend on a next-token distribution existing.

   It has **no defined behavior for a pooling or embedding model** — one served with
   ``--runner pooling``, with no ``lm_head`` or sampler, whose output is a fixed-size
   embedding vector. Such a model has no ``ForCausalLM`` class to supply for
   ``TARGET_CAUSAL_CLASS`` and no logits to compare, so the standard flow has no valid
   inputs. Do not force-fit one into this skill.

For a pooling or embedding model, validate with embedding cosine similarity against a
trusted reference instead:

1. Run the target and the reference (typically HuggingFace CPU-eager, BF16) over the same
   set of diverse prompts — short factual, technical, narrative, and code, so that
   different attention and pooling behavior is exercised.
2. Compute ``cos(embedding_target, embedding_reference)`` per prompt.
3. Treat **> 0.999 as a pass** for a well-ported BF16 model. Correct ports typically land
   at 0.9999 or better, and tensor-parallel sharding should not measurably move this number.
4. Also check a **cross-prompt discriminative** cosine — two unrelated prompts compared
   against each other should score far lower (roughly 0.3 versus 0.9999). A pooling bug
   that collapses every output toward a constant vector still yields a plausible
   same-prompt cosine, and only this cross-check catches it.
5. If the model supports batched or packed prefill, verify **no cross-sequence leakage**:
   a sequence's embedding computed alone versus computed alongside others should match to
   ~1.000000. This catches bidirectional-attention bugs where padding or a neighboring
   sequence leaks into a query's attention window.

This is a lighter-weight, model-specific check — not this skill's eight-stage pipeline. Do
not route it through ``run_stage0.py``, ``run_stage5.py``, or the other stage scripts, which
all assume a ``ForCausalLM`` class exists.

The workflow has eight stages:

1. **Structural scaffolding** — build model trees and create a component mapping
   between source and target architectures.
2. **Smoke testing** — quick liveness check using greedy token matching to verify the port
   produces coherent output.
3. **Component-level testing** — isolate each mapped component using the 3-tensor R-ratio
   method to identify which components diverge.
4. **Fault localization** — automatically classify root causes and rank suspect components.
5. **Debugging and patching** — fix failing components with standalone monkey patches
   without modifying the original port.
6. **End-to-end comparison** — verify the assembled model with real weights under teacher
   forcing using R-ratio, cosine similarity, and KL divergence.
7. **Downstream evaluation** — confirm production readiness using industry-standard
   benchmarks.

.. _equiv-prerequisites:

Hardware and software requirements
-----------------------------------

* **Instance type:** ``trn1.32xlarge`` (32 NeuronCores, 16 GB per core) or equivalent
  Trainium instance. CPU-mode testing (Stages 0, 2, 3, 4) can run on any instance.
* **Neuron SDK:** Version 2.28+ with the following system packages installed:

  - ``aws-neuronx-dkms``
  - ``aws-neuronx-runtime-lib``
  - ``aws-neuronx-collectives``
  - ``aws-neuronx-tools``

* **Python:** 3.10 or later.
* **Neuron SDK Python packages:**

  - ``neuronx-distributed-inference`` (0.8.x)
  - ``neuronx-distributed`` (0.17.x)
  - ``transformers`` (4.57+)
  - ``torch`` (2.x+)
  - ``numpy``
  - ``matplotlib``

* **Model weights:** Downloaded from HuggingFace Hub or available locally.
* **Compiled model:** A compiled (NEFF) version of the target model for device-mode
  testing (Stages 1, 5, 6, 7).
* **Disk space:** Sufficient for model weights, compiled artifacts, and experiment
  outputs (typically 2-5x the model size).

.. note::
   Stages 0, 2, 3, and 4 run in **CPU mode** (``NXD_CPU_MODE=1``, TP=1) and do not require
   Neuron hardware. Stages **1**, 5, 6, and 7 require a compiled model and Neuron device
   access.

   Stage 1 is a device stage despite its low stage number — the smoke test runs the
   compiled model through the adapter's ``device_inference()`` and needs
   ``COMPILED_MODEL_PATH``. Stages 0-4 are therefore **not** a contiguous hardware-free
   block.

.. _equiv-parameters:

Inputs
------

Before the agent begins the workflow, it collects these required parameters from the user:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``SOURCE_MODEL_PATH``
     - Path to reference model weights in HuggingFace format.
   * - ``COMPILED_MODEL_PATH``
     - Path to the compiled/quantized target model (NEFF artifacts).
   * - ``TARGET_MODELING_FILE``
     - Path to the target port's modeling Python file.
   * - ``TARGET_INNER_CLASS``
     - Inner model class name (extends ``NeuronBaseModel``).
   * - ``TARGET_CAUSAL_CLASS``
     - ``ForCausalLM`` wrapper class name.
   * - ``TARGET_CONFIG_CLASS``
     - ``InferenceConfig`` class name.
   * - ``VENV``
     - Path to Python virtual environment with torch and neuronx packages.
   * - ``EXP_DIR``
     - Experiment output directory for all artifacts.

.. _equiv-concepts:

Key concepts
------------

The R-ratio metric
^^^^^^^^^^^^^^^^^^

The R-ratio is the core metric used throughout the skill to quantify divergence:

.. code-block:: text

   R = ||target - source_fp32||_F / (||source_bf16 - source_fp32||_F + ε)

Where:

- ``source_fp32`` is the reference implementation running in FP32 (ground truth).
- ``source_bf16`` is the reference implementation running in BF16 (precision baseline).
- ``target`` is the target port running in BF16 (under test).
- ``|| . ||_F`` is the Frobenius norm (L2 norm of the flattened tensor).
- ``ε`` is a small constant to avoid division by zero.

The denominator measures the *expected* precision loss from FP32 to BF16 — the irreducible
error from the precision downgrade. The numerator measures the *actual* error of the port.
An R-ratio near 1.0 means the port introduces no additional error beyond the precision
baseline.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - R-ratio
     - Interpretation
   * - ≈ 1.0
     - Port matches precision baseline. No porting bug.
   * - < 1.2
     - Within acceptable tolerance. Minor TP rounding or kernel differences.
   * - 1.2 – 3.0
     - Possible porting bug. Missing multiplier or precision ordering issue.
   * - 3.0 – 10.0
     - Likely porting bug. Missing multiplier or precision ordering issue.
   * - >> 10
     - Missing algorithm or wrong formula (e.g., YaRN scaling absent from RoPE).
   * - >> 100
     - Completely wrong computation.
   * - < 1.0
     - Over-precision. Extra ``.float()`` calls not present in reference.

The 3-tensor comparison method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every component test produces three outputs from the same input:

1. **ref_fp32** — source (HuggingFace) model class, FP32 weights, FP32 input.
2. **ref_bf16** — source model class, BF16 weights, BF16 input.
3. **target_bf16** — target (Neuron port) model class, BF16 weights, BF16 input.

All three share the same FP32 weights (with BF16 versions created by downcasting). This
isolates the porting error from precision error: the denominator of R captures only
precision drift, while the numerator captures precision drift *plus* any porting bugs.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - When to use
     - Baseline
   * - **3-tensor**
     - Reference can run in FP32
     - Precision error from FP32 to BF16 downgrade
   * - **2-tensor**
     - Reference can only run in target precision
     - Machine-epsilon perturbation baseline

Expected structural differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When comparing HuggingFace and NxD Inference model trees, these differences are
**expected** and do not indicate bugs:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - HuggingFace
     - Neuron Port
     - Reason
   * - ``nn.Linear``
     - ``ColumnParallelLinear`` / ``RowParallelLinear``
     - Tensor parallel sharding
   * - ``nn.Embedding``
     - ``ParallelEmbedding``
     - Embedding sharded across TP ranks
   * - Flat q/k/v projections
     - Wrapped in ``GroupQueryAttention_QKV``
     - NxDI attention framework
   * - Single ``RotaryEmbedding`` at model level
     - Per-layer ``RotaryEmbedding``
     - Implementation choice
   * - ``XxxRMSNorm`` (source class)
     - ``LlamaRMSNorm`` (CPU) or ``CustomRMSNorm`` (device)
     - Framework normalization
   * - Fused ``gate_up_proj``
     - Split ``gate_proj`` + ``up_proj``
     - TP requires separate sharding
   * - (none)
     - ``SPMDRank``, ``KVCacheManager``
     - Neuron-specific infrastructure

Differences that **do** indicate bugs:

- Missing modules (norm layer absent in port)
- Extra unexpected modules with no framework explanation
- Wrong nesting (MLP inside attention instead of parallel)
- Mismatched layer counts (47 instead of 48)
- Missing activation functions

.. _equiv-workflow:

Workflow
--------

The skill follows a strict 8-stage sequential workflow. Stages must not be skipped,
reordered, or parallelized.

.. _equiv-stage0:

Stage 0: Structural scaffolding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build the alignment map between source and target model hierarchies.

**Purpose:** Understand both model structures and create a formal mapping between their
components.

Build model trees
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   source ${VENV}/bin/activate
   PYTHONPATH=${SCRIPTS_DIR} python3 ${SCRIPTS_DIR}/run_stage0.py \
     --source-model-path ${SOURCE_MODEL_PATH} \
     --target-model-path ${SOURCE_MODEL_PATH} \
     --target-module-file ${TARGET_MODELING_FILE} \
     --target-inner-class ${TARGET_INNER_CLASS} \
     --target-config-class ${TARGET_CONFIG_CLASS} \
     --output-dir ${EXP_DIR}/model_tree

The script instantiates the target in CPU mode (``NXD_CPU_MODE=1``, TP=1) to produce a
structure-only comparison without device dependencies.

**Outputs:**

.. code-block:: text

   ${EXP_DIR}/model_tree/
   ├── model_tree_source.json             # Compressed source tree
   ├── model_tree_source_full.json        # Uncompressed source tree
   ├── model_tree_source_pretty.txt       # ASCII pretty-print
   ├── model_tree_source_flat_paths.txt   # Flat module path list
   ├── model_tree_target.json             # Compressed target tree
   ├── model_tree_target_full.json        # Uncompressed target tree
   ├── model_tree_target_pretty.txt       # ASCII pretty-print
   └── model_tree_target_flat_paths.txt   # Flat module path list

Create component mapping
~~~~~~~~~~~~~~~~~~~~~~~~

Manually compare the printed trees and create ``${EXP_DIR}/component_mapping.json``. This
file maps each source module (or group of modules) to its target equivalent(s).

The mapping uses an array format where each entry is a pair of ``[source_modules, target_modules]``
with indexed variables (``{i}`` for layer indices) and reasoning:

- **One-to-one:** ``["model.layers.{i}.norm"]`` maps to ``["model.language_model.layers.{i}.norm"]``
- **One-to-many (fused):** ``["model.q_proj", "model.k_proj", "model.v_proj"]`` maps to ``["model.qkv_proj"]``
- **No counterpart:** Document the reasoning (framework scaffolding, TP-specific structure)

Detect CPU vs device class divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 ${SCRIPTS_DIR}/detect_class_divergence.py \
     --target-module-file ${TARGET_MODELING_FILE} \
     --output ${EXP_DIR}/class_divergence_report.json

This scans the target modeling file for patterns where different classes are used in CPU
mode versus device mode:

- **Factory functions** (``get_rmsnorm_cls()``) that branch on ``NXD_CPU_MODE`` or ``on_cpu``
- **Conditional assignments** (``self.norm = ClassA() if cpu else ClassB()``)
- **NKI kernel imports** (e.g., ``LlamaRMSNorm`` on CPU, ``CustomRMSNorm`` on device)

Components with class divergence require **dual testing** in Stage 2 — one test for the CPU
class and one for the device class.

.. _equiv-stage1:

Stage 1: Smoke test
^^^^^^^^^^^^^^^^^^^^

Quick liveness check — does the port produce coherent output?

.. code-block:: bash

   PYTHONPATH=${SCRIPTS_DIR} python3 ${SCRIPTS_DIR}/run_stage1.py \
     --model-path ${SOURCE_MODEL_PATH} \
     --compiled-model-path ${COMPILED_MODEL_PATH} \
     --model-class ${TARGET_MODELING_FILE}:${TARGET_CAUSAL_CLASS} \
     --config-class ${TARGET_MODELING_FILE}:${TARGET_CONFIG_CLASS} \
     --num-tokens 32 \
     --output ${EXP_DIR}/results/stage1.json

The script runs 10-prompt greedy token matching and computes per-position distribution
metrics: cosine similarity, KL divergence, top-k agreement, and relative L2 error.

**Interpreting results:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Match rate
     - Meaning
     - Action
   * - > 30%
     - Liveness threshold met
     - Continue to Stage 2
   * - 100% on most prompts
     - Normal BF16 precision drift
     - Continue to Stage 2
   * - < 30%
     - Catastrophic failure
     - Proceed to Stage 2 for localization

High cosine similarity (> 0.95) with low token match suggests margin-sensitive divergence
— the top two token probabilities are close, and BF16 rounding flips the argmax. This is
expected behavior and not a bug.

.. note::
   Stage 1's gate is the 30% liveness threshold above — ``run_stage1.py --pass-threshold``,
   default ``0.30``. It is deliberately permissive: this stage only asks whether the port
   runs and produces coherent text, not whether it is numerically correct. Correctness is
   established by Stages 2 and 5/6.

.. _equiv-stage2:

Stage 2: Component-level testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test each mapped component using the 3-tensor R-ratio method to isolate which component(s)
diverge.

Set up test infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Copy the comparison utility into the test directory:

   .. code-block:: bash

      cp ${SCRIPTS_DIR}/tensor_compare.py ${EXP_DIR}/tests/

2. Create ``conftest.py`` from the provided template. Fill in model-specific constants:
   ``HIDDEN_SIZE``, ``NUM_HEADS``, ``NUM_KV_HEADS``, ``VOCAB_SIZE``, ``INTERMEDIATE_SIZE``,
   and other values from the model's ``config.json``.

3. Write one test file per component, ordered bottom-up from simplest to most complex:
   ``test_00_rmsnorm.py``, ``test_01_embedding.py``, ``test_02_linear.py``, etc.

Write component tests
~~~~~~~~~~~~~~~~~~~~~

Each test follows the 3-tensor pattern:

.. code-block:: python

   def test_component_name():
       torch.manual_seed(42)
       weight_fp32 = torch.randn(OUT_DIM, IN_DIM)

       # ref_fp32: Source class, FP32 weights
       ref_fp32 = SourceClass(config)
       ref_fp32.weight.data.copy_(weight_fp32)

       # ref_bf16: Source class, BF16 weights
       ref_bf16 = SourceClass(config)
       ref_bf16.weight = nn.Parameter(weight_fp32.to(torch.bfloat16))

       # target_bf16: Target port's class, BF16 weights
       target_bf16 = TargetClass(neuron_config)
       target_bf16.weight = nn.Parameter(weight_fp32.to(torch.bfloat16))
       target_bf16.eval()

       x = torch.randn(BS, SEQ_LEN, IN_DIM)
       with torch.no_grad():
           out1 = ref_fp32(x.float()).float()
           out2 = ref_bf16(x.to(torch.bfloat16)).float()
           out3_raw = target_bf16(x.to(torch.bfloat16))
           out3 = out3_raw[0].float() if isinstance(out3_raw, tuple) else out3_raw.float()

       result = compare_3tensors(out1, out2, out3)
       assert check_3tensor_result(result, "component_name", TOLERANCE_RATIO)

**Critical rules for test writing:**

- ``ref_fp32`` and ``ref_bf16`` use the **source model's class** (HuggingFace).
- ``target_bf16`` uses the **target port's actual class** (may differ from source).
- All three share the same FP32 weights (BF16 versions created by downcasting).
- Use ``nn.Parameter()`` replacement for ``ColumnParallelLinear`` (not ``.copy_()``).
- Set ``.eval()`` mode on Neuron modules with ``pad=True``.
- Handle tuple outputs: ``out = out[0] if isinstance(out, tuple) else out``.
- Align shapes before comparison for fused components (QKV, gate/up projections).
- Cast all outputs to ``.float()`` before comparison.
- Check ``class_divergence_report.json`` — write **dual tests** for components with
  CPU/device class differences.

Run component tests
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   NXD_CPU_MODE=1 python3 ${SCRIPTS_DIR}/run_stage2.py \
     --tests-dir ${EXP_DIR}/tests \
     --tau-r 1.2 \
     --output ${EXP_DIR}/results/stage2.json

**Decision:** If all R < 1.2, proceed to Stage 5 (E2E). If any R >= 1.2, proceed to
Stage 3 for fault localization.

.. _equiv-stage3:

Stage 3: Fault localization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyze Stage 2 R-ratios to identify where divergence originates and classify root causes.

.. code-block:: bash

   python3 ${SCRIPTS_DIR}/run_stage3.py \
     --stage2-output ${EXP_DIR}/results/stage2.json \
     --tau-r 1.2 \
     --output ${EXP_DIR}/results/stage3.json

Change-point detection
~~~~~~~~~~~~~~~~~~~~~~

The script identifies two divergence patterns:

- **Spike:** High R at a single point that returns to baseline at the next component.
  Indicates an alignment artifact or transient error.
- **Step:** High R that persists for all subsequent components. Indicates a
  functional bug whose error propagates downstream.

The earliest step-pattern point is the primary fault candidate.

Root-cause classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - R magnitude
     - Likely cause
     - Examples
   * - R >> 10
     - Missing algorithm or wrong formula
     - YaRN scaling absent from RoPE, MoE routing ignored, masking wrong
   * - 1.2 < R < 3
     - Precision ordering or missing multiplier
     - Variance computed in BF16 instead of FP32, attention scaling omitted
   * - R < 1.0
     - Over-precision (unintended FP32 upcast)
     - Extra ``.float()`` call not present in reference

The output is a ranked list of suspect components with: component name, R-ratio,
divergence pattern (spike or step), root cause label, description, and mapped module paths.

.. _equiv-stage4:

Stage 4: Debug and patch
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix failing components with standalone monkey patches. This is the **only** stage where
code changes are made, and the original port is never modified directly.

Debugging workflow
~~~~~~~~~~~~~~~~~~

1. Read the Stage 3 fault localization report.
2. Compare the HuggingFace and Neuron implementations side-by-side:

   - Config parameters consumed by HuggingFace but missing in Neuron.
   - Operations present in one implementation but not the other.
   - Dtype casting differences.

3. Write a standalone monkey-patch file.
4. Re-run Stage 2 with the patch applied.
5. Verify the R-ratio drops to approximately 1.0.

Patch structure
~~~~~~~~~~~~~~~

.. code-block:: python

   def apply_component_patch():
       """Monkey-patch TargetClass to fix the issue. Call BEFORE instantiation."""
       from modeling_xxx import TargetClass
       if getattr(TargetClass, "_patched", False):
           return  # Idempotent guard

       _original_init = TargetClass.__init__
       def _patched_init(self, config):
           _original_init(self, config)
           # Fix: compute corrected values

       def _patched_forward(self, *args, **kwargs):
           # Fix: use corrected computation
           pass

       TargetClass.__init__ = _patched_init
       TargetClass.forward = _patched_forward
       TargetClass._patched = True

**Key rules:**

- Never modify the original port files. All fixes are delivered as standalone patches.
- Include an idempotent guard (``_patched`` flag) to prevent double-patching.
- Apply patches before model instantiation.
- If a patch fixes one module but breaks a downstream composite, the fix is incomplete —
  re-run the full bottom-up test suite.

Common pitfalls
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pitfall
     - Solution
   * - Config parameter gaps
     - Derive missing values from known config fields.
   * - Precision ordering
     - Scaling must be applied before BF16 cast, not after.
   * - Buffer assignment
     - ``register_buffer("name", None)`` resists direct assignment — store on wrapper instead.
   * - Output shape conventions
     - Match the target's shape format so downstream code works.
   * - Dtype mismatch
     - No extra ``.float()`` calls, no missing ``.float()`` calls — match reference exactly.

Repeat Stage 4 until all component R-ratios are below the threshold (default 1.2).

.. _equiv-stage5:

Stage 5: End-to-end comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify the assembled model with real weights under teacher forcing.

.. code-block:: bash

   PYTHONPATH=${SCRIPTS_DIR} \
   python3 ${SCRIPTS_DIR}/run_teacher_forced_comparison.py \
     --model-path ${SOURCE_MODEL_PATH} \
     --compiled-model-path ${COMPILED_MODEL_PATH} \
     --model-class ${TARGET_MODELING_FILE}:${TARGET_CAUSAL_CLASS} \
     --config-class ${TARGET_MODELING_FILE}:${TARGET_CONFIG_CLASS} \
     --num-tokens 32 \
     --output ${EXP_DIR}/results/teacher_forced.json

Teacher forcing explained
~~~~~~~~~~~~~~~~~~~~~~~~~

At each generation position *t*, all three models (source FP32, source BF16, target BF16)
receive the **same prefix tokens** — taken from the source FP32 greedy output. This ensures
logits are compared under identical contexts and prevents trajectory divergence from
contaminating per-position metrics.

.. _equiv-stage6:

Stage 6: Distributional and semantic validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This stage is combined with Stage 5 in a single script invocation. It adds two additional
conditions beyond the E2E R-ratio:

- **Condition B (Cosine similarity):** ``cos(v_source, v_target) >= θ`` (default θ = 0.95)
- **Condition C (KL divergence):** ``D_KL(P_source || P_target) <= δ`` (calibrated from known-good ports)

Pass criteria
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Threshold
   * - E2E R-ratio (p95)
     - < 1.2 (default τ_R)
   * - Cosine similarity (p5)
     - >= 0.95 (default θ)
   * - KL divergence (p95)
     - <= δ (calibrated)
   * - Top-1 agreement
     - > 50%

Interpreting Stage 5/6 results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Scenario
     - Likely cause
     - Action
   * - Stage 2 all-pass + Stage 5 fail
     - Compilation-induced divergence (operator fusion, kernel numerics)
     - Not a porting bug. Capture the compiled artifacts and the Stage 2/Stage 5 result
       JSON, and open an issue on the Neuron SDK GitHub repository.
   * - Stage 2 fail + Stage 5 fail
     - Porting bug propagates to E2E
     - Fix via Stage 4 first
   * - Condition B pass + Condition C fail
     - Logit directions agree, probability mass differs
     - Threshold calibration needed
   * - Stage 5 fail, all components clean
     - Unmapped component or different execution path on device
     - Run ``detect_class_divergence.py``; add device-specific tests

.. _equiv-stage7:

Stage 7: Downstream task evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Confirm the port remains usable for production workloads using industry-standard benchmarks.

.. code-block:: bash

   python3 ${SCRIPTS_DIR}/run_stage7.py \
     --bench-config ${EXP_DIR}/bench_config.yaml \
     --output-dir ${EXP_DIR}/results/stage7 \
     --tolerance 0.02

Benchmark configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     model_class: "path/to/modeling.py:NeuronXxxForCausalLM"
     config_class: "path/to/modeling.py:XxxInferenceConfig"
     model_path: "/path/to/hf_model"
     compiled_model_path: "/path/to/compiled_model"

   benchmarks:
     lm_eval:
       accuracy:
         tasks: ["gsm8k_cot", "mmlu_pro"]
         limit: 200
         use_chat: true

   run_hf_baseline: true

**Pass criteria:** Score regression <= 2 percentage points on all tasks.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Result
     - Meaning
     - Action
   * - All tasks within tolerance
     - Port is production-ready
     - PASS
   * - Math/reasoning tasks fail, knowledge passes
     - Precision-sensitive computation affected
     - Return to Stage 4
   * - All tasks fail
     - Fundamental porting issue
     - Return to Stage 2

.. _equiv-vllm-neuron:

Validate a vLLM Neuron port
---------------------------

The Equivalence *functionality* is stack-agnostic: the eight-stage workflow, the 3-tensor
method, and the progressive structure → component → end-to-end logic apply no matter which
serving stack your target port uses. What is **not** shared is the comparison machinery
itself. When the target is a `vLLM Neuron <https://github.com/vllm-project/vllm-neuron>`_
port, the skill delegates the actual three-way tensor comparison and the failure
diagnostics to vLLM Neuron's own ``accuracy`` and ``accuracy_debugger`` modules rather than
the bundled Frobenius-norm R-ratio.

The integration points that touch the target model — distributed init, model construction,
weight loading, the forward signature, device inference, and the comparison metrics — are
isolated behind a *stack adapter*. This section covers what differs when your target port
imports from ``vllm_neuron`` instead of ``neuronx_distributed_inference``. If you are
validating an NxD Inference port, you can skip it — the defaults already apply to you.

Select the vLLM Neuron stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The skill supports two target stacks today:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Stack
     - Adapter
     - Target imports
   * - NxD Inference (default)
     - ``scripts/adapters/nxdi.py``
     - ``from neuronx_distributed_inference.*``
   * - vLLM Neuron
     - ``scripts/adapters/vllm_neuron.py``
     - ``from vllm_neuron.*``

The skill auto-detects the stack by reading the imports in your ``TARGET_MODELING_FILE``.
If the modeling file imports from ``vllm_neuron``, the vLLM Neuron adapter is selected
automatically. To select it explicitly, set ``target_stack`` in the experiment config:

.. code-block:: json

   {
       "target_stack": "vllm_neuron",
       "model_class": "modeling_xxx.py:XxxForCausalLM",
       "config_class": "config.py:XxxConfig"
   }

.. note::
   vLLM Neuron uses a separate config module. Point ``config_class`` at the ``config.py``
   that defines the model config, not at the modeling file, and the adapter loads it with
   ``ConfigClass.from_configs(hf_config)`` rather than ``from_pretrained()``.

Weight differences
^^^^^^^^^^^^^^^^^^

The single most important difference is that **all linear weights in vLLM Neuron are
transposed** relative to HuggingFace, and the separate q/k/v projections are fused into a
single ``qkv_proj_weight``. The adapter applies these transforms when it loads weights, but
you must account for them when you write component tests.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Weight
     - HuggingFace shape
     - vLLM Neuron transform
   * - ``gate_proj`` / ``up_proj``
     - ``[I, H]``
     - ``.t()`` → ``[H, I]``
   * - ``down_proj``
     - ``[H, I]``
     - ``.t()`` → ``[I, H]``
   * - ``o_proj``
     - ``[H, q]``
     - ``.t()`` → ``[q, H]``
   * - Fused QKV
     - ``q_proj``, ``k_proj``, ``v_proj`` separate
     - ``cat([Q.t(), K.t(), V.t()], dim=-1)`` → ``[H, q+2kv]``
   * - Norms, embedding, LM head
     - ``[H]`` / ``[V, H]``
     - Direct copy (no transpose)

vLLM Neuron also names weights differently: HuggingFace uses ``module.weight`` (an
``nn.Parameter`` inside an ``nn.Linear``), while vLLM Neuron uses a bare ``module_weight``
parameter. For example, ``self_attn.q_proj.weight`` becomes ``self_attn.qkv_proj_weight``.

.. warning::
   Do not call vLLM Neuron's own ``load_weights()`` for equivalence testing. It calls
   ``get_current_vllm_config()``, which only resolves inside the vLLM serving stack. The
   adapter bypasses it and maps HuggingFace weights manually with the transposes and QKV
   fusion shown above.

Forward signature and shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

vLLM Neuron's forward signature is different from HuggingFace's. The adapter constructs the
``attn_metadata`` and ``sampling_positions`` arguments for you, but you should understand
the shape conventions when comparing outputs:

.. code-block:: python

   model(
       input_ids,           # [seq_len] — no batch dimension
       positions,           # [seq_len] — position IDs
       attn_metadata,       # dict[str, dict] — per-layer attention config
       sampling_positions,  # [num_positions] — which positions return logits
   )

Two shape rules follow from this:

- **Squeeze the batch dimension before comparison.** HuggingFace outputs ``[1, T, H]``;
  vLLM Neuron outputs ``[T, H]``. Align the two before comparing.
- **The KV cache layout is** ``[num_blocks, kv_heads_per_rank, block_size, head_dim]`` —
  heads come *before* ``block_size``. Getting this order wrong is a common source of wasted
  debugging time.

Component test differences
^^^^^^^^^^^^^^^^^^^^^^^^^^

When you write Stage 2 component tests for a vLLM Neuron target, account for these
per-component differences:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Component
     - HuggingFace forward
     - vLLM Neuron forward
     - Weight setup
   * - MLP
     - ``forward(x)``
     - ``forward(x, is_prefill=True)``
     - ``.t()`` on gate/up/down
   * - Q/K/V
     - 3 separate ``F.linear()``
     - Single fused ``qkv_proj()``
     - ``cat([Q.t(), K.t(), V.t()], dim=-1)``
   * - O projection
     - ``F.linear(x, o_proj.weight)``
     - ``o_proj(x, o_proj_weight)``
     - ``.t()``
   * - Full attention
     - Independently testable
     - Needs KV cache + ``attn_metadata``
     - Test QKV and O projections separately

.. important::
   Full attention is **not** independently testable for vLLM Neuron, because it requires a
   live KV cache and ``attn_metadata``. Test the QKV projection and the O projection as
   separate components instead.

Use the ``conftest_vllm_template.py`` template (not the default ``conftest_template.py``)
for the test infrastructure. It initializes the vLLM distributed environment instead of the
NxDI one:

.. code-block:: python

   from vllm.distributed.parallel_state import (
       init_distributed_environment,
       initialize_model_parallel,
   )
   init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
   initialize_model_parallel(tensor_model_parallel_size=1)

Environment variables and tensor parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPU-mode and device-mode testing require different environment variables for vLLM Neuron:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Context
     - Required environment variables
   * - CPU testing (Stages 0, 2, 3, 4)
     - ``NXD_CPU_MODE=1``, ``WORLD_SIZE`` = TP degree (``1`` for CPU-mode testing),
       ``MASTER_ADDR=localhost``, ``MASTER_PORT=8099``, ``RANK=0``
   * - Device testing (Stages 1, 5, 6, 7)
     - ``NEURON_SKIP_EFA_AFFINITY=1``, ``TOKENIZERS_PARALLELISM=false``, and
       ``NXD_CPU_MODE`` **unset** — a value left over from a CPU-mode stage keeps the model
       off the device
   * - Expert-parallel (EP) models
     - Add ``NXDI_SWITCH_CC=1``

The adapter sets these itself, so you normally do not export them by hand. It also points
``NEURON_COMPILE_CACHE_URL`` at ``${COMPILED_MODEL_PATH}/neuron/compile_cache`` when that
directory exists, so the device stages reuse pre-compiled NEFFs instead of recompiling.

.. warning::
   On ``trn2.48xlarge``, **TP=1 can cause a SIGSEGV with no diagnostic output** for models
   that require multi-core sharding. The minimum TP degree is model-specific (it depends on
   model size and the per-core memory budget). Always read the model's example run script
   (``examples/vllm_neuron/models/<MODEL>/run.py``) for the correct tensor parallel size.

Comparison metrics
^^^^^^^^^^^^^^^^^^

The vLLM Neuron adapter does **not** compute the Frobenius-norm R-ratio. It delegates the
three-way comparison to ``assert_close_three_way`` from ``vllm_neuron.accuracy.testing``.

The three tensors being compared are the same as the NxDI path:

1. **Baseline** — the HuggingFace reference in FP32 (ground truth).
2. **Expected** — the HuggingFace reference in BF16 (precision baseline).
3. **Actual** — the vLLM Neuron port in BF16 (target under test).

The function computes the element-wise errors ``expected - baseline`` (the irreducible
precision error) and ``actual - baseline`` (the port's total error), then compares the two
error distributions using these metrics:

- **BC (Bhattacharyya Coefficient)** — overlap between the port's error distribution and
  the precision-baseline error distribution. A BC near 1.0 means the port's errors look
  statistically identical to the precision-only errors — no porting bug.
- **σ-ratio** — ``RMS(actual_errors) / RMS(baseline_errors)``, aggregated across all
  inputs. A σ-ratio at or below 1.0 means the port is no less accurate than the baseline.

A comparison passes when ``BC >= bc_threshold`` **or** ``σ-ratio <= 1.0``, *and* both the
L-infinity and L2 error ratios stay under their guards:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Default threshold
   * - BC (Bhattacharyya Coefficient)
     - >= 0.99
   * - σ-ratio
     - <= 1.0
   * - L-infinity ratio
     - < 5.0
   * - L2 ratio
     - < 3.0

Diagnostic tools
^^^^^^^^^^^^^^^^

Because vLLM Neuron ships its own ``accuracy_debugger`` module, the adapter exposes
additional diagnostics beyond the standard stages. The agent invokes these automatically
when a stage fails:

.. list-table::
   :header-rows: 1
   :widths: 28 52 20

   * - Diagnostic
     - What it does
     - Used by
   * - Logit validation (``LogitValPlugin``)
     - Compares device logits against the FP32 baseline per token.
     - Stage 5
   * - KV cache analysis (``KvCachePlugin``)
     - Three-way per-layer, per-position KV cache comparison (FP32 vs HF-dtype vs vLLM).
     - Stage 5
   * - Accuracy analysis (``LmEvalAnalyzer``)
     - Task-level accuracy with per-sample deviation tracking, run against a vLLM server.
     - Stage 7

These diagnostics are driven through ``run_prompt_analysis`` and ``run_task_analysis`` from
the ``accuracy_debugger`` module in
`vLLM Neuron <https://github.com/vllm-project/vllm-neuron>`_. See the
``vllm-neuron-adaptation.md`` and ``adapter-contract.md`` references in the
:ref:`knowledge base <equiv-knowledge-base>` for the full per-stack detail.

.. _equiv-file-organization:

File organization
-----------------

The Equivalence skill enforces a strict file organization:

.. code-block:: text

   ${EXP_DIR}/
   ├── model_tree/                        # Stage 0 outputs
   │   ├── model_tree_source.json         # Compressed source tree
   │   ├── model_tree_source_full.json    # Uncompressed source tree
   │   ├── model_tree_source_pretty.txt   # ASCII pretty-print
   │   ├── model_tree_source_flat_paths.txt
   │   ├── model_tree_target.json         # Compressed target tree
   │   ├── model_tree_target_full.json    # Uncompressed target tree
   │   ├── model_tree_target_pretty.txt   # ASCII pretty-print
   │   └── model_tree_target_flat_paths.txt
   ├── component_mapping.json             # Manual source-to-target mapping
   ├── class_divergence_report.json       # CPU vs device class branching
   ├── tests/                             # Stage 2 component tests
   │   ├── conftest.py                    # Shared infrastructure
   │   ├── tensor_compare.py              # Comparison utility (copied from scripts/)
   │   ├── test_00_rmsnorm.py             # Simplest component first
   │   ├── test_01_embedding.py
   │   ├── test_02_linear.py
   │   ├── test_03_rotary.py
   │   ├── test_04_mlp.py
   │   ├── test_05_attention_qkv.py
   │   ├── test_06_lm_head.py
   │   └── test_07_decoder_layer.py       # Most complex last
   ├── patches/                           # Stage 4 monkey patches (one file per fix)
   │   ├── rmsnorm_patch.py
   │   └── rotary_patch.py
   ├── results/                           # Test results (JSON)
   │   ├── stage1.json
   │   ├── stage2.json
   │   ├── stage3.json
   │   ├── teacher_forced.json
   │   └── stage7/
   ├── bench_config.yaml                  # Stage 7 benchmark configuration
   └── EQUIVALENCE_REPORT.md              # Final aggregated report

For a vLLM Neuron target (see :ref:`equiv-vllm-neuron`), the layout is the same with two
additions:

* ``tests/conftest.py`` is created from ``conftest_vllm_template.py`` instead of the default
  ``conftest_template.py`` — the file name is the same, but it initializes the vLLM
  distributed environment.
* When the failure diagnostics run, the ``accuracy_debugger`` reports are written under
  ``results/``:

.. code-block:: text

   ${EXP_DIR}/results/
   ├── kv_analysis/                       # Stage 5 KV cache analysis (--kv-analysis-on-fail)
   └── stage7/
       └── prompt_diagnosis/              # Stage 7 per-prompt diagnosis (--diagnose-failures)

.. _equiv-tools-reference:

Tools reference
---------------

Tree builder
^^^^^^^^^^^^

**Module:** ``scripts/run_stage0.py``

Builds compressed and uncompressed model trees for both source and target architectures.
Instantiates the target model in CPU mode (``NXD_CPU_MODE=1``, TP=1) to avoid device
dependencies. Uses ``scripts/stage0_scaffolding.py`` for tree generation utilities.

Class divergence detector
^^^^^^^^^^^^^^^^^^^^^^^^^

**Module:** ``scripts/detect_class_divergence.py``

Scans the target modeling file for conditional class selection patterns (factory functions,
conditional assignments, NKI kernel imports). Produces a JSON report listing each divergence
with the CPU class, device class, and recommendation for dual testing.

Smoke test runner
^^^^^^^^^^^^^^^^^

**Module:** ``scripts/run_stage1.py``

Runs 10-prompt greedy token matching against an FP32 ``AutoModelForCausalLM`` reference,
computed in the script itself. The output JSON reports the overall and per-prompt token
match rates.

**NxDI targets only:** the compiled model is loaded through
``scripts/nxdi_compiled_loader.py``, which rebuilds the ``NeuronConfig`` from the compiled
directory's ``neuron_config.json`` using public NxDI API only. This loader does **not**
apply to vLLM Neuron targets — that adapter runs inference through ``vllm.LLM`` instead.

Component test runner
^^^^^^^^^^^^^^^^^^^^^

**Module:** ``scripts/run_stage2.py``

Discovers and executes all ``test_*.py`` files in the specified test directory. Collects
R-ratios and produces a pass/fail summary with the configured threshold (default τ_R = 1.2).

Fault localizer
^^^^^^^^^^^^^^^

**Module:** ``scripts/run_stage3.py``

Analyzes Stage 2 results using change-point detection to identify spike vs step patterns.
Classifies root causes and produces a ranked list of suspect components.

Tensor comparator
^^^^^^^^^^^^^^^^^

**Module:** ``scripts/tensor_compare.py``

Core utility for 3-tensor comparison. Computes R-ratio, generates QQ plots and histograms
for visual analysis, and provides the ``compare_3tensors()`` and ``check_3tensor_result()``
functions used by all component tests.

Teacher-forced comparator
^^^^^^^^^^^^^^^^^^^^^^^^^

**Module:** ``scripts/run_teacher_forced_comparison.py``

Runs Stages 5 and 6 in a single pass. Compares source FP32, source BF16, and target BF16
models under teacher forcing. Reports per-position R-ratio, cosine similarity, KL
divergence, and top-1 agreement.

Downstream evaluator
^^^^^^^^^^^^^^^^^^^^

**Module:** ``scripts/run_stage7.py``

Runs industry-standard benchmarks (via ``lm_eval``) and compares scores against a HuggingFace
baseline. Reports per-task accuracy with a configurable regression tolerance (default 2
percentage points).

Calibration tool
^^^^^^^^^^^^^^^^

**Module:** ``scripts/run_calibration.py``

Computes threshold values (τ_R, θ, δ) from known-good ports. Use this to establish
project-specific thresholds rather than relying on defaults.

.. _equiv-debugging:

Common issues
-------------

Stage 0 failures
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - ``'NoneType' has no attribute 'windowed_context_encoding_size'``
     - Config validation requires ``neuron_config``. Pass ``neuron_config=NeuronConfig(...)`` to ``from_pretrained()``.
   * - ``intra_layer_model parallel group is not initialized``
     - ``run_stage0.py`` handles this automatically. If running manually, call ``init_process_group("gloo")`` then ``initialize_model_parallel(tp=1)``.
   * - ``Please initialize parallel processing via 'torchrun'``
     - Use ``tp_degree=1, world_size=1`` for structure inspection.
   * - ``No module named 'modeling_xxx'``
     - Verify ``--target-module-file`` path is correct and its parent directory is on ``sys.path``.

Stage 2 failures
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - Test import failures
     - Ensure ``tensor_compare.py`` is copied to the tests directory and ``conftest.py`` is present.
   * - ``TOLERANCE_RATIO`` not defined
     - Fill in all constants in ``conftest.py`` from the ``conftest_template.py``.
   * - Shape mismatch on comparison
     - Align shapes before calling ``compare_3tensors`` for fused components (QKV, gate/up projections).
   * - ``nn.Parameter()`` replacement failed
     - Use ``target.weight = nn.Parameter(weight.to(torch.bfloat16))`` instead of ``.copy_()``.

Stage 4 failures
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - Patch applied but test still fails
     - Ensure the ``_patched`` flag is set and the patch is idempotent.
   * - Downstream composite test breaks after patch
     - Incomplete fix — run full bottom-up test suite and patch both component and composite if needed.
   * - Buffer assignment does not persist
     - ``register_buffer()`` resists assignment. Store corrected value on the wrapper object instead.

Stage 5/6 failures
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Solution
   * - ``OnDeviceSamplingConfig`` required but missing
     - Add ``on_device_sampling_config=OnDeviceSamplingConfig(...)`` to ``NeuronConfig``.
   * - Device tensors have wrong shape
     - Device pads to full ``seq_len``. Slice device tensor: ``device_tensor[:, :hf_seq_len, :]``.
   * - ``embed_tokens`` comparison shows infinite error ratio
     - Embeddings are lookups (no computation), so ``baseline_err=0``. Check cosine similarity instead — cosine = 1.0 means PASS.

.. _equiv-guidelines:

Project guidelines
------------------

The Equivalence skill enforces these rules during the validation process:

- **Do NOT write tree generation, test runner, or validation scripts.** All major scripts
  are bundled and must be run as-is.
- **The only files the user creates are:** ``component_mapping.json`` (Stage 0, manual),
  ``test_NN_*.py`` test files (Stage 2, following templates), and monkey-patch files
  (Stage 4, debugging only).
- **Do NOT modify original model files.** All fixes are delivered as standalone patches.
- **Follow stage order strictly.** Do not skip ahead, reorder, or parallelize stages.
- **Show full output from every script run.** Do not summarize or truncate.
- **No try/except statements** in test files. Let errors surface directly.
- **No additional pip installs during the run.** Use only the packages already present in
  the virtual environment — the Neuron SDK packages listed under
  :ref:`equiv-prerequisites`. Satisfy every dependency before starting Stage 0.

.. _equiv-knowledge-base:

Knowledge base
--------------

The skill includes a curated knowledge base at ``references/`` containing solutions gathered
from prior validation sessions.

Foundational concepts
^^^^^^^^^^^^^^^^^^^^^

- ``equiv-concept.md`` — foundational concepts: 3-way comparison, R-ratio derivation,
  QQ plot interpretation.
- ``expected_structural_diffs.md`` — catalog of expected HuggingFace to Neuron
  structural differences.
- ``mapping_example.json`` — full worked example of a 31-component mapping (Llama4
  multimodal model).

Debugging guides
^^^^^^^^^^^^^^^^

- ``cpu-component-debugging.md`` — full workflow for CPU-level component debugging with
  patterns and pitfalls.
- ``device-component-debugging.md`` — XLA-compatible patch patterns for device-mode
  execution.
- ``device-e2e-debugging.md`` — device E2E with 1-layer isolation and fix-compile-verify
  cycle.
- ``cpu-e2e-debugging.md`` — CPU E2E with ``mp.spawn``, TP > 1, and bias restoration.
- ``dump-tensors.md`` — intermediate tensor capture methodology for per-layer comparison.
- ``debug-orchestration.md`` — the five-stage debugging escalation workflow, with stage
  gates and subagent delegation.
- ``enable-model-run.md`` — device compilation workflow, troubleshooting, and the BF16
  gloo fix. Prerequisite for the device stages.

Reporting
^^^^^^^^^

- ``report-template.md`` — ``EQUIVALENCE_REPORT.md`` structure and the completion gate that
  must be satisfied before a report is generated.

Multi-stack support
^^^^^^^^^^^^^^^^^^^

- ``adapter-contract.md`` — the five-method adapter interface for adding a serving stack,
  plus the optional diagnostic methods.
- ``vllm-neuron-adaptation.md`` — vLLM Neuron weight transpositions, forward signature, TP
  detection, KV cache shapes, and BC / σ-ratio metrics.

Case studies
^^^^^^^^^^^^

- ``debugging-case-study-gptoss.md`` — real worked example from GPT-OSS 20B showing
  error ratios, root causes, and patches applied.

Visual references
^^^^^^^^^^^^^^^^^

- ``example_plots/positive_samples/`` — QQ plots and histograms showing passing
  distributions (errors follow the 45-degree line).
- ``example_plots/negative_samples/`` — QQ plots and histograms showing failing
  distributions (divergent error patterns).

Related resources
-----------------

- :ref:`neuron-framework-autoport` — AI agent-driven model porting workflow.
- :ref:`nxdi-feature-guide` — NxD Inference features configuration reference.
- :ref:`nxdi-onboarding-models` — integrating custom models into NxD Inference.
- :ref:`nxdi-parallelism-user-guide` — tensor and sequence parallelism techniques.
- :ref:`moe-inference-deep-dive` — MoE architecture support in NxD Inference.
- :ref:`nxdi-vllm-user-guide-v1` — deploying models with vLLM on Neuron.
- `NxD Inference source code <https://github.com/aws-neuron/neuronx-distributed-inference>`_ — reference model implementations.
- `Neuron Agentic Development <https://github.com/aws-neuron/neuron-agentic-development>`_ — the Equivalence skill source and knowledge base.
