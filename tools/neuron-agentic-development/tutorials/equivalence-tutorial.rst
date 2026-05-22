.. meta::
   :description: Tutorial for validating a ported model using the Neuron Agentic Development Equivalence skill.
   :keywords: equivalence, tutorial, model validation, NxD Inference, Neuron Agentic Development, Trainium, R-ratio
   :date-modified: 2026-05-12

.. _equivalence-tutorial:

Tutorial: Validate a model port using the Equivalence skill
=============================================================

This tutorial walks you through validating a ported NxD Inference model against its
HuggingFace reference using the Equivalence agent. You will provide model parameters,
invoke the agent, and the agent will handle the rest: structural analysis, component-level
testing, fault localization, debugging, and end-to-end accuracy verification.

By the end of this tutorial you will have a validated model port with R-ratio metrics
confirming numerical equivalence between the Neuron implementation and the HuggingFace
reference.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites
-------------

Set up a Trainium instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a ``trn1.32xlarge`` instance (32 NeuronCores) or equivalent. Launch it from
the Neuron Deep Learning AMI (DLAMI) and SSH in.

Verify Neuron devices are available.

.. code-block:: bash

   neuron-ls

You should see 32 NeuronCores. If you see 0, your instance does not have Neuron hardware
attached. Stop here and fix that first.

.. note::
   Stages 0 through 4 of the Equivalence workflow run in CPU mode and do not require
   Neuron hardware. Only Stages 5 through 7 require a compiled model and device access.

Install Neuron Agentic Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have not already installed the package, follow the
:ref:`Getting Started guide <neuron-agentic-development-getting-started>`.

Make sure the deploy step completed.

.. code-block:: bash

   # For Claude Code
   deploy-neuron-agentic-development-to-claude

   # For Kiro
   deploy-neuron-agentic-development-to-kiro

Activate your Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate

Verify the required packages are installed.

.. code-block:: bash

   pip list | grep neuronx-distributed-inference

Have a completed model port
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Equivalence skill validates an existing port. It does not perform porting itself.
You need:

- A ported modeling file (e.g., ``modeling_arcee.py``) with NxD Inference classes.
- A compiled model (NEFF artifacts) from a successful compilation.
- The original HuggingFace model weights downloaded locally.

If you do not have a port yet, use the :ref:`Autoport skill <autoport-tutorial>` first.

Download your model weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the HuggingFace model you want to validate against. For this example we use a
small model to keep validation fast.

.. code-block:: bash

   mkdir -p agent_artifacts/data
   huggingface-cli download arcee-ai/AFM-4.5B-Base --local-dir agent_artifacts/data

Step 1. Gather your model parameters
--------------------------------------

The Equivalence agent needs nine pieces of information about your model. Gather these
before you start.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - What it is
     - Example value
   * - ``SOURCE_MODEL_PATH``
     - Path to reference model weights (HuggingFace format)
     - ``agent_artifacts/data``
   * - ``COMPILED_MODEL_PATH``
     - Path to the compiled target model (NEFF artifacts)
     - ``agent_artifacts/data/compiled_model``
   * - ``TARGET_MODELING_FILE``
     - Path to the target port's modeling Python file
     - ``neuron_port/modeling_arcee.py``
   * - ``TARGET_INNER_CLASS``
     - Inner model class name (extends ``NeuronBaseModel``)
     - ``NeuronArceeModel``
   * - ``TARGET_CAUSAL_CLASS``
     - ``ForCausalLM`` wrapper class name
     - ``NeuronArceeForCausalLM``
   * - ``TARGET_CONFIG_CLASS``
     - ``InferenceConfig`` class name
     - ``ArceeInferenceConfig``
   * - ``VENV``
     - Path to Python virtual environment with torch and neuronx packages
     - ``~/opt/aws_neuronx_venv_pytorch_2_9``
   * - ``MODEL_VALIDATION_DIR``
     - Path to the ``model_validation`` package directory
     - ``path/to/NeuroborosFoundations/model_validation``
   * - ``EXP_DIR``
     - Experiment output directory for all artifacts
     - ``agent_artifacts/equiv_arcee``

To find the class names, open the target modeling file and look for classes that extend
``NeuronBaseModel`` (inner class), ``NeuronBaseForCausalLM`` (causal class), and
``InferenceConfig`` (config class).

Step 2. Invoke the Equivalence agent
--------------------------------------

Open your agentic IDE (Claude Code or Kiro) on the Trainium instance.

Invoke the agent with your parameters.

.. code-block:: text

   Validate equivalence with inputs as SOURCE_MODEL_PATH is agent_artifacts/data,
   COMPILED_MODEL_PATH is agent_artifacts/data/compiled_model,
   TARGET_MODELING_FILE is neuron_port/modeling_arcee.py,
   TARGET_INNER_CLASS is NeuronArceeModel,
   TARGET_CAUSAL_CLASS is NeuronArceeForCausalLM,
   TARGET_CONFIG_CLASS is ArceeInferenceConfig,
   VENV is ~/opt/aws_neuronx_venv_pytorch_2_9,
   MODEL_VALIDATION_DIR is path/to/NeuroborosFoundations/model_validation,
   EXP_DIR is agent_artifacts/equiv_arcee

The agent confirms your parameters and starts working. You do not need to do anything
else. The agent runs through all stages automatically.

Step 3. What happens during validation
----------------------------------------

The agent works through eight stages. Here is what you will see at each one.

**Stage 0: Structural scaffolding.** The agent builds model trees for both the HuggingFace
reference and the Neuron port. It compares the two hierarchies and creates a component
mapping that pairs each source module with its target equivalent. It also detects
components that use different classes in CPU mode versus device mode. This takes a few
seconds.

**Stage 1: Smoke test.** The agent runs 10 prompts through both models with greedy decoding
and checks token match rate. This is a quick liveness check — if the port produces
coherent output, it passes. A match rate above 30% is sufficient to continue.

**Stage 2: Component-level testing.** The agent writes test files for each mapped
component using the 3-tensor R-ratio method. It tests normalization, embeddings, linear
projections, rotary encoding, MLP, attention, and full decoder layers — ordered from
simplest to most complex. Each test compares the source in FP32, source in BF16, and
target in BF16 using the same weights.

**Stage 3: Fault localization.** If any component has an R-ratio above 1.2, the agent
analyzes the pattern of divergence. It identifies whether errors are spikes (transient)
or steps (propagating), and classifies root causes: missing algorithms, precision
ordering issues, or over-precision.

**Stage 4: Debug and patch.** The agent fixes failing components by writing standalone
monkey patches. It never modifies the original port files. After each patch, it re-runs
the affected tests to verify the R-ratio drops to approximately 1.0. It iterates until
all components pass.

**Stage 5: End-to-end comparison.** The agent compares the full assembled model with real
weights under teacher forcing. At each token position, both models receive the same prefix
(from the FP32 reference), so logits are compared under identical contexts.

**Stage 6: Distributional validation.** The agent checks three conditions beyond the
R-ratio: cosine similarity (>= 0.95), KL divergence (<= calibrated threshold), and
top-1 agreement (> 50%). All must pass for the port to be considered equivalent.

**Stage 7: Downstream evaluation (optional).** The agent runs industry-standard
benchmarks (GSM8K, MMLU) and compares scores against the HuggingFace baseline. Scores
must not regress by more than 2 percentage points.

Step 4. Understand the R-ratio results
----------------------------------------

The R-ratio is the core metric. It measures how much additional error the port introduces
beyond the expected precision loss from FP32 to BF16.

.. code-block:: text

   R = ||target - source_fp32|| / (||source_bf16 - source_fp32|| + epsilon)

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - R-ratio
     - Interpretation
   * - near 1.0
     - Port matches the precision baseline. No porting bug.
   * - < 1.2
     - Within acceptable tolerance. Minor TP rounding or kernel differences.
   * - 1.2 to 3.0
     - Possible porting bug. Missing multiplier or precision ordering issue.
   * - 3.0 to 10.0
     - Likely porting bug. Investigate the root cause.
   * - >> 10
     - Missing algorithm (e.g., YaRN scaling absent from RoPE).
   * - >> 100
     - Completely wrong computation.
   * - < 1.0
     - Over-precision. Extra ``.float()`` calls not present in reference.

The agent considers the validation successful when all component R-ratios are below 1.2,
end-to-end metrics pass all three conditions, and downstream benchmarks show no
regression.

Step 5. Check the results
--------------------------

When the agent finishes, you will have these outputs.

.. code-block:: text

   agent_artifacts/equiv_arcee/
   ├── model_tree/                        # Stage 0: model structure analysis
   │   ├── model_tree_source_pretty.txt   # Human-readable source tree
   │   ├── model_tree_target_pretty.txt   # Human-readable target tree
   │   └── ...                            # JSON trees and flat paths
   ├── component_mapping.json             # Source-to-target component mapping
   ├── class_divergence_report.json       # CPU vs device class differences
   ├── tests/                             # Stage 2: component test files
   │   ├── conftest.py                    # Test infrastructure
   │   ├── tensor_compare.py             # Comparison utility
   │   ├── test_00_rmsnorm.py            # Normalization test
   │   ├── test_01_embedding.py          # Embedding test
   │   └── ...                            # More component tests
   └── results/                           # Test results (JSON)
       ├── stage1.json                    # Smoke test results
       ├── stage2.json                    # Component R-ratios
       ├── stage3.json                    # Fault localization (if needed)
       └── teacher_forced.json            # E2E comparison results

The ``results/`` directory contains the quantitative validation data. The ``tests/``
directory contains the component tests the agent wrote — you can inspect or re-run them
independently.

Step 6. Re-run validation after fixing your port (optional)
-------------------------------------------------------------

If you manually update your port after validation, you can re-run the Equivalence agent
to confirm the changes did not introduce regressions. Point it at the same ``EXP_DIR``
and it will regenerate the results.

You can also run individual stages manually. For example, to re-run only the component
tests after a code change:

.. code-block:: bash

   source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate
   NXD_CPU_MODE=1 python3 ${SCRIPTS_DIR}/run_stage2.py \
     --tests-dir agent_artifacts/equiv_arcee/tests \
     --tau-r 1.2 \
     --output agent_artifacts/equiv_arcee/results/stage2.json

Troubleshooting
---------------

The agent handles most issues automatically, but here are things that might require
your input.

**Agent asks for missing parameters.** If you forgot a parameter, the agent will ask
for it. Provide the value and it continues.

**Stage 0 fails with "NoneType has no attribute windowed_context_encoding_size".** The
target config requires ``neuron_config``. The bundled script handles this, but if you
see this error, verify your ``TARGET_CONFIG_CLASS`` is correct.

**Component tests import failures.** Make sure ``tensor_compare.py`` is in the tests
directory and ``conftest.py`` has all required constants filled in. The agent does this
automatically, but manual re-runs may need you to verify the test directory structure.

**Stage 5 fails but all Stage 2 tests passed.** This typically indicates
compilation-induced divergence (operator fusion or kernel numerics) rather than a porting
bug. The agent will note this in its output and may recommend escalating to the compiler
team.

**Validation keeps failing on the same component.** Check the
``class_divergence_report.json`` — the component may use a different class on device than
in CPU mode. The agent writes dual tests (CPU class and device algorithm) to catch this,
but complex NKI kernels may need manual inspection.

**No NeuronCores detected.** Run ``neuron-ls``. If it shows 0 devices, your instance
either does not have Neuron hardware or the driver is not loaded. Check that
``aws-neuronx-dkms`` is installed. Note that Stages 0 through 4 do not need hardware.

Related resources
-----------------

- :doc:`/tools/neuron-agentic-development/developer_guides/neuron-framework-equivalence` for the deep dive on how the Equivalence skill works internally.
- :doc:`/tools/neuron-agentic-development/tutorials/autoport-tutorial` for porting a model before validating it.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- :ref:`nxdi-vllm-user-guide-v1` for deploying validated models with vLLM.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_ for source code and issue tracker.
