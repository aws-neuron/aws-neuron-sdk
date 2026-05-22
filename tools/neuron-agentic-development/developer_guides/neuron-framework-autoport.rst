.. meta::
   :description: Port HuggingFace transformer models to NxD Inference for AWS Trainium using the Autoport skill
   :keywords: autoport, model porting, NxD Inference, NeuronX, Trainium, HuggingFace, transformer, compilation, validation
   :date-modified: 2026-05-11

.. _neuron-framework-autoport:

Deep dive: Port HuggingFace models to Neuron with the Autoport skill
=====================================================================

**Why read this guide?** You are an ML engineer who needs to run a HuggingFace
transformer model on AWS Trainium through NxD Inference. The model
architecture does not have existing support, so you need to port it. The Autoport
skill handles this for you. It is an AI agent workflow that analyzes your model,
builds a Neuron compatible implementation, compiles it, runs inference, and validates
accuracy against the original.

**How to use this guide:** Use this guide when you need to port a model that is not yet available on NxD Inference. Jump to :ref:`workflow steps <autoport-workflow>`
if you already have your environment ready.

You need experience with PyTorch model development and the NxD Inference library
structure to get the most out of this content.

Prerequisites
-------------

Before you start, make sure you understand the following topics.

- **NxD Inference library overview.** How to build and deploy models
  using NxD Inference. See :doc:`../index`.
- **PyTorch model architecture.** Transformer building blocks (attention, MLP,
  embeddings) and how HuggingFace organizes model code.
- **Neuron compilation model.** How ``torch-neuronx`` traces Python code into
  HLO and compiles it to NEFF for NeuronCores. See :ref:`nxdi-feature-guide`.
- **Tensor parallelism concepts.** How models get sharded across NeuronCores.
  See :doc:`/libraries/nxd-inference/app-notes/parallelism`.

Overview
--------

Porting a HuggingFace model to run on Neuron hardware usually takes multiple days
of engineering work. The Autoport skill replaces that manual effort. An AI coding
agent executes the full porting process from analysis through validation.

The skill works with dense transformer models (decoder only and encoder decoder),
Mixture of Experts (MoE) models, models with novel attention mechanisms (sliding window,
grouped query attention, multi latent attention), and models that need weight
dequantization (MXFP4, INT8).

The workflow has six stages.

1. **Knowledge base analysis.** The agent consults a curated library of porting patterns,
   known issues, and architecture specific solutions gathered from prior successful ports.
2. **Architecture analysis.** The agent examines the HuggingFace model code and maps its
   components to existing NxD Inference building blocks.
3. **NeuronX implementation.** The agent creates a Neuron compatible model using NxD Inference
   modules for attention, MLP, MoE, embeddings, and parallelism.
4. **Compilation.** The agent compiles the ported model to NEFF using the Neuron compiler.
5. **Inference testing.** The agent runs the compiled model and checks for coherent output.
6. **Accuracy validation.** The agent compares Neuron output against the HuggingFace reference
   model. A port passes when greedy token match rate reaches 95% or higher.

..
   TODO: Add autoport-workflow.png diagram showing six stages from analysis to validation.
   Place in /libraries/nxd-inference/developer_guides/images/autoport-workflow.png


.. _autoport-prerequisites:

Hardware and software requirements
-----------------------------------

* **Instance type.** ``trn1.32xlarge`` (32 NeuronCores, 16 GB per core) or equivalent
  Trainium instance.
* **Neuron SDK.** Version 2.28+ with the following system packages installed.

  - ``aws-neuronx-dkms``
  - ``aws-neuronx-runtime-lib``
  - ``aws-neuronx-collectives``
  - ``aws-neuronx-tools``

* **Python.** 3.10 or later.
* **Neuron SDK Python packages.**

  - ``neuronx-distributed-inference`` (0.8.x)
  - ``neuronx-distributed`` (0.17.x)
  - ``transformers`` (4.57+)

* **Model weights.** Downloaded from HuggingFace Hub or available locally.
* **Disk space.** You need enough room for model weights, compiled artifacts, and sharded
  checkpoints. Plan for 2x to 5x the model size.

.. note::
   The Autoport skill supports a dry run mode for environments without Neuron hardware.
   In dry run mode the agent performs architecture analysis and code generation but skips
   compilation, inference, and validation. See :ref:`autoport-dry-run`.

.. _autoport-environment-setup:

Environment setup
-----------------

The skill includes a setup script that the agent runs automatically to validate your
environment, create a Python virtual environment, and install required packages.

Setup script location
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   skills/neuron-framework-autoport/scripts/setup_autoport.sh

Usage
^^^^^

.. code-block:: bash

   # Validate and install (creates ./_venv by default)
   bash setup_autoport.sh

   # Use or create venv at a custom path
   bash setup_autoport.sh --venv /path/to/venv

   # Validate only (no install, exits 2 if unusable)
   bash setup_autoport.sh --venv /path/to/venv --validate-only

   # Dry run. Shows what would be installed, asks for consent
   bash setup_autoport.sh --dry-run

Setup sequence
^^^^^^^^^^^^^^

The script runs four steps.

1. **Detect OS.** Identifies the operating system and picks the correct requirements file
   (Ubuntu or Amazon Linux 2023).
2. **Check system packages.** Verifies that Neuron system packages (``aws-neuronx-dkms``,
   ``aws-neuronx-runtime-lib``, ``aws-neuronx-collectives``, ``aws-neuronx-tools``) are installed.
   Exits with code 2 if any are missing.
3. **Resolve Python environment.** Activates an existing venv or creates a new one. Installs
   packages from the requirements file.
4. **Validate.** Runs ``check_env.py`` to confirm that all required packages import
   successfully and resolves the three source paths used throughout the workflow.

On success, the script emits three ``RESOLVED`` lines.

.. code-block:: text

   RESOLVED:NXDI_SRC=/path/to/neuronx_distributed_inference
   RESOLVED:NXD_SRC=/path/to/neuronx_distributed
   RESOLVED:TRANSFORMERS_SRC=/path/to/transformers

The agent uses these paths to navigate framework source code during porting.

Exit codes
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 50

   * - Code
     - Meaning
   * - 0
     - Environment is ready. Resolved paths emitted.
   * - 2
     - Hard failure. Missing system packages, no Python 3.10+, pip failure, or unusable env with validate only flag.
   * - 3
     - Validation still failing after install. Offer alternatives.
   * - 4
     - (dry run only) Install required. Do not proceed without user consent.
   * - 5
     - Runtime error. Packages present but imports fail (GLIBC, driver mismatch). Reinstalling will not fix this.

.. _autoport-parameters:

Porting parameters
------------------

Before the agent begins work, it collects six required parameters from you.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``ModelName``
     - The HuggingFace model class name (for example, ``ArceeForCausalLM``).
   * - ``pathToModelImplementationDirectory``
     - Path to the model source directory containing the HuggingFace implementation.
   * - ``NameOfImplementationFile``
     - The modeling file name (for example, ``modeling_arcee.py``).
   * - ``NameOfConfigurationFile``
     - The configuration file name (for example, ``configuration_arcee.py``).
   * - ``huggingFaceModelID``
     - The HuggingFace model identifier (for example, ``arcee-ai/AFM-4.5B-Base``).
   * - ``pathToModelWeightsDirectory``
     - Path to store or load model weights.

An optional seventh parameter, ``pathToVenv``, specifies an existing virtual environment path.

.. _autoport-workflow:

Workflow
--------

Step 1. Analyze knowledge base and existing implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The agent first consults the knowledge base at ``references/knowledge_base/`` which contains
porting patterns, implementation guides, known issues, and architecture references.

The porting patterns document (``NEURONX_PORTING_GUIDE.md``) covers systematic procedures
for porting transformer models with critical framework patterns. The implementation guide
(``MODEL_IMPLEMENTATION_GUIDE.md``) describes common patterns for attention, MLP, MoE,
and embedding components. The known issues files contain categorized solutions for compilation
errors, sharding problems, accuracy failures, and debugging strategies. The architecture
references document (``EXISTING_MODEL_ARCHITECTURES.md``) provides detailed analysis of all
currently supported model architectures in NxD Inference.

The agent then analyzes the NxD Inference source code at ``${NXDI_SRC}`` to understand
available model implementations in ``models/`` as reference ports, reusable modules in
``modules/`` (attention, MoE, checkpoint loading, padding), and how existing models handle
configuration, compilation, and weight loading.

Similarly, the agent examines ``${NXD_SRC}`` for transformer modules (attention, LoRA, MoE)
in ``modules/``, model specific operators in ``operators/``, RoPE and other positional
encoding overrides in ``overrides/``, and tracing and compilation utilities in ``trace/``.

Step 2. Port the model implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the analysis from Step 1, the agent creates a Neuron compatible implementation.

**Reuse existing components.** The agent maps each HuggingFace model component to an
existing NxD Inference module wherever possible. It only implements components from scratch
when no existing equivalent exists.

**Follow framework patterns exactly.** Every successfully ported model follows specific
required patterns. The most critical is the custom ``NeuronConfig`` subclass.

.. code-block:: python

   class YourModelNeuronConfig(NeuronConfig):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           from .modeling_module import YourModelAttention
           self.attn_cls = YourModelAttention

**Keep names consistent with HuggingFace.** Component names match the original model
class names. When a direct mapping is not possible, the agent appends ``_u`` to indicate
the divergence.

**Implement incrementally.** The agent builds and tests each component (MLP, attention,
decoder layer) individually with a forward pass before moving to the next one.

**Output location.** The ported implementation goes into a ``neuron_port/`` subdirectory
within the project root.

Step 3. Compile the model
^^^^^^^^^^^^^^^^^^^^^^^^^

The agent uses the compilation tool (``scripts/model_compiler.py``) to compile the ported
model to NEFF format.

Compilation configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scripts.model_compiler import DirectModelCompiler, CompilationConfig

   config = CompilationConfig(
       model_class=NeuronYourModel,
       config_class=YourModelInferenceConfig,
       neuron_config_class=NeuronConfig,
       model_path="agent_artifacts/data",
       output_path="agent_artifacts/data/compiled_model",
       batch_size=1,
       seq_len=2048,
       tp_degree=8,
       use_fp16=True,
   )

   compiler = DirectModelCompiler(config)
   success = compiler.compile()

Key compilation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``model_class``
     - Type
     - The Neuron model class to compile.
   * - ``config_class``
     - Type
     - Configuration class (must support ``from_pretrained``).
   * - ``neuron_config_class``
     - Type
     - NeuronConfig subclass for hardware configuration.
   * - ``model_path``
     - str
     - Path to model weights directory.
   * - ``output_path``
     - str
     - Where to save compiled artifacts.
   * - ``batch_size``
     - int
     - Batch size for compilation (default is 1).
   * - ``seq_len``
     - int
     - Maximum sequence length (default is 128).
   * - ``tp_degree``
     - int
     - Tensor parallel degree (number of NeuronCores).
   * - ``ep_degree``
     - int
     - Expert parallel degree for MoE models (default is 1).
   * - ``use_fp16``
     - bool
     - Use bfloat16 precision (default is True).
   * - ``reduce_layers``
     - int
     - Reduce layer count for faster test compilations.
   * - ``on_cpu``
     - bool
     - Compile for CPU execution (testing only).

The compiler automatically sets environment variables for XLA debugging and Neuron
compilation, disables the HLO verifier (workaround for complex models), enables modular
flow for MoE models (GPT OSS pattern), saves ``neuron_config.json`` alongside compiled
artifacts, and preserves HLO artifacts on failure for debugging.

Compilation must succeed before the agent moves to inference.

Step 4. Test inference
^^^^^^^^^^^^^^^^^^^^^^

The agent uses the inference runner (``scripts/run_inference.py``) to verify that the
compiled model produces coherent output.

.. code-block:: python

   from scripts.run_inference import run_inference_with_classes

   success, response, metrics = run_inference_with_classes(
       model_class=NeuronYourModel,
       config_class=YourModelInferenceConfig,
       model_path="agent_artifacts/data",
       compiled_path="agent_artifacts/data/compiled_model",
       prompt="The future of artificial intelligence is",
       max_new_tokens=50,
       temperature=0.0,  # Greedy decoding for determinism
   )

The inference runner loads the compiled model and tokenizer, reconstructs ``NeuronConfig``
from the saved ``neuron_config.json``, runs a manual generation loop with greedy or sampled
decoding, suppresses special tokens to prevent degenerate output, and reports performance
metrics (tokens per second, latency).

The agent only moves to validation after inference produces coherent output.

Step 5. Validate accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^

The validation tool (``scripts/validate_model.py``) compares the Neuron model output against
the HuggingFace reference model to confirm correctness.

Validation configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Create a JSON config file (based on ``assets/example_validation_config.json``).

.. code-block:: json

   {
       "model_name": "your-model",
       "model_class": "neuron_port/modeling_yourmodel.py:NeuronYourModel",
       "config_class": "neuron_port/modeling_yourmodel.py:YourModelInferenceConfig",
       "model_path": "agent_artifacts/data",
       "compiled_model_path": "agent_artifacts/data/compiled_model",
       "tp_degree": 8,
       "num_tokens_to_check": 64,
       "test_parameters": [
           {
               "batch_size": 1,
               "seq_len": 2048
           }
       ]
   }

Running validation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/validate_model.py \
       --config agent_artifacts/tmp/validation_config.json \
       --mode token \
       --batch-size 1 \
       --seq-len 2048

Validation modes
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``token``
     - Greedy token matching (default). Compares argmax token IDs between Neuron and HuggingFace outputs.
   * - ``logit``
     - Distribution comparison. Compares full logit distributions for debugging precision issues.
   * - ``comprehensive``
     - Both token matching and logit comparison plus additional metrics for final validation.

Success criteria
~~~~~~~~~~~~~~~~

The port passes validation when the greedy token match rate reaches 95% or higher. The
validator exits with code 0 on pass and code 1 on failure.

If validation fails, the agent enters an iteration loop. It analyzes the failure (token
divergence positions, logit differences), fixes the implementation code, deletes compiled
artifacts, recompiles, and validates again. The agent does not declare success until
validation passes.

.. code-block:: bash

   rm -rf agent_artifacts/data/compiled_model && rm -rf /var/tmp/neuron-compile-cache

.. _autoport-dry-run:

Dry run mode
------------

When hardware is not available, the agent operates in dry run mode.

In this mode, the agent skips the full dependency resolution step. It activates the venv
and resolves source paths using Python introspection. Steps 1 and 2 (analysis and
implementation) run normally. The agent skips compilation, inference, and validation
because no Neuron hardware is present.

Invoke dry run mode by specifying ``dry-run`` when calling the skill.

.. _autoport-file-organization:

File organization
-----------------

The Autoport skill enforces a strict file organization.

.. code-block:: text

   project_root/
   ├── neuron_port/                  # Ported model implementation (final product)
   │   ├── modeling_yourmodel.py     # Neuron model class
   │   └── configuration_yourmodel.py  # Model config (if needed)
   ├── agent_artifacts/
   │   ├── data/                     # Model weights and compiled output
   │   │   ├── compiled_model/       # NEFF artifacts
   │   │   └── *.safetensors         # Model weights
   │   ├── tmp/                      # Temporary files (compile scripts, test scripts, logs)
   │   │   └── validation_config.json
   │   ├── traces/                   # Audit trail of agent decisions
   │   │   └── port_summary.md       # Final summary
   │   └── results/                  # Validation results (JSON)

.. _autoport-tools-reference:

Tools reference
---------------

Compilation tool
^^^^^^^^^^^^^^^^

**Module.** ``scripts/model_compiler.py``

**Class.** ``DirectModelCompiler``

The compiler accepts model classes directly through ``CompilationConfig``. It does not use CLI
arguments. All configuration happens through the Python API. It handles environment setup
(XLA flags, Neuron compiler flags), model specific optimizations (modular flow for MoE,
blockwise matmul config), Neuron config creation with tensor parallel, expert parallel, and
dtype settings, artifact verification and sharded checkpoint detection, and HLO artifact
preservation on failure for post mortem debugging.

Inference tool
^^^^^^^^^^^^^^

**Module.** ``scripts/run_inference.py``

**Function.** ``run_inference_with_classes(...)``

Supports both hardware and CPU modes. In CPU mode, the model runs without compilation
for rapid iteration during development. Returns a tuple of
``(success: bool, response: str, metrics: dict)``.

Validation tool
^^^^^^^^^^^^^^^

**Module.** ``scripts/validate_model.py``

This is a CLI entry point with three stages.

1. **Smoke test.** Verify the model loads correctly.
2. **Accuracy.** Compare against HuggingFace golden reference (FP32).
3. **Performance.** Measure TTFT and throughput (optional).

The HuggingFace golden model always loads in FP32 to measure actual precision drift
from the Neuron BF16 model.

Environment validator
^^^^^^^^^^^^^^^^^^^^^

**Module.** ``scripts/envSetup/check_env.py``

Validates that all required packages import successfully and emits resolved source paths.
Distinguishes between missing packages (reinstall may help) and runtime errors (system level
issues that reinstalling cannot fix).

.. _autoport-knowledge-base:

Knowledge base
--------------

The skill includes a curated knowledge base at ``references/knowledge_base/`` containing
solutions gathered from prior porting sessions.

Porting patterns
^^^^^^^^^^^^^^^^

- ``NEURONX_PORTING_GUIDE.md`` contains the complete porting procedure with critical framework patterns.
- ``MODEL_IMPLEMENTATION_GUIDE.md`` covers component by component implementation patterns.
- ``NOVEL_NEURONX_PORTING_PATTERNS.md`` describes patterns for unusual architectures.
- ``OVERRIDING_FORWARD_GUIDANCE.md`` explains when and how to override forward methods.

Compilation issues
^^^^^^^^^^^^^^^^^^

- ``Category1_Porting_Config_Compilation_Issues.md`` covers configuration related compilation failures.
- ``compilation_errors_and_fixes.md`` lists common compiler errors with solutions.
- ``MODULAR_FLOW_COMPILATION_SUMMARY.md`` explains modular flow for MoE models.

Sharding and memory
^^^^^^^^^^^^^^^^^^^

- ``Category2_Sharding_Memory_Issues.md`` covers weight sharding and memory management issues.
- ``WEIGHT_SHARDING_FIXES_SUMMARY.md`` lists common weight sharding fixes.

Accuracy debugging
^^^^^^^^^^^^^^^^^^

- ``Category3_Accuracy_Debugging_Analysis.md`` covers diagnosing accuracy failures.
- ``ROOT_CAUSE_REPEATED_OUTPUTS.md`` explains debugging repeated or degenerate outputs.
- ``layernorm_vs_rmsnorm_analysis.md`` discusses normalization layer compatibility.

Architecture specific
^^^^^^^^^^^^^^^^^^^^^

- ``EXISTING_MODEL_ARCHITECTURES.md`` documents all supported architectures.
- ``genericmoe_port_session.md`` walks through a complete MoE porting session.
- ``PORTING_SLIDING_WINDOW.md`` covers sliding window attention porting.

.. _autoport-debugging:

Common issues
-------------

Compilation failures
^^^^^^^^^^^^^^^^^^^^

**JSON parse error** (``[NLA001] Unhandled exception with message: [json.exception.parse_error.101]``).
Delete the compiler cache at ``/var/tmp/neuron-compile-cache`` and retry.

**FileNotFoundError on NEFF output paths.** Delete the compiler cache and retry.

**HLO verifier exit code 70.** The compiler automatically disables the HLO verifier via
``--internal-hlo2tensorizer-options=--verify-hlo=false``. If this still occurs, check for
unsupported operations in the model.

**Vocabulary size not divisible by TP degree.** The compiler warns when ``vocab_size % tp_degree != 0``.
Adjust TP degree to a divisor of the vocabulary size, or add padding in the embedding layer.

Inference failures
^^^^^^^^^^^^^^^^^^

**Token generation produces wrong shapes.** The custom NeuronConfig is missing ``self.attn_cls``.
This is the most common cause of token generation failure.

**Degenerate or repeated output.** Check that special token suppression is enabled and that the
generation loop uses greedy decoding (``temperature=0.0``).

**Model loads but forward pass fails.** Verify that ``neuron_config.json`` in the compiled
directory matches the parameters used during compilation.

Validation failures
^^^^^^^^^^^^^^^^^^^

**Match rate below 95%.** Common causes include incorrect normalization (LayerNorm vs RMSNorm),
wrong RoPE implementation, or weight loading order mismatches. Check the knowledge base
accuracy debugging documents.

**HuggingFace golden model OOM.** Reduce ``num_tokens_to_check`` or use a smaller batch size
for validation.

Ignorable warnings
^^^^^^^^^^^^^^^^^^

``WARNING:Neuron:TP degree (XX) and KV heads (YY) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!``
This is expected behavior and does not indicate a problem.

.. _autoport-project-guidelines:

Project guidelines
------------------

The Autoport skill enforces these guidelines during the porting process.

- No try/except statements. Let errors surface directly for cleaner debugging.
- No additional pip installs. Use only NxD Inference, NxD Core, and PyTorch.
- No modifications to framework code. The ported model must work with the NxD libraries as they are.
- No imports from transformers_neuronx. This is a deprecated library.
- Use print statements instead of Python logging for debugging output.
- Read helper function signatures before calling them. Use only the parameters they accept.

Related resources
-----------------

- :ref:`nxdi-feature-guide` for NxD Inference features configuration reference.
- :ref:`nxdi-onboarding-models` for integrating custom models into NxD Inference.
- :ref:`nxdi-parallelism-user-guide` for tensor and sequence parallelism techniques.
- :ref:`moe-inference-deep-dive` for MoE architecture support in NxD Inference.
- :ref:`nxdi-vllm-user-guide-v1` for deploying models with vLLM on Neuron.
- `NxD Inference source code <https://github.com/aws-neuron/neuronx-distributed-inference>`_ for reference model implementations.
- `Neuron Agentic Development <https://github.com/aws-neuron/neuron-agentic-development>`_ for the Autoport skill source and knowledge base.
