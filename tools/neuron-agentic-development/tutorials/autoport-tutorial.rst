.. meta::
   :description: Tutorial for porting a HuggingFace model to NxD Inference using the Neuron Agentic Development Autoport skill.
   :keywords: autoport, tutorial, model porting, NxD Inference, Neuron Agentic Development, Trainium
   :date-modified: 2026-05-11

.. _autoport-tutorial:

Tutorial: Port a HuggingFace model using the Autoport skill
=============================================================

This tutorial walks you through porting a HuggingFace transformer model to NxD Inference
using the Autoport agent. You will provide model parameters, invoke the agent, and the
agent will handle the rest: analysis, implementation, compilation, inference testing,
and accuracy validation.

By the end of this tutorial you will have a working NxD Inference implementation of
your model that passes a 95% greedy token match against the HuggingFace reference.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites
-------------

Set up a Trainium instance
^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a Trainium instance (``trn1`` or ``trn2``). Launch it from the Neuron Deep
Learning AMI (DLAMI) and SSH in.

Verify Neuron devices are available.

.. code-block:: bash

   neuron-ls

You should see 32 NeuronCores. If you see 0, your instance does not have Neuron hardware
attached. Stop here and fix that first.

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

Download your model weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the HuggingFace model you want to port. For this example we use a small model
to keep compilation fast.

.. code-block:: bash

   mkdir -p agent_artifacts/data
   huggingface-cli download arcee-ai/AFM-4.5B-Base --local-dir agent_artifacts/data

Step 1. Gather your model parameters
--------------------------------------

The Autoport agent needs six pieces of information about your model. Gather these before
you start.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - What it is
     - Example value
   * - ``ModelName``
     - The HuggingFace model class name
     - ``ArceeForCausalLM``
   * - ``pathToModelImplementationDirectory``
     - Path to the HuggingFace model source
     - ``transformers/src/transformers/models/arcee``
   * - ``NameOfImplementationFile``
     - The modeling file
     - ``modeling_arcee.py``
   * - ``NameOfConfigurationFile``
     - The config file
     - ``configuration_arcee.py``
   * - ``huggingFaceModelID``
     - HuggingFace model identifier
     - ``arcee-ai/AFM-4.5B-Base``
   * - ``pathToModelWeightsDirectory``
     - Where model weights live
     - ``agent_artifacts/data``

You can also pass an optional ``pathToVenv`` if you use a non default virtual environment.

To find the model class name, open the modeling file in the HuggingFace transformers
source and look for the main class (usually ``<ModelName>ForCausalLM``).

Step 2. Invoke the Autoport agent
-----------------------------------

Open your agentic IDE (Claude Code or Kiro) on the Trainium instance.

Invoke the agent with your parameters.

.. code-block:: text

   Port with inputs as ModelName is ArceeForCausalLM,
   pathToModelImplementationDirectory is transformers/src/transformers/models/arcee,
   nameOfImplementationFile is modeling_arcee.py,
   nameOfConfigurationFile is configuration_arcee.py,
   huggingFaceModelID is arcee-ai/AFM-4.5B-Base,
   pathToModelWeightsDirectory agent_artifacts/data

The agent confirms your parameters and starts working. You do not need to do anything
else. The agent runs through all six stages automatically.

If you want a dry run (analysis and code generation only, no compilation or hardware), add
``dry-run`` to your request.

Step 3. What happens during the port
--------------------------------------

The agent works through six stages. Here is what you will see at each one.

**Stage 1: Knowledge base analysis.** The agent reads its internal porting guides and
known issues database. It identifies patterns relevant to your model architecture. This
takes a few seconds.

**Stage 2: Architecture analysis.** The agent reads the HuggingFace model source code
and maps each component (attention, MLP, embeddings) to existing NxD Inference modules.
It identifies what can be reused and what needs custom implementation.

**Stage 3: Implementation.** The agent writes the Neuron compatible model code. It creates
files in a ``neuron_port/`` directory. You can watch the code appear in real time.

**Stage 4: Compilation.** The agent compiles the model to NEFF format using the Neuron
compiler. This is the longest step and can take 10 to 30 minutes depending on model size.
The agent sets ``tp_degree=8`` by default (8 NeuronCores). You will see compiler output
scroll by.

**Stage 5: Inference testing.** The agent loads the compiled model and generates text.
It verifies the output is coherent (not garbage or repeated tokens).

**Stage 6: Accuracy validation.** The agent compares Neuron model output against the
HuggingFace reference model loaded in FP32. It checks 64 tokens with greedy decoding.
The port passes when match rate reaches 95% or higher.

If validation fails, the agent automatically iterates. It analyzes what diverged, fixes
the code, recompiles, and validates again. It does not stop until it passes.

Step 4. Check the results
--------------------------

When the agent finishes, you will have these outputs.

.. code-block:: text

   project_root/
   ├── neuron_port/
   │   └── modeling_yourmodel.py     # The ported implementation
   ├── agent_artifacts/
   │   ├── data/compiled_model/      # Compiled NEFF artifacts
   │   ├── traces/port_summary.md    # Summary of decisions made
   │   └── results/                  # Validation JSON results

The ported model in ``neuron_port/`` is the final product. You can use it directly with
NxD Inference.

Step 5. Deploy with vLLM (optional)
-------------------------------------

Once you have a validated port, you can serve it with vLLM. See the
:ref:`vLLM User Guide <nxdi-vllm-user-guide-v1>` for deployment instructions.

Troubleshooting
---------------

The agent handles most issues automatically, but here are things that might require
your input.

**Agent asks for missing parameters.** If you forgot a parameter, the agent will ask
for it. Provide the value and it continues.

**Compilation takes too long.** Large models (70B+) can take 30 to 60 minutes to compile.
This is normal. You can reduce compilation time by setting a smaller ``seq_len`` for testing.

**Agent cannot find model weights.** Make sure ``pathToModelWeightsDirectory`` points to
a directory containing the actual weight files (``.safetensors`` or ``.bin``).

**Validation keeps failing.** If the agent iterates more than 3 times on validation,
check the ``agent_artifacts/traces/`` directory for the agent's analysis of what is
diverging. Common causes include wrong normalization layers (LayerNorm vs RMSNorm),
incorrect RoPE implementation, or weight loading mismatches.

**No NeuronCores detected.** Run ``neuron-ls``. If it shows 0 devices, your instance
either does not have Neuron hardware or the driver is not loaded. Check that
``aws-neuronx-dkms`` is installed.

Related resources
-----------------

- :doc:`/tools/neuron-agentic-development/developer_guides/neuron-framework-autoport` for the deep dive on how the Autoport skill works internally.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- :ref:`nxdi-vllm-user-guide-v1` for deploying ported models with vLLM.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_ for source code and issue tracker.
