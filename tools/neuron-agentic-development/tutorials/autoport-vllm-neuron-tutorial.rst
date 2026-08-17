.. meta::
   :description: Tutorial for porting a GPU-compatible model to the vLLM Neuron Trainium2 backend using the Neuron Agentic Development Autoport vLLM Neuron skill.
   :keywords: autoport, vllm, vllm-neuron, tutorial, model porting, Trainium2, trn2, Neuron Agentic Development
   :date-modified: 2026-08-13

.. _autoport-vllm-neuron-tutorial:

Tutorial: Port a GPU-compatible model to vLLM Neuron using the Autoport skill
=============================================================================

This tutorial walks you through porting a GPU-compatible transformer model to the
vLLM Neuron Trainium2 backend using the Autoport vLLM Neuron agent. You provide a
model name and a source model ID, invoke the agent, and the agent handles the rest:
architecture research, code generation, model registration, and validation.

By the end of this tutorial you will have a model registered with vLLM Neuron that
serves through the vLLM OpenAI API server and passes logit and equivalence
validation against the GPU-compatible reference implementation.

.. note::
   This tutorial ports a model to the **vLLM Neuron** serving backend. If you want to
   port to NxD Inference and compile to NEFF instead, see
   :ref:`autoport-tutorial`. For background on the two skills, see the
   :ref:`Autoport vLLM Neuron deep dive <neuron-framework-autoport-vllm-neuron>`.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites
-------------

Set up a Trainium2 instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a ``trn2.48xlarge`` instance. The vLLM Neuron backend targets Trainium2.
Launch the instance from the Neuron Deep Learning AMI (DLAMI) and SSH in.

Verify that Neuron devices are available.

.. code-block:: bash

   neuron-ls

If you see 0 devices, your instance does not have Neuron hardware attached or the
driver is not loaded. Stop here and fix that first.

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

Verify the vLLM Neuron package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The skill needs the ``vllm_neuron`` (or ``private_vllm_neuron``) package in your
workspace. If it is not present, the agent stops without porting.

.. code-block:: bash

   python3 -c "import vllm_neuron; print(vllm_neuron.__file__)"

Download your model weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the model that you want to port. For this example, use a small model to
keep compilation and validation fast.

.. code-block:: bash

   huggingface-cli download 01-ai/Yi-6B-Chat --local-dir ~/models/yi-6b-chat

Step 1. Choose your model name and ID
-------------------------------------

The Autoport vLLM Neuron agent needs two values: a short snake_case name for the
port and the source model ID.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Argument
     - What it is
     - Example value
   * - ``model-name``
     - The snake_case name for your port. Becomes the directory under
       ``vllm_neuron/model/``.
     - ``yi``
   * - ``hf-model-id``
     - The source model identifier.
     - ``01-ai/Yi-6B-Chat``

The agent derives the PascalCase class prefix from your model name (``yi`` becomes
``Yi``). Pick a name that matches the convention of the existing models in
``vllm_neuron/model/`` (for example, ``llama3``, ``qwen3_moe``, ``deepseek_v2``).

Step 2. Invoke the Autoport agent
---------------------------------

Open your agentic IDE (Claude Code or Kiro) on the Trainium2 instance.

Invoke the skill with your model name and source model ID.

.. code-block:: text

   Port the model yi from 01-ai/Yi-6B-Chat to vLLM Neuron.

The agent confirms the inputs and starts working. You do not need to do anything
else.

If you want to review the agent's architecture decisions before it writes any model
code, add the ``--review`` flag. The agent pauses after the architecture analysis
(Step 3 below) and waits for your confirmation.

.. code-block:: text

   Port the model yi from 01-ai/Yi-6B-Chat to vLLM Neuron with --review.

If you want a dry run (research and code generation only, no hardware), add
``dry-run`` to your request.

Step 3. What happens during the port
------------------------------------

The agent works through three phases. Here is what you will see.

**Phase A: Research and analysis.** The agent fetches the source model config, reads
the modeling source, inspects the checkpoint weight keys, and computes valid tensor
parallel sizes. It then picks the closest existing vLLM Neuron model as a reference
(for Yi, a standard GQA + RoPE model, this is ``llama3/``) and prints an architecture
analysis table comparing the two. If you passed ``--review``, it pauses here.

**Phase B: Code generation.** The agent generates the model package under
``vllm_neuron/model/yi/`` (``config.py``, ``factory.py``, ``__init__.py``, and
``model.py``), registers the model in ``vllm_neuron/model/registry.py``, and creates
an example run script and a model README. You can watch the code appear in real time.

**Phase C: Validation.** The agent runs four validation steps in order.

1. *Smoke test.* Offline inference through the example script. The agent checks that
   weights load, the model compiles, and the generated text is coherent.
2. *Online serving.* The agent starts the vLLM OpenAI API server and runs completion,
   batch, streaming, and counting tests, then re-runs them with prefix caching.
3. *Logit validation.* The agent compares Neuron logits against a CPU reference
   implementation and reports sigma values across top-k levels.
4. *Deep equivalence validation.* The agent invokes the Equivalence skill with the
   vLLM Neuron adapter for component-level and end-to-end verification.

Step 4. Check the results
-------------------------

When the agent finishes, you will have these outputs.

.. code-block:: text

   vllm_neuron/model/yi/
   ├── __init__.py
   ├── config.py
   ├── factory.py
   ├── model.py             # The ported implementation
   └── README.md            # Architecture and feature status

   vllm_neuron/model/registry.py            # Modified: yi registered

   examples/vllm_neuron/models/yi/
   ├── run.py
   └── results/results.md   # Offline and online serving results

   test/vllm_neuron/model/yi/bf16/e2e/
   └── test_logits.py       # Logit validation test

The model in ``vllm_neuron/model/yi/`` is the final product, and the registry entry
makes it available to vLLM. Review ``results/results.md`` for the inference and
serving numbers, and the README for the feature status table.

Step 5. Serve your model
-------------------------

Once the port is registered and validated, serve it through the vLLM OpenAI API
server.

.. code-block:: bash

   NEURON_SKIP_EFA_AFFINITY=1 python3 -m vllm.entrypoints.openai.api_server \
       --model 01-ai/Yi-6B-Chat \
       --tensor-parallel-size 8 \
       --max-model-len 256

Send a test completion.

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "01-ai/Yi-6B-Chat", "prompt": "The capital of France is", "max_tokens": 16}'

Use the tensor parallel size that the agent recommended in Phase A. See
:ref:`nxdi-vllm-user-guide-v1` for the full set of serving options.

Troubleshooting
---------------

The agent handles most issues automatically, but here are things that might require
your input.

**Agent stops because it cannot find vllm_neuron.** The ``vllm_neuron`` or
``private_vllm_neuron`` package must be present in your workspace. Confirm with
``python3 -c "import vllm_neuron"`` and check that your virtual environment is
activated.

**Out of memory during weight loading or warmup** (``nrt_tensor_allocate
status=4``). The model does not fit at the chosen tensor parallel size. Ask the agent
to increase the TP size. MoE models are especially memory-hungry and may need expert
parallelism.

**Degenerate output (repeated tokens).** This usually points to a decode path bug
(bias shape, KV cache, or normalization). The agent diagnoses this by testing prefill
only first, then debugging the decode path.

**Logit validation reports high sigma.** This is not a hard failure. The agent
reports the sigma values and which top-k levels diverged and continues to completion.
Decide whether the accuracy is acceptable for your use case, and consult the
equivalence report for component-level detail.

**No EFA device found.** PCI topology mismatch on trn2. The agent runs with
``NEURON_SKIP_EFA_AFFINITY=1``, which is safe for single-node tensor parallelism.

**No NeuronCores detected.** Run ``neuron-ls``. If it shows 0 devices, your instance
either does not have Neuron hardware or the driver is not loaded. Check that
``aws-neuronx-dkms`` is installed.

Related resources
-----------------

- :ref:`neuron-framework-autoport-vllm-neuron` for the deep dive on how the Autoport
  vLLM Neuron skill works internally.
- :ref:`autoport-tutorial` for porting a model to NxD Inference instead of
  vLLM Neuron.
- :ref:`nxdi-vllm-user-guide-v1` for serving models with vLLM on Neuron.
- :doc:`/tools/neuron-agentic-development/getting-started` for installation and setup.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_
  for source code and issue tracker.
