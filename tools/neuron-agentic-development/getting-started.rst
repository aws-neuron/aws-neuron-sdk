.. meta::
   :description: Get started with Neuron Agentic Development. Install the package, set up your environment, and run your first agent on Trainium.
   :keywords: Neuron Agentic Development, getting started, install, setup, Claude Code, Kiro, Trainium
   :date-modified: 2026-05-11

.. _neuron-agentic-development-getting-started:

===============
Getting Started
===============

This guide walks you through installing Neuron Agentic Development and running your
first agent on a Trainium instance.

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 30 30 30

   * - #
     - Requirement
     - Details
     - Needed for
   * - 1
     - Trainium instance
     - ``trn1`` or ``trn2`` EC2 instance (AL2023 DLAMI recommended)
     - Compiling, profiling, model porting
   * - 2
     - Neuron SDK
     - ``aws-neuronx-tools`` (pre installed on DLAMI)
     - All on device skills
   * - 3
     - Python venv with Neuron packages
     - ``neuronx-cc``, ``torch-neuronx``, ``neuron-explorer``
     - Compilation, profiling, analysis
   * - 4
     - Claude Code or Kiro
     - Installed on the Trainium instance
     - Running agents and skills
   * - 5
     - Anthropic API key
     - For Claude model inference
     - Agent reasoning

.. note::
   The agent runs on the same machine as the hardware. There is no laptop to remote
   file transfer step. Everything is co located. Writing and documentation skills work
   anywhere (no hardware needed), but profiling, debugging, and model porting require
   on instance execution.

Step 1. Launch and verify your Trainium instance
-------------------------------------------------

Launch an instance using the `Neuron Deep Learning AMI (DLAMI) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/dlami.html>`_,
then SSH into it.

Verify Neuron devices are visible.

.. code-block:: bash

   neuron-ls

   which neuron-explorer && neuron-explorer --version

Step 2. Activate your Python environment
------------------------------------------

The DLAMI comes with a pre installed virtual environment.

.. code-block:: bash

   source ~/opt/aws_neuronx_venv_pytorch_2_9/bin/activate

Step 3. Install Neuron Agentic Development
-------------------------------------------

Install from the Neuron PyPI repository.

.. code-block:: bash

   pip install --upgrade neuron-agentic-development \
       --extra-index-url https://pip.repos.neuron.amazonaws.com

Or clone from GitHub if you want to customize or contribute.

.. code-block:: bash

   git clone https://github.com/aws-neuron/neuron-agentic-development.git
   cd neuron-agentic-development
   pip install .

Then deploy to your preferred tool.

.. code-block:: bash

   # For Claude Code
   deploy-neuron-agentic-development-to-claude

   # For Kiro
   deploy-neuron-agentic-development-to-kiro

Step 4. Install your agentic IDE (if not already installed)
------------------------------------------------------------

For Kiro.

.. code-block:: bash

   curl -fsSL https://cli.kiro.dev/install | bash

For Claude Code, follow the `Claude Code installation guide <https://docs.anthropic.com/en/docs/claude-code/overview>`_.

Step 5. Run your first agent
-----------------------------

Start the unified NKI agent.

.. code-block:: bash

   kiro-cli chat --agent neuron-nki-agent

The ``neuron-nki-agent`` is the main entry point. It picks the right workflow based on
your request and orchestrates the appropriate skills.

Example prompts
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - What you want to do
     - What to say
     - Hardware needed
   * - Write a new kernel
     - "Write a fused softmax kernel for bf16 inputs"
     - No
   * - Debug a compilation error
     - "Fix this kernel" (with error output)
     - Yes
   * - Profile a kernel
     - "Profile my kernel and show me the metrics"
     - Yes
   * - Port a model to Neuron
     - "Port ArceeForCausalLM to NxD Inference" (with parameters)
     - Yes

Next steps
----------

- :doc:`/tools/neuron-agentic-development/tutorials/index` for step by step walkthroughs of specific tasks.
- :doc:`/tools/neuron-agentic-development/developer_guides/index` for deep dives on how skills work internally.
- `Neuron Agentic Development GitHub <https://github.com/aws-neuron/neuron-agentic-development>`_ for the full source, skill catalog, and issue tracker.
